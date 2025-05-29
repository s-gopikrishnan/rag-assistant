import requests
import json
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import time
import traceback

@dataclass
class RAGResponse:
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    context_used: str
    response_time: float

class LlamaRAGSystem:
    def __init__(self, 
                 llama_base_url: str = "http://localhost:11434",
                 model_name: str = "llama3.2",
                 db_path: str = "./chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG system with local Llama and ChromaDB
        """
        
        self.llama_url = f"{llama_base_url}/api/generate"
        self.model_name = model_name
        self.embedding_model_name = embedding_model
        
        # Test Llama connection
        self._test_llama_connection()
        
        # Initialize ChromaDB
        print(f"Initializing ChromaDB at: {db_path}")
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"description": "Document chunks for RAG system"}
        )
        
        # Debug: Check existing data
        existing_count = self.collection.count()
        print(f"Found {existing_count} existing chunks in database")
        
        print("RAG system initialized successfully!")
    
    def get_existing_count(self):
        return self.collection.count()

    def _test_llama_connection(self):
        """Test connection to local Llama instance"""
        try:
            response = requests.post(
                self.llama_url,
                json={
                    "model": self.model_name,
                    "prompt": "Hello",
                    "stream": False
                },
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"✓ Connected to Llama 3.2 at {self.llama_url}")
            else:
                raise Exception(f"HTTP {response.status_code}")
                
        except Exception as e:
            print(f"❌ Failed to connect to Llama: {e}")
            print("Please ensure Ollama is running with: ollama serve")
            print(f"And that {self.model_name} is installed: ollama pull {self.model_name}")
            raise
    
    def query_llama(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        """Query local Llama model"""
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "top_k": 40
            }
        }
        
        try:
            response = requests.post(self.llama_url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '').strip()
            
        except requests.exceptions.Timeout:
            return "Error: Request timed out. The model might be processing a complex query."
        except Exception as e:
            return f"Error querying Llama: {str(e)}"
    
    def retrieve_context(self, query: str, n_results: int = 5, min_similarity: float = 0.3) -> List[Dict[str, Any]]:
        """Retrieve relevant context from ChromaDB with enhanced debugging"""
        
        print(f"🔍 Retrieving context for query: '{query}'")
        print(f"   Requested results: {n_results}, Min similarity: {min_similarity}")
        
        try:
            # Check if database has any data
            total_chunks = self.collection.count()
            print(f"   Total chunks in database: {total_chunks}")
            
            if total_chunks == 0:
                print("❌ No chunks found in database!")
                return []
            
            # Generate query embedding
            print("   Generating query embedding...")
            query_embedding = self.embedding_model.encode([query]).tolist()[0]
            print(f"   Query embedding shape: {len(query_embedding)}")
            
            # Search in ChromaDB
            print("   Searching ChromaDB...")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, total_chunks),  # Don't request more than available
                include=["documents", "metadatas", "distances"]
            )
            
            print(f"   Raw results returned: {len(results['documents'][0])}")
            
            # Debug: Print raw distances
            if results['distances'][0]:
                distances = results['distances'][0]
                similarities = [1 - d for d in distances]
                print(f"   Raw similarities: {[f'{s:.3f}' for s in similarities]}")
            
            # Filter by minimum similarity and format results
            relevant_chunks = []
            
            for i, (doc, meta, distance, chunk_id) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0], 
                results['distances'][0],
                results['ids'][0]
            )):
                similarity_score = 1 - distance
                
                print(f"   Chunk {i+1}: Similarity = {similarity_score:.3f}, Above threshold = {similarity_score >= min_similarity}")
                
                if similarity_score >= min_similarity:
                    relevant_chunks.append({
                        'content': doc,
                        'metadata': meta,
                        'similarity_score': similarity_score,
                        'chunk_id': chunk_id
                    })
                    print(f"     ✓ Added chunk from: {meta.get('source_file', 'Unknown')}")
                    print(f"     Content preview: {doc[:100]}...")
            
            print(f"   Final relevant chunks: {len(relevant_chunks)}")
            return relevant_chunks
            
        except Exception as e:
            print(f"❌ Error in retrieve_context: {str(e)}")
            print(f"   Exception type: {type(e).__name__}")
            print(f"   Traceback: {traceback.format_exc()}")
            return []
    
    def build_context_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Build a comprehensive context prompt for Llama"""
        
        if not context_chunks:
            print("⚠️  No context chunks available, using knowledge-only prompt")
            return f"""You are a helpful assistant. Please answer the following question based on your knowledge:

Question: {query}

Answer:"""
        
        print(f"📝 Building context prompt with {len(context_chunks)} chunks")
        
        # Group context by document type and source
        context_by_source = {}
        for chunk in context_chunks:
            source = chunk['metadata'].get('source_file', 'Unknown')
            doc_type = chunk['metadata'].get('doc_type', 'document')
            
            if source not in context_by_source:
                context_by_source[source] = {
                    'type': doc_type,
                    'chunks': []
                }
            context_by_source[source]['chunks'].append(chunk)
        
        # Build context sections
        context_sections = []
        for source, data in context_by_source.items():
            filename = source.split('/')[-1] if '/' in source else source
            
            section = f"=== From {data['type'].upper()}: {filename} ==="
            
            for i, chunk in enumerate(data['chunks'], 1):
                # Add metadata context
                meta_info = []
                if 'page_number' in chunk['metadata']:
                    meta_info.append(f"Page {chunk['metadata']['page_number']}")
                elif 'slide_number' in chunk['metadata']:
                    meta_info.append(f"Slide {chunk['metadata']['slide_number']}")
                elif 'section_title' in chunk['metadata']:
                    meta_info.append(f"Section: {chunk['metadata']['section_title']}")
                
                meta_str = f" ({', '.join(meta_info)})" if meta_info else ""
                
                section += f"\n\n[Context {i}{meta_str}]:\n{chunk['content']}"
            
            context_sections.append(section)
        
        context_text = "\n\n" + "\n\n".join(context_sections)
        
        prompt = f"""You are a knowledgeable assistant helping to answer questions based on provided documents. Use the context information below to provide accurate, helpful answers.

CONTEXT INFORMATION:
{context_text}

INSTRUCTIONS:
- Base your answer primarily on the provided context
- If the context doesn't contain enough information, clearly state what's missing
- Reference specific documents/pages when relevant
- Be concise but comprehensive
- If you're making inferences beyond the context, clearly indicate this

QUESTION: {query}

ANSWER:"""
        
        return prompt
    
    def ask(self, 
            query: str, 
            max_context_chunks: int = 5,
            min_similarity: float = 0.3,
            max_tokens: int = 512,
            temperature: float = 0.1) -> RAGResponse:
        """Main RAG query method with enhanced debugging"""
        
        start_time = time.time()
        
        print(f"\n🤔 Processing query: '{query}'")
        print(f"   Parameters: max_chunks={max_context_chunks}, min_sim={min_similarity}")
        
        # Step 1: Retrieve relevant context
        context_chunks = self.retrieve_context(
            query=query,
            n_results=max_context_chunks,
            min_similarity=min_similarity
        )
        
        print(f"📄 Found {len(context_chunks)} relevant chunks")
        
        # Step 2: Build prompt with context
        prompt = self.build_context_prompt(query, context_chunks)
        
        # Debug: Show prompt length
        print(f"📝 Generated prompt length: {len(prompt)} characters")
        
        print(f"🤖 Querying Llama 3.2...")
        
        # Step 3: Query Llama
        answer = self.query_llama(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        response_time = time.time() - start_time
        
        print(f"✅ Response generated in {response_time:.2f}s")
        
        # Format sources for response
        sources = []
        for chunk in context_chunks:
            source_info = {
                'filename': chunk['metadata'].get('source_file', 'Unknown').split('/')[-1],
                'doc_type': chunk['metadata'].get('doc_type', 'document'),
                'similarity_score': chunk['similarity_score'],
                'content_preview': chunk['content'][:200] + "..." if len(chunk['content']) > 200 else chunk['content']
            }
            
            # Add location info
            if 'page_number' in chunk['metadata']:
                source_info['location'] = f"Page {chunk['metadata']['page_number']}"
            elif 'slide_number' in chunk['metadata']:
                source_info['location'] = f"Slide {chunk['metadata']['slide_number']}"
            elif 'section_title' in chunk['metadata']:
                source_info['location'] = f"Section: {chunk['metadata']['section_title']}"
            else:
                source_info['location'] = "Document content"
            
            sources.append(source_info)
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            query=query,
            context_used=prompt,
            response_time=response_time
        )
    
    def add_documents(self, chunks: List) -> None:
        """Add document chunks to the database with better error handling"""
        if not chunks:
            print("⚠️  No chunks provided to add")
            return
        
        print(f"Adding {len(chunks)} chunks to database...")
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            if not chunk.content.strip():  # Skip empty chunks
                print(f"   Skipping empty chunk {i+1}")
                continue
                
            documents.append(chunk.content)
            
            # Debug: Show first few chunks being added
            if i < 3:
                print(f"   Chunk {i+1} preview: {chunk.content[:100]}...")
                print(f"   Chunk {i+1} metadata: {chunk.metadata}")
            
            # Convert metadata to strings (ChromaDB requirement)
            metadata = {}
            for key, value in chunk.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = str(value)
                else:
                    metadata[key] = json.dumps(value)
            
            metadatas.append(metadata)
            ids.append(chunk.chunk_id)
        
        if not documents:
            print("❌ No valid documents to add after filtering")
            return
        
        try:
            # Generate embeddings locally
            print("Generating embeddings...")
            embeddings = self.embedding_model.encode(documents, show_progress_bar=True).tolist()
            print(f"Generated {len(embeddings)} embeddings")
            
            # Add to ChromaDB in batches
            batch_size = 100
            total_batches = (len(documents) - 1) // batch_size + 1
            
            for i in range(0, len(documents), batch_size):
                batch_end = min(i + batch_size, len(documents))
                current_batch = i // batch_size + 1
                
                try:
                    self.collection.add(
                        documents=documents[i:batch_end],
                        embeddings=embeddings[i:batch_end],
                        metadatas=metadatas[i:batch_end],
                        ids=ids[i:batch_end]
                    )
                    
                    print(f"✓ Added batch {current_batch}/{total_batches}")
                    
                except Exception as batch_error:
                    print(f"❌ Error adding batch {current_batch}: {str(batch_error)}")
                    # Continue with other batches
            
            # Verify addition
            final_count = self.collection.count()
            print(f"Database now contains {final_count} total chunks")
            
        except Exception as e:
            print(f"❌ Error during document addition: {str(e)}")
            print(f"   Exception type: {type(e).__name__}")
            print(f"   Traceback: {traceback.format_exc()}")
    
    def debug_database_content(self, limit: int = 5):
        """Debug method to inspect database content"""
        print(f"\n🔍 Database Debug Information:")
        
        try:
            count = self.collection.count()
            print(f"   Total chunks: {count}")
            
            if count == 0:
                print("   ❌ Database is empty!")
                return
            
            # Get sample data
            sample = self.collection.get(
                limit=min(limit, count),
                include=["documents", "metadatas"]
            )
            
            print(f"   Sample {len(sample['documents'])} chunks:")
            for i, (doc, meta, chunk_id) in enumerate(zip(
                sample['documents'],
                sample['metadatas'],
                sample['ids']
            )):
                print(f"   Chunk {i+1}:")
                print(f"     ID: {chunk_id}")
                print(f"     Source: {meta.get('source_file', 'Unknown')}")
                print(f"     Type: {meta.get('doc_type', 'Unknown')}")
                print(f"     Content: {doc[:100]}...")
                print()
                
        except Exception as e:
            traceback.print_exc()
            print(f"   ❌ Error inspecting database: {str(e)}")
            print(f"   Exception type: {type(e).__name__}")
            print(f"   Traceback: {traceback.format_exc()}")
    
    def test_retrieval(self, test_query: str = "test"):
        """Test retrieval with a simple query"""
        msg = f"\n🧪 Testing retrieval with query: '{test_query}'"
        print(f"\n🧪 Testing retrieval with query: '{test_query}'")
        
        try:
            # Try with very low similarity threshold
            results = self.retrieve_context(test_query, n_results=3, min_similarity=0.0)
            print(f"   Results with min_similarity=0.0: {len(results)}")
            msg += f"\n   Results with min_similarity=0.0: {len(results)}"
            if results:
                print("   ✅ Retrieval working!")
                msg += "\n   ✅ Retrieval working!"
                for i, result in enumerate(results[:2]):
                    print(f"   Result {i+1}: Score={result['similarity_score']:.3f}")
                    msg += f"\n   Result {i+1}: Score={result['similarity_score']:.3f}"
            else:
                print("   ❌ No results even with similarity=0.0")
                msg += "\n   ❌ No results even with similarity=0.0"
                
        except Exception as e:
            print(f"   ❌ Retrieval test failed: {str(e)}")
            msg += "\n   ❌ Retrieval test failed: {str(e)}"
        return msg
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        count = self.collection.count()
        
        if count == 0:
            return {
                'total_chunks': 0,
                'document_types': {},
                'unique_sources': 0,
                'sample_sources': []
            }
        
        # Get sample to analyze metadata
        sample = self.collection.get(limit=min(100, count), include=["metadatas"])
        
        doc_types = {}
        sources = set()
        
        for metadata in sample['metadatas']:
            doc_type = metadata.get('doc_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            source = metadata.get('source_file', '')
            if source:
                sources.add(source.split('/')[-1])  # Just filename
        
        return {
            'total_chunks': count,
            'document_types': doc_types,
            'unique_sources': len(sources),
            'sample_sources': list(sources)[:10]  # First 10 sources
        }