import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import json
from sentence_transformers import SentenceTransformer
import numpy as np

# Import our custom classes (assuming they're in separate files)
from local_doc_extractor import LocalDocumentExtractor, process_document_folder, DocumentChunk

class LocalRAGDatabase:
    def __init__(self, db_path: str = "./chroma_db", embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize ChromaDB with local embeddings"""
        
        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(
            path=db_path
        )
        
        # Initialize local embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"description": "Document chunks for RAG system"}
        )
    
    def add_document_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to ChromaDB"""
        if not chunks:
            return
        
        print(f"Adding {len(chunks)} chunks to database...")
        
        # Prepare data for ChromaDB
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            documents.append(chunk.content)
            
            # Convert metadata to strings (ChromaDB requirement)
            metadata = {}
            for key, value in chunk.metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    metadata[key] = str(value)
                else:
                    metadata[key] = json.dumps(value)
            
            metadatas.append(metadata)
            ids.append(chunk.chunk_id)
        
        # Generate embeddings locally
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Add to ChromaDB in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            
            self.collection.add(
                documents=documents[i:batch_end],
                embeddings=embeddings[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
            )
            
            print(f"Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
        
        print(f"Successfully added {len(chunks)} chunks to database")
    
    def search(self, query: str, n_results: int = 5, filter_metadata: Dict = None) -> Dict[str, Any]:
        """Search for relevant chunks"""
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )
        
        return {
            'query': query,
            'results': [
                {
                    'content': doc,
                    'metadata': meta,
                    'similarity_score': 1 - distance,  # Convert distance to similarity
                    'chunk_id': chunk_id
                }
                for doc, meta, distance, chunk_id in zip(
                    results['documents'][0],
                    results['metadatas'][0], 
                    results['distances'][0],
                    results['ids'][0]
                )
            ]
        }
    
    def search_by_document_type(self, query: str, doc_type: str, n_results: int = 5) -> Dict[str, Any]:
        """Search within specific document type"""
        return self.search(
            query=query,
            n_results=n_results,
            filter_metadata={"doc_type": doc_type}
        )
    
    def search_by_source(self, query: str, source_file: str, n_results: int = 5) -> Dict[str, Any]:
        """Search within specific source file"""
        return self.search(
            query=query,
            n_results=n_results,
            filter_metadata={"source_file": {"$contains": source_file}}
        )
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        count = self.collection.count()
        
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
    
    def delete_by_source(self, source_file: str) -> int:
        """Delete all chunks from a specific source file"""
        # Get IDs of chunks from this source
        results = self.collection.get(
            where={"source_file": {"$contains": source_file}},
            include=["ids"]
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            return len(results['ids'])
        
        return 0

# Complete workflow example
def complete_rag_workflow(documents_folder: str, db_path: str = "./chroma_db"):
    """Complete workflow from documents to searchable RAG database"""
    
    print("Step 1: Extracting documents...")
    extractor = LocalDocumentExtractor()
    all_chunks = process_document_folder(documents_folder)
    
    print(f"Step 2: Extracted {len(all_chunks)} total chunks")
    
    print("Step 3: Initializing RAG database...")
    rag_db = LocalRAGDatabase(db_path)
    
    print("Step 4: Adding chunks to ChromaDB...")
    rag_db.add_document_chunks(all_chunks)
    
    print("Step 5: Database ready!")
    stats = rag_db.get_collection_stats()
    print("Database statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return rag_db

# Example usage and testing
if __name__ == "__main__":
    # Initialize database
    rag_db = LocalRAGDatabase()
    
    # Example: Add some chunks (you would get these from document extraction)
    # rag_db.add_document_chunks(chunks)
    
    # Example searches
    queries = [
        "software engineering experience",
        "project management skills", 
        "technical requirements",
        "budget and timeline"
    ]
    
    print("Example searches:")
    for query in queries:
        results = rag_db.search(query, n_results=3)
        print(f"\nQuery: {query}")
        
        for i, result in enumerate(results['results'], 1):
            print(f"  Result {i} (Score: {result['similarity_score']:.3f}):")
            print(f"    Source: {result['metadata'].get('source_file', 'Unknown')}")
            print(f"    Type: {result['metadata'].get('doc_type', 'Unknown')}")
            print(f"    Content: {result['content'][:150]}...")
    
    # Show database statistics
    print(f"\nDatabase Statistics:")
    stats = rag_db.get_collection_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")