import streamlit as st
import os
import time
from pathlib import Path
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Import our custom classes (assuming they're in separate files)
from local_doc_extractor import LocalDocumentExtractor, process_document_folder
from llama_rag_system import LlamaRAGSystem

# Page configuration
st.set_page_config(
    page_title="Local RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .source-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #28a745;
    }
    
    .query-response {
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
    }
    
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False

def initialize_rag_system():
    """Initialize RAG system with error handling"""
    try:
        with st.spinner("Initializing RAG system..."):
            rag_system = LlamaRAGSystem(
                llama_base_url=st.session_state.llama_url,
                model_name=st.session_state.model_name,
                db_path=st.session_state.db_path,
                embedding_model=st.session_state.embedding_model
            )
            st.session_state.rag_system = rag_system
            st.success("✅ RAG system initialized successfully!")
            return True
    except Exception as e:
        st.error(f"❌ Failed to initialize RAG system: {str(e)}")
        return False

def process_uploaded_files(uploaded_files):
    """Process uploaded files"""
    if not uploaded_files:
        return
    
    # Create temporary directory
    temp_dir = Path("./temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    # Save uploaded files
    saved_files = []
    for uploaded_file in uploaded_files:
        print('Uploaded file:', uploaded_file.name)
        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_files.append(file_path)
    
    # Process documents
    try:
        with st.spinner(f"Processing {len(saved_files)} documents..."):
            extractor = LocalDocumentExtractor()
            all_chunks = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file_path in enumerate(saved_files):
                status_text.text(f"Processing: {file_path.name}")
                
                try:
                    chunks = extractor.extract_document(file_path)
                    all_chunks.extend(chunks)
                    st.write(f"✅ {file_path.name}: {len(chunks)} chunks extracted")
                except Exception as e:
                    st.error(f"❌ Error processing {file_path.name}: {str(e)}")
                
                progress_bar.progress((i + 1) / len(saved_files))
            
            # Add to RAG system
            if all_chunks and st.session_state.rag_system:
                status_text.text("Adding chunks to database...")
                st.session_state.rag_system.add_documents(all_chunks)
                st.session_state.documents_processed = True
                st.success(f"🎉 Successfully processed {len(all_chunks)} chunks from {len(saved_files)} documents!")
            
            progress_bar.empty()
            status_text.empty()
            
    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
    
    finally:
        # Cleanup temp files
        for file_path in saved_files:
            try:
                file_path.unlink()
            except:
                pass

def display_chat_interface():
    """Display chat interface"""
    st.markdown("### 💬 Chat with your Documents")
    
    # Chat history
    if st.session_state.chat_history:
        st.markdown("#### Previous Conversations")
        for i, (query, response) in enumerate(st.session_state.chat_history[-5:]):  # Show last 5
            with st.expander(f"Q: {query[:60]}..." if len(query) > 60 else f"Q: {query}"):
                st.markdown(f"**Question:** {query}")
                st.markdown(f"**Answer:** {response.answer}")
                
                if response.sources:
                    st.markdown("**Sources:**")
                    for source in response.sources:
                        st.markdown(f"- {source['filename']} ({source['location']}) - Score: {source['similarity_score']:.3f}")
    
    # New query
    with st.form("chat_form"):
        query = st.text_area(
            "Ask a question about your documents:",
            placeholder="e.g., What technical skills are mentioned in the profiles?",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submit = st.form_submit_button("Ask Question", type="primary")
        
        with col2:
            clear_history = st.form_submit_button("Clear History")
    
    if clear_history:
        st.session_state.chat_history = []
        st.rerun()
    
    if submit and query.strip():
        if not st.session_state.rag_system:
            st.error("Please initialize the RAG system first!")
            return
        
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.rag_system.ask(
                    query=query,
                    max_context_chunks=st.session_state.max_context_chunks,
                    min_similarity=st.session_state.min_similarity,
                    max_tokens=st.session_state.max_tokens,
                    temperature=st.session_state.temperature
                )
                
                # Display response
                st.markdown("### 🤖 Response")
                st.markdown(f'<div class="query-response">{response.answer}</div>', unsafe_allow_html=True)
                
                # Display sources
                if response.sources:
                    st.markdown("### 📚 Sources")
                    for i, source in enumerate(response.sources, 1):
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>{i}. {source['filename']}</strong><br>
                            <em>{source['location']} | Similarity: {source['similarity_score']:.3f}</em><br>
                            <small>{source['content_preview']}</small>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Add to history
                st.session_state.chat_history.append((query, response))
                
                # Performance metrics
                st.markdown("### ⚡ Performance")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Response Time", f"{response.response_time:.2f}s")
                with col2:
                    st.metric("Sources Found", len(response.sources))
                with col3:
                    st.metric("Context Quality", f"{max([s['similarity_score'] for s in response.sources], default=0):.3f}")
                
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
# Add these to your Streamlit app in the Statistics tab

def display_debug_tools():
    """Display debugging tools"""
    st.markdown("### 🔧 Debug Tools")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Debug Database Content"):
            if st.session_state.rag_system:
                with st.expander("Database Content Debug", expanded=True):
                    # Capture debug output
                    import io
                    import sys
                    from contextlib import redirect_stdout
                    
                    f = io.StringIO()
                    with redirect_stdout(f):
                        st.session_state.rag_system.debug_database_content()
                    output = f.getvalue()
                    
                    st.code(output)
            else:
                st.error("Initialize RAG system first")
    
    with col2:
        search_text = st.text_input(label="Enter a Value to search", placeholder="Enter a quick search text")
        if st.button("Test Simple Retrieval"):
            if st.session_state.rag_system:
                #with st.expander("Retrieval Test Results", expanded=True):
                    #f = io.StringIO()
                    #with redirect_stdout(f):
                print("search_text: ", search_text)
                print("Search Button clicked")
                output = st.session_state.rag_system.test_retrieval(search_text)
                #output = f.getvalue()
            
                st.code(output)
            else:
                st.error("Initialize RAG system first")
    
    with col3:
        if st.button("Check Embedding Model"):
            if st.session_state.rag_system:
                st.write(f"**Embedding Model:** {st.session_state.rag_system.embedding_model_name}")
                st.write(f"**Model Loaded:** {st.session_state.rag_system.embedding_model is not None}")
                
                # Test embedding generation
                try:
                    test_embedding = st.session_state.rag_system.embedding_model.encode(["test"])
                    st.success(f"✅ Embedding generation works. Shape: {test_embedding.shape}")
                except Exception as e:
                    st.error(f"❌ Embedding generation failed: {e}")

def display_database_stats():
    """Display database statistics"""
    if not st.session_state.rag_system:
        st.warning("Initialize RAG system to view statistics")
        return
    
    stats = st.session_state.rag_system.get_stats()
    
    if stats['total_chunks'] == 0:
        st.info("No documents in database. Upload some documents to get started!")
        return
    
    st.markdown("### 📊 Database Statistics")
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><h3>{stats["total_chunks"]}</h3><p>Total Chunks</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h3>{stats["unique_sources"]}</h3><p>Documents</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h3>{len(stats["document_types"])}</h3><p>File Types</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><h3>{len(st.session_state.chat_history)}</h3><p>Queries Asked</p></div>', unsafe_allow_html=True)
    
    # Document types chart
    if stats['document_types']:
        st.markdown("#### Document Types Distribution")
        doc_types_df = pd.DataFrame([
            {'Type': k.upper(), 'Count': v} 
            for k, v in stats['document_types'].items()
        ])
        
        fig = px.pie(doc_types_df, values='Count', names='Type', 
                    title="Documents by Type")
        st.plotly_chart(fig, use_container_width=True)
    
    # Sources list
    if stats['sample_sources']:
        st.markdown("#### Recent Documents")
        for source in stats['sample_sources']:
            st.markdown(f"- {source}")

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">📚 Local RAG System</h1>', unsafe_allow_html=True)
    st.markdown("Chat with your documents using local Llama 3.2 and ChromaDB")
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # RAG System Settings
        st.markdown("### RAG System")
        st.session_state.llama_url = st.text_input("Llama URL", value="http://localhost:11434")
        st.session_state.model_name = st.text_input("Model Name", value="llama3.2")
        st.session_state.db_path = st.text_input("Database Path", value="./chroma_db")
        st.session_state.embedding_model = st.selectbox(
            "Embedding Model",
            ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "multi-qa-MiniLM-L6-cos-v1"]
        )
        
        if st.button("Initialize RAG System"):
            initialize_rag_system()
        
        # Query Settings
        st.markdown("### Query Settings")
        st.session_state.max_context_chunks = st.slider("Max Context Chunks", 1, 10, 5)
        st.session_state.min_similarity = st.slider("Min Similarity", 0.0, 1.0, 0.3, 0.1)
        st.session_state.max_tokens = st.slider("Max Response Tokens", 128, 2048, 512)
        st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.1)
        
        # System Status
        st.markdown("### System Status")
        if st.session_state.rag_system:
            st.success("✅ RAG System Ready")
        else:
            st.error("❌ RAG System Not Initialized")
        
        if st.session_state.documents_processed:
            st.success("✅ Documents Loaded")
        else:
            st.warning("⚠️ No Documents Loaded")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload Documents", "💬 Chat", "📊 Statistics", "🔧 System Info"])
    
    with tab1:
        st.markdown("### Upload Documents")
        st.markdown("Supported formats: PDF, DOCX, PPTX")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'docx', 'pptx'],
            accept_multiple_files=True,
            help="Upload your profile documents, presentations, RFPs, etc."
        )
        
        if st.button("Process Documents", disabled=not uploaded_files or not st.session_state.rag_system):
            process_uploaded_files(uploaded_files)
        
        # Processing instructions
        with st.expander("📖 How to use"):
            st.markdown("""
            1. **Initialize RAG System**: Configure settings in sidebar and click "Initialize RAG System"
            2. **Upload Documents**: Select your PDF, DOCX, or PPTX files
            3. **Process**: Click "Process Documents" to extract and index content
            4. **Chat**: Go to Chat tab to ask questions about your documents
            
            **Tips:**
            - Larger documents will take longer to process
            - The system extracts text, tables, and maintains document structure
            - Each document is chunked intelligently to preserve context
            """)
    
    with tab2:
        if st.session_state.rag_system and st.session_state.documents_processed:
            display_chat_interface()
        else:
            st.info("Please initialize the RAG system and upload documents first!")
            
            # Quick start guide
            st.markdown("### 🚀 Quick Start")
            st.markdown("""
            1. Go to **Upload Documents** tab
            2. Initialize the RAG system using sidebar settings
            3. Upload your documents (PDF, DOCX, PPTX)
            4. Process the documents
            5. Return here to start chatting!
            """)
    
    with tab3:
        display_database_stats()
        display_debug_tools()  # Add this line
    
    with tab4:
        st.markdown("### 🔧 System Information")
        
        # System requirements
        st.markdown("#### Requirements")
        st.markdown("""
        - **Ollama**: Running with Llama 3.2 model
        - **Python**: 3.8+ with required packages
        - **Storage**: Local ChromaDB for document embeddings
        - **Memory**: Sufficient RAM for embedding model and document processing
        """)
        
        # Installation guide
        st.markdown("#### Installation")
        with st.expander("Setup Instructions"):
            st.code("""
# Install Ollama and pull Llama 3.2
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2

# Install Python dependencies
pip install streamlit chromadb sentence-transformers
pip install pymupdf python-docx python-pptx pdfplumber
pip install plotly pandas requests

# Run the application
streamlit run app.py
            """)
        
        # Current configuration
        st.markdown("#### Current Configuration")
        config_data = {
            "Llama URL": st.session_state.get('llama_url', 'Not set'),
            "Model": st.session_state.get('model_name', 'Not set'),
            "Database": st.session_state.get('db_path', 'Not set'),
            "Embeddings": st.session_state.get('embedding_model', 'Not set'),
        }
        
        for key, value in config_data.items():
            st.text(f"{key}: {value}")

if __name__ == "__main__":
    main()