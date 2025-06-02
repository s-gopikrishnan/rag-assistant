# 📚 AI Assistant for Enterprise Document Processing

A powerful Retrieval-Augmented Generation (RAG) system that enables you to chat with your documents using local LLMs through Ollama, ChromaDB for vector storage, and a beautiful Streamlit interface.

## 🌟 Features

- **Local LLM Integration**: Uses Ollama for privacy-focused document processing
- **Multi-format Support**: Process PDF, DOCX, and PPTX files
- **Intelligent Chunking**: Preserves document structure and context
- **Vector Search**: Powered by ChromaDB with sentence transformers
- **Interactive Chat**: Beautiful Streamlit interface for document Q&A
- **Real-time Analytics**: Performance metrics and database statistics
- **Debug Tools**: Built-in debugging and system monitoring

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   Document       │    │     Ollama      │
│   Frontend      │────│   Processor      │────│   (Local LLM)   │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌──────────────────┐
                    │    ChromaDB      │
                    │ (Vector Storage) │
                    └──────────────────┘
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Ollama installed and running**

### 1. Install Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the required model
ollama pull llama3.2

# Start Ollama service
ollama serve
```

### 2. Install Python Dependencies

```bash
# Clone the repository
git clone <your-repo-url>
cd rag-document-processor

# Install dependencies
pip install -r requirements.txt

# Optional: Install spaCy language model
python -m spacy download en_core_web_sm
```

### 3. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📋 Usage Guide

### Step 1: Initialize the System
1. Configure settings in the sidebar:
   - **Llama URL**: Usually `http://localhost:11434`
   - **Model Name**: Choose from available models (llama3.2, gemma3:4b)
   - **Database Path**: Local ChromaDB storage location
   - **Embedding Model**: Sentence transformer for embeddings

2. Click **"Initialize RAG System"**

### Step 2: Upload Documents
1. Go to the **"Upload Documents"** tab
2. Select your PDF, DOCX, or PPTX files
3. Click **"Process Documents"**
4. Wait for processing to complete

### Step 3: Start Chatting
1. Navigate to the **"Chat"** tab
2. Ask questions about your documents
3. View responses with source citations
4. Monitor performance metrics

## 🔧 Configuration Options

### Query Settings
- **Max Context Chunks**: Number of relevant chunks to retrieve (1-10)
- **Min Similarity**: Threshold for relevance filtering (0.0-1.0)
- **Max Response Tokens**: Maximum length of LLM responses (128-2048)
- **Temperature**: Creativity level of responses (0.0-1.0)

### Supported Models
- **Llama 3.2**: Default recommended model
- **Gemma 3:4B**: Alternative lightweight option
- Custom models available through Ollama

### Embedding Models
- **all-MiniLM-L6-v2**: Default, fast and efficient
- **BAAI/bge-base-en**: High quality English embeddings
- **all-mpnet-base-v2**: Best overall performance
- **multi-qa-MiniLM-L6-cos-v1**: Optimized for Q&A tasks

## 📊 Features Overview

### Document Processing
- **PDF**: Text extraction with table support using PyMuPDF and pdfplumber
- **DOCX**: Section-aware processing with table extraction
- **PPTX**: Slide-by-slide content extraction with titles

### Vector Database
- **ChromaDB**: Persistent local storage
- **Embeddings**: Local sentence transformers (no API calls)
- **Metadata**: Rich document metadata preservation
- **Search**: Semantic similarity search with configurable thresholds

### Chat Interface
- **Context-Aware**: Uses relevant document chunks
- **Source Citations**: Shows which documents informed each answer
- **Chat History**: Maintains conversation context
- **Performance Metrics**: Response time and relevance scores

### Debug Tools
- **Database Inspector**: View stored document chunks
- **Retrieval Tester**: Test search functionality
- **Embedding Checker**: Verify model functionality
- **Statistics Dashboard**: Document and usage analytics

## 📁 Project Structure

```
├── app.py                      # Main Streamlit application
├── llama_rag_system.py        # Core RAG system implementation
├── local_doc_extractor.py     # Document processing and chunking
├── requirements.txt           # Python dependencies
├── chroma_db/                 # ChromaDB storage (created automatically)
└── temp_uploads/              # Temporary file storage (created automatically)
```

## 🔍 Core Components

### LlamaRAGSystem
- Handles LLM communication with Ollama
- Manages ChromaDB vector storage
- Implements semantic search and retrieval
- Provides debugging and monitoring tools

### LocalDocumentExtractor
- Multi-format document processing
- Intelligent content chunking
- Metadata preservation
- Table extraction support

### Streamlit Interface
- User-friendly web interface
- Real-time processing feedback
- Interactive configuration
- Performance monitoring

## 🛠️ Advanced Configuration

### Custom Ollama Models
```bash
# Pull additional models
ollama pull codellama
ollama pull mistral

# Use in the application by updating model_name
```

### Database Customization
```python
# Modify in llama_rag_system.py
client = chromadb.PersistentClient(
    path="./custom_db_path",
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True
    )
)
```

### Embedding Model Selection
Choose based on your needs:
- **Speed**: all-MiniLM-L6-v2
- **Quality**: all-mpnet-base-v2
- **Multilingual**: paraphrase-multilingual-MiniLM-L12-v2

## 🚨 Troubleshooting

### Common Issues

**1. Ollama Connection Failed**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama service
ollama serve
```

**2. Model Not Found**
```bash
# List available models
ollama list

# Pull required model
ollama pull llama3.2
```

**3. ChromaDB Issues**
- Delete the `chroma_db` directory to reset
- Check disk space for database storage
- Ensure proper write permissions

**4. Memory Issues**
- Reduce `max_context_chunks` setting
- Use smaller embedding models
- Process fewer documents simultaneously

### Performance Optimization

**For Better Speed:**
- Use smaller embedding models
- Reduce chunk size and overlap
- Lower similarity thresholds
- Use faster Ollama models

**For Better Quality:**
- Use larger embedding models
- Increase context chunks
- Higher similarity thresholds
- More sophisticated chunking strategies

## 📈 Usage Examples

### Example Queries
- "What technical skills are mentioned in the resumes?"
- "Summarize the key requirements from the RFP document"
- "What are the main topics covered in the presentation?"
- "Find information about project timelines"

### Sample Workflow
1. Upload company documents (resumes, proposals, reports)
2. Ask: "What are the common skills across all candidates?"
3. Follow up: "Which candidate has the most relevant experience?"
4. Review source citations for verification

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Please check the license file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review Ollama documentation
3. Check ChromaDB documentation
4. Create an issue in the repository

## 🔮 Future Enhancements

- [ ] Support for more document formats (TXT, CSV, etc.)
- [ ] Multi-language support
- [ ] Advanced chunking strategies
- [ ] Integration with cloud LLM providers
- [ ] Batch processing capabilities
- [ ] Advanced analytics and reporting
- [ ] API endpoint for programmatic access

---

**Built with ❤️ using Ollama, ChromaDB, and Streamlit**
