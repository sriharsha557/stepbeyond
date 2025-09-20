# StepBeyond - Career Counseling Assistant for Indian Students

## 📋 Overview

StepBeyond is an AI-powered career counseling assistant designed to help Indian students navigate their journey to study abroad. The Phase 1 MVP uses RAG (Retrieval-Augmented Generation) with FAISS vector database and CrewAI agents powered by Groq's fast LLM API.

## 🏗️ Architecture

- **LLM**: Groq API with `mixtral-8x7b-instruct` (free, fast, long context)
- **Agents**: CrewAI orchestrating Retriever Agent + Answer Agent
- **RAG**: FAISS vector database with sentence-transformers embeddings
- **UI**: Streamlit frontend with clean, intuitive design

## 📁 Project Structure

```
stepbeyond/
├── data/               # PDF or text docs (you need to add files here)
├── embeddings/         # FAISS index files (auto-generated)
├── app.py              # Streamlit UI
├── crew_agents.py      # CrewAI agent definitions
├── rag_pipeline.py     # FAISS + retrieval logic
├── requirements.txt    # dependencies
├── README.md          # this file
└── .env               # environment variables (create this)
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Create project directory
mkdir stepbeyond
cd stepbeyond

# Copy all the generated files to this directory

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file in the project root:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

**Get your free Groq API key:**
1. Visit https://console.groq.com/
2. Sign up for a free account
3. Generate an API key
4. Add it to your `.env` file

### 3. Add Knowledge Base Data

Create a `data/` folder and add your documents:

```bash
mkdir data
```

Add PDF or TXT files to the `data/` folder, such as:
- Canada student visa guide.pdf
- UK university application process.txt
- US scholarship opportunities.pdf
- Australia study requirements.pdf

### 4. Run the Application

```bash
streamlit run app.py
```

The app will automatically:
- Load your documents
- Create embeddings
- Build FAISS index
- Initialize CrewAI agents
- Launch the web interface

## 📖 Usage

### First Time Setup

1. **Add Documents**: Place PDF/TXT files in the `data/` folder
2. **Launch App**: Run `streamlit run app.py`
3. **Build Knowledge Base**: Click "Rebuild Knowledge Base" in the sidebar
4. **Start Asking**: Type your questions and get AI-powered answers!

### Example Questions

- "What are the requirements for a student visa to Canada?"
- "How do I apply for universities in the UK?"
- "What scholarships are available for Indian students in the US?"
- "What documents do I need for IELTS?"
- "How to write a strong Statement of Purpose?"

### Features

- **Intelligent Retrieval**: Finds relevant information from your knowledge base
- **Step-by-Step Guidance**: Breaks down complex processes into actionable steps
- **Chat History**: Keep track of your previous questions and answers
- **Real-time Processing**: Fast responses powered by Groq's optimized LLM
- **Fallback Handling**: Multiple response strategies for reliability

## 🛠️ Technical Details

### RAG Pipeline (`rag_pipeline.py`)

- **Document Loading**: Supports PDF and TXT files
- **Text Splitting**: 500 tokens per chunk, 50 token overlap
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS with L2 distance similarity
- **Persistence**: Saves/loads index and metadata to disk

### CrewAI Agents (`crew_agents.py`)

- **Retriever Agent**: Finds relevant information from knowledge base
- **Answer Agent**: Generates comprehensive, step-by-step responses
- **LLM Integration**: Groq's mixtral-8x7b-instruct model
- **Error Handling**: Fallback to quick-answer mode if crew fails

### Streamlit UI (`app.py`)

- **Clean Design**: Professional, student-friendly interface
- **System Status**: Real-time monitoring of components
- **Chat History**: Persistent conversation tracking
- **Example Queries**: Quick-start buttons for common questions

## 🔧 Troubleshooting

### Common Issues

**"GROQ_API_KEY not found"**
- Ensure you've created a `.env` file with your API key
- Check that the key is correct and active

**"No existing knowledge base found"**
- Add PDF or TXT files to the `data/` folder
- Click "Rebuild Knowledge Base" in the sidebar
- Wait for the process to complete

**"Error processing query"**
- Check your internet connection
- Verify your Groq API key is valid
- Try the query again (the app has fallback mechanisms)

**Slow responses**
- Groq is usually very fast, check your internet connection
- Large knowledge bases may take longer to process
- Consider reducing the number of chunks retrieved (modify `max_chunks` parameter)

### Performance Optimization

- **Document Size**: Keep individual documents under 10MB for faster processing
- **Chunk Size**: Adjust `chunk_size` and `chunk_overlap` in `RAGPipeline` for better retrieval
- **Retrieval Count**: Modify `max_chunks` parameter to balance accuracy vs speed

## 🗺️ Roadmap

### Phase 2: Enhanced Filtering
- Country-specific filters in Streamlit sidebar
- Course/field-specific guidance
- University ranking integration

### Phase 3: API & Frontend
- FastAPI backend for better scalability
- React frontend for improved UX
- User authentication and personalization

### Phase 4: Advanced Features
- Multi-language support (Hindi, regional languages)
- Document upload via UI
- Integration with university APIs
- Personalized recommendation engine

## 🤝 Contributing

This is a Phase 1 MVP. Future contributions welcome for:
- Additional document sources
- UI/UX improvements
- Performance optimizations
- New agent capabilities

## 📄 License

This project is for educational and demonstration purposes. Please ensure you have proper licensing for any documents added to your knowledge base.

## 🆘 Support

For issues and questions:
1. Check this README first
2. Review the error messages in the Streamlit interface
3. Check the console output for detailed logging
4. Ensure all dependencies are properly installed

---

**Built with ❤️ for Indian students pursuing their dreams of studying abroad**
