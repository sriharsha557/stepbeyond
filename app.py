import streamlit as st
import os
import time
from dotenv import load_dotenv
from crew_agents import setup_crew
from rag_pipeline import RAGPipeline
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="StepBeyond - Career Counseling Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .query-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .answer-box {
        background-color: #ffffff;
        border-left: 4px solid #2E86AB;
        border-radius: 5px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .info-box {
        background-color: #e8f4fd;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'crew' not in st.session_state:
        st.session_state.crew = None
    if 'rag' not in st.session_state:
        st.session_state.rag = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def setup_system():
    """Setup RAG and CrewAI system"""
    try:
        # Check for GROQ API key
        groq_api_key = os.getenv('GROQ_API_KEY')
        if not groq_api_key:
            st.error("⚠️ GROQ_API_KEY not found in environment variables. Please set it up.")
            st.info("""
            To set up your GROQ API key:
            1. Get your free API key from https://console.groq.com/
            2. Create a `.env` file in your project directory
            3. Add: `GROQ_API_KEY=your_api_key_here`
            """)
            return False
        
        # Initialize RAG if not already done
        if st.session_state.rag is None:
            with st.spinner("🔄 Initializing knowledge base..."):
                st.session_state.rag = RAGPipeline()
                
                # Try to load existing index
                if not st.session_state.rag.load_index():
                    st.warning("📚 No existing knowledge base found. Please add documents to the 'data' folder and run ingestion.")
                    st.info("""
                    To set up your knowledge base:
                    1. Create a 'data' folder in your project directory
                    2. Add PDF or TXT files (e.g., visa guides, university info)
                    3. Click 'Rebuild Knowledge Base' button below
                    """)
        
        # Initialize CrewAI if not already done
        if st.session_state.crew is None:
            with st.spinner("🤖 Initializing AI agents..."):
                st.session_state.crew = setup_crew(groq_api_key)
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error setting up system: {str(e)}")
        return False

def rebuild_knowledge_base():
    """Rebuild the knowledge base from data folder"""
    try:
        with st.spinner("🔄 Rebuilding knowledge base... This may take a few minutes."):
            if st.session_state.rag is None:
                st.session_state.rag = RAGPipeline()
            
            st.session_state.rag.ingest_data()
            st.success("✅ Knowledge base rebuilt successfully!")
            
            # Reinitialize crew with new knowledge base
            st.session_state.crew = None
            setup_system()
            
    except Exception as e:
        st.error(f"❌ Error rebuilding knowledge base: {str(e)}")

def process_user_query(query: str):
    """Process user query and return response"""
    try:
        if not st.session_state.crew:
            return "❌ System not initialized. Please check your setup."
        
        # Add to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # Process query
        with st.spinner("🤔 Thinking... This may take a moment."):
            start_time = time.time()
            
            try:
                # Try CrewAI first
                response = st.session_state.crew.process_query(query)
            except Exception as e:
                logger.warning(f"CrewAI failed, trying quick answer: {str(e)}")
                # Fallback to quick answer
                response = st.session_state.crew.quick_answer(query)
            
            processing_time = time.time() - start_time
        
        # Add response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        return response, processing_time
        
    except Exception as e:
        error_msg = f"❌ Error processing query: {str(e)}"
        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
        return error_msg, 0

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🎓 StepBeyond</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Your AI-powered career counseling assistant for studying abroad</p>', unsafe_allow_html=True)
    
    # System setup
    if not setup_system():
        st.stop()
    
    # Sidebar for system controls
    with st.sidebar:
        st.header("🔧 System Controls")
        
        if st.button("🔄 Rebuild Knowledge Base", help="Rebuild the knowledge base from files in the data folder"):
            rebuild_knowledge_base()
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
        
        # System status
        st.header("📊 System Status")
        if st.session_state.rag and st.session_state.rag.index:
            st.success(f"✅ Knowledge Base: {st.session_state.rag.index.ntotal} documents")
        else:
            st.warning("⚠️ Knowledge Base: Not loaded")
            
        if st.session_state.crew:
            st.success("✅ AI Agents: Ready")
        else:
            st.warning("⚠️ AI Agents: Not initialized")
    
    # Main content area
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        # Query input
        st.markdown('<div class="query-box">', unsafe_allow_html=True)
        
        user_query = st.text_area(
            "💬 What would you like to know about studying abroad?",
            placeholder="e.g., What are the requirements for a student visa to Canada?\nHow do I apply for universities in the UK?\nWhat scholarships are available for Indian students?",
            height=100,
            key="user_input"
        )
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            ask_button = st.button("🚀 Ask StepBeyond", type="primary", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process query
        if ask_button and user_query.strip():
            response, processing_time = process_user_query(user_query.strip())
            
            # Display response
            st.markdown('<div class="answer-box">', unsafe_allow_html=True)
            st.markdown("### 🎯 StepBeyond's Guidance:")
            st.markdown(response)
            if processing_time > 0:
                st.caption(f"⏱️ Response generated in {processing_time:.2f} seconds")
            st.markdown('</div>', unsafe_allow_html=True)
        
        elif ask_button and not user_query.strip():
            st.warning("⚠️ Please enter a question before asking!")
    
    # Chat history
    if st.session_state.chat_history:
        st.header("💬 Chat History")
        
        for i, message in enumerate(reversed(st.session_state.chat_history[-6:])):  # Show last 6 messages
            if message["role"] == "user":
                st.markdown(f"**🧑‍🎓 You:** {message['content']}")
            else:
                with st.expander(f"🤖 StepBeyond Response {len(st.session_state.chat_history) - i}", expanded=False):
                    st.markdown(message['content'])
    
    # Example queries
    st.header("💡 Example Questions")
    col1, col2, col3 = st.columns(3)
    
    example_queries = [
        "What documents do I need for a UK student visa?",
        "How to get scholarships for studying in the US?",
        "Best universities in Canada for engineering?",
        "IELTS requirements for Australian universities",
        "How to write a strong SOP?",
        "Work permit rules for international students"
    ]
    
    for i, example in enumerate(example_queries):
        col = [col1, col2, col3][i % 3]
        with col:
            if st.button(f"📝 {example}", key=f"example_{i}", use_container_width=True):
                st.session_state.user_input = example
                st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>🚀 <strong>StepBeyond Phase 1 MVP</strong> - Empowering Indian students to achieve their dreams of studying abroad</p>
            <p>Built with ❤️ using Streamlit, CrewAI, FAISS, and Groq AI</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()