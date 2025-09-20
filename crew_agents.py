import os
from typing import Dict, Any
from crewai import Agent, Task, Crew, Process
from crewai.llm import LLM
from rag_pipeline import RAGPipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StepBeyondCrew:
    def __init__(self, groq_api_key: str, model_name: str = "groq/llama-3.3-70b-versatile"):
        """
        Initialize StepBeyond CrewAI agents with Groq LLM
        
        Args:
            groq_api_key: Groq API key
            model_name: Groq model to use. Options:
                       - groq/llama-3.3-70b-versatile (recommended)
                       - groq/llama-3.1-8b-instant (faster, lighter)
                       - groq/gemma2-9b-it (alternative)
        """
        self.groq_api_key = groq_api_key
        self.model_name = model_name
        
        # Initialize RAG pipeline
        self.rag = RAGPipeline()
        self.rag.load_index()  # Load existing index if available
        
        # Initialize Groq LLM with error handling
        try:
            self.llm = LLM(
                model=model_name,
                api_key=groq_api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            logger.info(f"Initialized Groq LLM with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM with {model_name}: {str(e)}")
            # Fallback to faster model
            self.model_name = "groq/llama-3.1-8b-instant"
            self.llm = LLM(
                model=self.model_name,
                api_key=groq_api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            logger.info(f"Fallback: Using model {self.model_name}")
        
        # Initialize agents
        self._setup_agents()
        
    def _setup_agents(self):
        """Setup CrewAI agents"""
        
        # Retriever Agent - Responsible for finding relevant information
        self.retriever_agent = Agent(
            role='Information Retriever',
            goal='Find the most relevant information from the knowledge base to answer student queries about studying abroad',
            backstory="""You are an expert information retriever specialized in helping Indian students 
            who want to study abroad. You have access to comprehensive guides about visa processes, 
            university applications, scholarships, and country-specific requirements. Your job is to 
            identify and retrieve the most relevant pieces of information that can help answer the student's question.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Answer Agent - Responsible for generating comprehensive answers
        self.answer_agent = Agent(
            role='Career Counselor',
            goal='Provide comprehensive, step-by-step guidance to Indian students planning to study abroad',
            backstory="""You are an experienced career counselor and education consultant who has helped 
            hundreds of Indian students successfully navigate their journey to study abroad. You specialize 
            in breaking down complex processes into clear, actionable steps. You understand the unique 
            challenges faced by Indian students and provide culturally relevant advice while being encouraging 
            and supportive.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

    def create_retrieval_task(self, query: str) -> Task:
        """Create task for retrieving relevant information"""
        return Task(
            description=f"""
            Search the knowledge base for information relevant to this student query: "{query}"
            
            Your task is to:
            1. Identify key topics and keywords in the query
            2. Retrieve the most relevant document chunks from the knowledge base
            3. Summarize what information was found and how it relates to the query
            4. Return the relevant context that will help answer the student's question
            
            Focus on finding information about:
            - Visa requirements and processes
            - University application procedures
            - Documentation needed
            - Timeline and deadlines
            - Country-specific requirements
            - Scholarship opportunities
            - Any other relevant guidance
            """,
            agent=self.retriever_agent,
            expected_output="A summary of relevant information found in the knowledge base with key details that address the student's query."
        )

    def create_answer_task(self, query: str, context: str) -> Task:
        """Create task for generating comprehensive answer"""
        return Task(
            description=f"""
            Based on the retrieved information, provide a comprehensive answer to this student query: "{query}"
            
            Retrieved Context:
            {context}
            
            Your task is to:
            1. Analyze the student's question and the retrieved context
            2. Create a detailed, step-by-step response that directly addresses their needs
            3. Structure your answer in a clear, easy-to-follow format
            4. Include specific requirements, documents, timelines, and procedures
            5. Provide practical tips and advice for Indian students
            6. Mention any important deadlines or critical information
            7. Be encouraging and supportive in your tone
            
            Structure your response with:
            - Clear headings and sections
            - Step-by-step instructions where applicable
            - Important notes and warnings
            - Additional tips and recommendations
            
            Remember: You're helping an Indian student navigate studying abroad. Be specific, practical, and supportive.
            """,
            agent=self.answer_agent,
            expected_output="A comprehensive, well-structured answer that provides step-by-step guidance to help the Indian student with their query about studying abroad."
        )

    def process_query(self, user_query: str) -> str:
        """
        Process user query through the crew workflow
        """
        try:
            logger.info(f"Processing query: {user_query}")
            
            # First, get relevant context from RAG
            context = self.rag.get_context_for_query(user_query, max_chunks=5)
            
            if not context or context == "No relevant information found.":
                # If no relevant context found, still try to answer with general knowledge
                context = "No specific information found in knowledge base. Please use your general knowledge about studying abroad for Indian students."
                logger.warning("No relevant context found in knowledge base")
            
            # Create tasks
            retrieval_task = self.create_retrieval_task(user_query)
            answer_task = self.create_answer_task(user_query, context)
            
            # Create crew
            crew = Crew(
                agents=[self.retriever_agent, self.answer_agent],
                tasks=[retrieval_task, answer_task],
                verbose=True,
                process=Process.sequential
            )
            
            # Execute crew workflow
            logger.info("Starting crew execution...")
            result = crew.kickoff()
            
            logger.info("Crew execution completed successfully")
            return str(result)
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"I apologize, but I encountered an error while processing your query. Please try rephrasing your question or contact support. Error: {str(e)}"

    def quick_answer(self, user_query: str) -> str:
        """
        Generate a quick answer using just the RAG context and LLM
        (Alternative method if CrewAI has issues)
        """
        try:
            logger.info(f"Generating quick answer for: {user_query}")
            
            # Get context from RAG
            context = self.rag.get_context_for_query(user_query, max_chunks=3)
            
            # Create a simple prompt
            prompt = f"""
            You are an experienced career counselor helping Indian students study abroad. 
            
            Student Question: {user_query}
            
            Relevant Information from Knowledge Base:
            {context}
            
            Please provide a comprehensive, step-by-step answer to help this Indian student. 
            Structure your response with clear headings and actionable advice.
            Be specific about requirements, documents, and procedures.
            """
            
            # Generate response using LLM directly
            response = self.llm.call(prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in quick_answer: {str(e)}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again later."

# Utility functions
def setup_crew(groq_api_key: str = None, model_name: str = None) -> StepBeyondCrew:
    """
    Setup and return StepBeyond crew instance
    
    Args:
        groq_api_key: Groq API key (if None, will get from environment)
        model_name: Model to use (if None, will use default)
    """
    if not groq_api_key:
        groq_api_key = os.getenv('GROQ_API_KEY')
        
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found. Please set it in your environment variables or pass it directly.")
    
    # Use environment variable for model if available
    if not model_name:
        model_name = os.getenv('LLM_MODEL', 'groq/llama-3.3-70b-versatile')
    
    return StepBeyondCrew(groq_api_key, model_name)

if __name__ == "__main__":
    # Example usage
    import dotenv
    dotenv.load_dotenv()
    
    # Initialize crew
    crew = setup_crew()
    
    # Test queries
    test_queries = [
        "What are the requirements for a student visa to Canada?",
        "How do I apply for universities in the UK?",
        "What scholarships are available for Indian students in the US?"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        
        # Try both methods
        try:
            answer = crew.process_query(query)
            print("CrewAI Answer:")
            print(answer)
        except:
            print("CrewAI failed, trying quick answer:")
            answer = crew.quick_answer(query)
            print(answer)