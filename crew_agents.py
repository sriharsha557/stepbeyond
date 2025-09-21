import os
from typing import Dict, Any
import httpx
from qdrant_rag_pipeline import QdrantRAGPipeline
from loguru import logger

class StepBeyondAgent:
    """
    Simplified StepBeyond agent using direct LLM calls instead of CrewAI
    This eliminates dependency conflicts while maintaining functionality
    """
    
    def __init__(self, groq_api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        """
        Initialize StepBeyond agent with Groq LLM and Qdrant RAG
        
        Args:
            groq_api_key: Groq API key
            model_name: Groq model to use
        """
        self.groq_api_key = groq_api_key
        self.model_name = model_name
        self.groq_base_url = "https://api.groq.com/openai/v1"
        
        # Initialize Qdrant RAG pipeline
        try:
            self.rag = QdrantRAGPipeline()
            logger.info("Qdrant RAG pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant RAG: {str(e)}")
            raise
        
        # Test Groq connection
        self._test_groq_connection()

    def _test_groq_connection(self):
        """Test connection to Groq API"""
        try:
            response = self._call_groq_api("Test connection", max_tokens=10)
            logger.info("Groq API connection successful")
        except Exception as e:
            logger.error(f"Groq API connection failed: {str(e)}")
            raise

    def _call_groq_api(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Call Groq API directly
        """
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 1,
            "stream": False
        }
        
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{self.groq_base_url}/chat/completions",
                    headers=headers,
                    json=data
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error calling Groq API: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error calling Groq API: {str(e)}")
            raise

    def process_query(self, user_query: str) -> str:
        """
        Process user query with context retrieval and LLM response
        """
        try:
            logger.info(f"Processing query: {user_query}")
            
            # Get relevant context from Qdrant
            context = self.rag.get_context_for_query(user_query, max_chunks=5)
            
            # Create comprehensive prompt
            prompt = self._create_comprehensive_prompt(user_query, context)
            
            # Generate response
            response = self._call_groq_api(prompt, max_tokens=2000)
            
            logger.info("Query processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return self._fallback_response(user_query, str(e))

    def _create_comprehensive_prompt(self, query: str, context: str) -> str:
        """Create a comprehensive prompt for the LLM"""
        
        prompt = f"""You are StepBeyond, an expert career counselor and education consultant specializing in helping Indian students study abroad. You have extensive knowledge about visa processes, university applications, scholarships, and country-specific requirements for popular study destinations like USA, UK, Canada, Australia, Germany, and others.

STUDENT QUERY: {query}

RELEVANT KNOWLEDGE BASE CONTEXT:
{context}

INSTRUCTIONS:
1. Provide a comprehensive, step-by-step answer to the student's question
2. Use the context from the knowledge base when relevant, but also draw from your general expertise
3. Structure your response clearly with headings and bullet points where appropriate
4. Include specific requirements, documents, timelines, and procedures
5. Provide practical tips and advice specifically for Indian students
6. Mention important deadlines, costs, or critical information
7. Be encouraging and supportive in your tone
8. If the query is about a specific country, focus on that country's requirements
9. If asking about multiple options, provide a comparative analysis

FORMAT YOUR RESPONSE:
- Use clear headings (##) for main sections
- Use bullet points for lists and steps
- Include specific details like document names, timelines, costs
- Add "💡 Pro Tips" section with practical advice
- End with "🎯 Next Steps" outlining immediate actions

Remember: You're helping an ambitious Indian student achieve their dream of studying abroad. Be specific, practical, and motivational!

RESPONSE:"""

        return prompt

    def _fallback_response(self, query: str, error: str) -> str:
        """Generate a fallback response when main processing fails"""
        return f"""I apologize, but I encountered a technical issue while processing your question about "{query}". 

However, I can offer some general guidance:

## 🎓 General Study Abroad Guidance

**For Visa Applications:**
- Gather all required documents (passport, I-20/CAS, financial statements, transcripts)
- Apply early - visa processing can take 2-8 weeks
- Prepare for visa interviews with practice questions

**For University Applications:**
- Research deadlines (usually fall between December-March)
- Prepare standardized tests (IELTS/TOEFL, GRE/GMAT) well in advance
- Draft strong Statement of Purpose highlighting your goals

**For Financial Planning:**
- Budget for tuition, living expenses, and other costs
- Research scholarships and assistantships
- Ensure sufficient funds for visa requirements

## 🎯 Next Steps
1. Please try asking your question again with more specific details
2. If the issue persists, consider breaking down your question into smaller parts
3. For urgent queries, you may want to consult with a local education consultant

I'm here to help you succeed in your study abroad journey! 🚀

*Technical note: {error}*"""

    def quick_answer(self, user_query: str) -> str:
        """Generate a quick answer with minimal context"""
        try:
            logger.info(f"Generating quick answer for: {user_query}")
            
            # Get limited context
            context = self.rag.get_context_for_query(user_query, max_chunks=3)
            
            # Create simpler prompt
            prompt = f"""You are a career counselor helping Indian students study abroad.

Question: {user_query}

Available context: {context}

Provide a clear, concise answer with practical steps. Focus on actionable advice for Indian students."""
            
            response = self._call_groq_api(prompt, max_tokens=1000)
            return response
            
        except Exception as e:
            logger.error(f"Error in quick_answer: {str(e)}")
            return "I'm having trouble processing your request right now. Please try again in a moment, or rephrase your question."

    def get_system_status(self) -> Dict[str, Any]:
        """Get system status information"""
        try:
            # Get Qdrant collection info
            collection_info = self.rag.get_collection_info()
            
            status = {
                "rag_system": "operational" if collection_info else "error",
                "knowledge_base": {
                    "documents_count": collection_info.get('points_count', 0),
                    "status": collection_info.get('status', 'unknown')
                },
                "llm_system": "operational",  # If we got here, Groq is working
                "model": self.model_name
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {
                "rag_system": "error",
                "llm_system": "unknown",
                "error": str(e)
            }

    def ingest_documents(self, data_dir: str = "data") -> int:
        """Ingest documents from directory into Qdrant"""
        try:
            count = self.rag.ingest_documents_from_directory(data_dir)
            logger.info(f"Ingested {count} document chunks")
            return count
        except Exception as e:
            logger.error(f"Error ingesting documents: {str(e)}")
            return 0

    def clear_knowledge_base(self):
        """Clear the knowledge base"""
        try:
            self.rag.clear_collection()
            logger.info("Knowledge base cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {str(e)}")
            raise

def setup_agent(groq_api_key: str = None, model_name: str = None) -> StepBeyondAgent:
    """Setup and return StepBeyond agent instance"""
    
    if not groq_api_key:
        groq_api_key = os.getenv('GROQ_API_KEY')
        
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found. Please set it in your environment variables or pass it directly.")
    
    # Use environment variable for model if available
    if not model_name:
        model_name = os.getenv('LLM_MODEL', 'llama-3.3-70b-versatile')
    
    return StepBeyondAgent(groq_api_key, model_name)

# Backward compatibility wrapper
class StepBeyondCrew:
    """Backward compatibility wrapper for existing code"""
    
    def __init__(self, groq_api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        self.agent = StepBeyondAgent(groq_api_key, model_name)
    
    def process_query(self, query: str) -> str:
        return self.agent.process_query(query)
    
    def quick_answer(self, query: str) -> str:
        return self.agent.quick_answer(query)

def setup_crew(groq_api_key: str = None, model_name: str = None) -> StepBeyondCrew:
    """Backward compatibility function"""
    return StepBeyondCrew(groq_api_key or os.getenv('GROQ_API_KEY'), model_name)
