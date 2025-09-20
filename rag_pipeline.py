import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 data_dir: str = "data",
                 embeddings_dir: str = "embeddings"):
        """
        Initialize RAG Pipeline with FAISS vector database
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.data_dir = data_dir
        self.embeddings_dir = embeddings_dir
        
        # Initialize sentence transformer model
        logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = None
        self.documents = []
        self.metadata = []
        
        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(embeddings_dir, exist_ok=True)
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def load_documents(self) -> List[Document]:
        """Load documents from data directory"""
        documents = []
        
        if not os.path.exists(self.data_dir):
            logger.warning(f"Data directory {self.data_dir} does not exist")
            return documents
            
        for filename in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, filename)
            
            try:
                if filename.endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    logger.info(f"Loaded PDF: {filename} with {len(docs)} pages")
                    documents.extend(docs)
                    
                elif filename.endswith('.txt'):
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    logger.info(f"Loaded TXT: {filename}")
                    documents.extend(docs)
                    
            except Exception as e:
                logger.error(f"Error loading {filename}: {str(e)}")
                
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        if not documents:
            logger.warning("No documents to split")
            return []
            
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for text chunks"""
        if not texts:
            logger.warning("No texts to embed")
            return np.array([])
            
        logger.info(f"Creating embeddings for {len(texts)} text chunks")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return np.array(embeddings)

    def build_faiss_index(self, embeddings: np.ndarray):
        """Build FAISS index from embeddings"""
        if embeddings.size == 0:
            logger.warning("No embeddings to index")
            return
            
        # Create FAISS index (L2 distance)
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index.add(embeddings.astype('float32'))
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors")

    def save_index(self):
        """Save FAISS index and metadata to disk"""
        if self.index is None:
            logger.warning("No index to save")
            return
            
        # Save FAISS index
        index_path = os.path.join(self.embeddings_dir, "faiss_index.index")
        faiss.write_index(self.index, index_path)
        
        # Save documents and metadata
        docs_path = os.path.join(self.embeddings_dir, "documents.pkl")
        metadata_path = os.path.join(self.embeddings_dir, "metadata.pkl")
        
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
            
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
            
        logger.info(f"Saved index and metadata to {self.embeddings_dir}")

    def load_index(self) -> bool:
        """Load FAISS index and metadata from disk"""
        index_path = os.path.join(self.embeddings_dir, "faiss_index.index")
        docs_path = os.path.join(self.embeddings_dir, "documents.pkl")
        metadata_path = os.path.join(self.embeddings_dir, "metadata.pkl")
        
        try:
            if os.path.exists(index_path) and os.path.exists(docs_path):
                # Load FAISS index
                self.index = faiss.read_index(index_path)
                
                # Load documents
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
                    
                # Load metadata if exists
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'rb') as f:
                        self.metadata = pickle.load(f)
                
                logger.info(f"Loaded index with {self.index.ntotal} vectors and {len(self.documents)} documents")
                return True
                
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            
        return False

    def ingest_data(self):
        """Full data ingestion pipeline"""
        logger.info("Starting data ingestion pipeline...")
        
        # Load documents
        documents = self.load_documents()
        if not documents:
            logger.warning("No documents found to process")
            return
        
        # Split documents
        chunks = self.split_documents(documents)
        if not chunks:
            logger.warning("No chunks created")
            return
        
        # Store documents and create metadata
        self.documents = chunks
        self.metadata = [{"source": doc.metadata.get("source", "unknown"), 
                         "chunk_id": i} for i, doc in enumerate(chunks)]
        
        # Create embeddings
        texts = [doc.page_content for doc in chunks]
        embeddings = self.create_embeddings(texts)
        
        if embeddings.size == 0:
            logger.warning("No embeddings created")
            return
        
        # Build FAISS index
        self.build_faiss_index(embeddings)
        
        # Save everything
        self.save_index()
        
        logger.info("Data ingestion completed successfully!")

    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if self.index is None:
            logger.warning("No index loaded. Please run ingest_data() first.")
            return []
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Prepare results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):
                result = {
                    "content": self.documents[idx].page_content,
                    "metadata": self.documents[idx].metadata,
                    "distance": float(distance),
                    "rank": i + 1
                }
                results.append(result)
        
        logger.info(f"Found {len(results)} similar documents for query")
        return results

    def get_context_for_query(self, query: str, max_chunks: int = 5) -> str:
        """Get relevant context for a query"""
        results = self.similarity_search(query, k=max_chunks)
        
        if not results:
            return "No relevant information found."
        
        context_parts = []
        for result in results:
            context_parts.append(f"[Source: {result['metadata'].get('source', 'Unknown')}]\n{result['content']}")
        
        return "\n\n---\n\n".join(context_parts)

if __name__ == "__main__":
    # Example usage
    rag = RAGPipeline()
    
    # Check if index exists, if not create it
    if not rag.load_index():
        print("No existing index found. Creating new index...")
        rag.ingest_data()
    else:
        print("Loaded existing index.")
    
    # Test query
    query = "What are the requirements for student visa?"
    context = rag.get_context_for_query(query)
    print(f"\nQuery: {query}")
    print(f"\nContext:\n{context}")