import os
import uuid
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Range
from qdrant_client.http.exceptions import ResponseHandlingException
from langchain_text_splitters import RecursiveCharacterTextSplitter
import PyPDF2
import docx
from loguru import logger
import time

class QdrantRAGPipeline:
    def __init__(self, 
                 collection_name: str = "stepbeyond_docs",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        """
        Initialize RAG Pipeline with Qdrant vector database
        """
        self.collection_name = collection_name
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Get Qdrant credentials from environment
        self.qdrant_url = os.getenv('QDRANT_URL')
        self.qdrant_api_key = os.getenv('QDRANT_API_KEY')
        
        if not self.qdrant_url or not self.qdrant_api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in environment variables")
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            timeout=60
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize collection
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Ensure the collection exists in Qdrant"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection {self.collection_name} created successfully")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            raise

    def load_text_file(self, file_path: str) -> str:
        """Load text from a text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {str(e)}")
            return ""

    def load_pdf_file(self, file_path: str) -> str:
        """Load text from a PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error loading PDF file {file_path}: {str(e)}")
            return ""

    def load_docx_file(self, file_path: str) -> str:
        """Load text from a DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error loading DOCX file {file_path}: {str(e)}")
            return ""

    def load_documents_from_directory(self, data_dir: str = "data") -> List[Dict[str, Any]]:
        """Load documents from directory"""
        documents = []
        
        if not os.path.exists(data_dir):
            logger.warning(f"Data directory {data_dir} does not exist")
            return documents
            
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            
            if not os.path.isfile(file_path):
                continue
                
            try:
                text_content = ""
                
                if filename.lower().endswith('.pdf'):
                    text_content = self.load_pdf_file(file_path)
                    logger.info(f"Loaded PDF: {filename}")
                    
                elif filename.lower().endswith('.txt'):
                    text_content = self.load_text_file(file_path)
                    logger.info(f"Loaded TXT: {filename}")
                    
                elif filename.lower().endswith('.docx'):
                    text_content = self.load_docx_file(file_path)
                    logger.info(f"Loaded DOCX: {filename}")
                    
                else:
                    logger.warning(f"Unsupported file format: {filename}")
                    continue
                
                if text_content.strip():
                    documents.append({
                        'content': text_content,
                        'filename': filename,
                        'filepath': file_path,
                        'file_type': filename.split('.')[-1].lower()
                    })
                    
            except Exception as e:
                logger.error(f"Error loading {filename}: {str(e)}")
                
        logger.info(f"Loaded {len(documents)} documents from {data_dir}")
        return documents

    def split_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into chunks"""
        chunks = []
        
        for doc in documents:
            try:
                text_chunks = self.text_splitter.split_text(doc['content'])
                
                for i, chunk in enumerate(text_chunks):
                    if chunk.strip():  # Only add non-empty chunks
                        chunks.append({
                            'content': chunk,
                            'filename': doc['filename'],
                            'filepath': doc['filepath'],
                            'file_type': doc['file_type'],
                            'chunk_index': i,
                            'chunk_id': f"{doc['filename']}_{i}"
                        })
                        
                logger.info(f"Split {doc['filename']} into {len(text_chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error splitting document {doc['filename']}: {str(e)}")
        
        logger.info(f"Total chunks created: {len(chunks)}")
        return chunks

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for text chunks"""
        if not texts:
            return []
            
        logger.info(f"Creating embeddings for {len(texts)} chunks")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()

    def ingest_documents_from_directory(self, data_dir: str = "data", batch_size: int = 50):
        """Ingest documents from directory into Qdrant"""
        logger.info(f"Starting document ingestion from {data_dir}")
        
        # Load documents
        documents = self.load_documents_from_directory(data_dir)
        if not documents:
            logger.warning("No documents found to ingest")
            return 0
        
        # Split into chunks
        chunks = self.split_documents(documents)
        if not chunks:
            logger.warning("No chunks created")
            return 0
        
        # Create embeddings
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.create_embeddings(texts)
        
        if len(embeddings) != len(chunks):
            logger.error("Mismatch between embeddings and chunks")
            return 0
        
        # Prepare points for Qdrant
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    'content': chunk['content'],
                    'filename': chunk['filename'],
                    'filepath': chunk['filepath'],
                    'file_type': chunk['file_type'],
                    'chunk_index': chunk['chunk_index'],
                    'chunk_id': chunk['chunk_id'],
                    'ingestion_timestamp': int(time.time())
                }
            )
            points.append(point)
        
        # Upload to Qdrant in batches
        total_uploaded = 0
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                total_uploaded += len(batch)
                logger.info(f"Uploaded batch {i//batch_size + 1}: {len(batch)} points")
                
            except Exception as e:
                logger.error(f"Error uploading batch {i//batch_size + 1}: {str(e)}")
        
        logger.info(f"Successfully ingested {total_uploaded} chunks from {len(documents)} documents")
        return total_uploaded

    def similarity_search(self, query: str, limit: int = 5, score_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Search for similar documents in Qdrant"""
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold
            )
            
            # Format results
            results = []
            for result in search_results:
                results.append({
                    'content': result.payload['content'],
                    'filename': result.payload['filename'],
                    'file_type': result.payload['file_type'],
                    'chunk_index': result.payload['chunk_index'],
                    'score': result.score,
                    'metadata': {
                        'filepath': result.payload.get('filepath', ''),
                        'chunk_id': result.payload.get('chunk_id', ''),
                        'ingestion_timestamp': result.payload.get('ingestion_timestamp', 0)
                    }
                })
            
            logger.info(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []

    def get_context_for_query(self, query: str, max_chunks: int = 5) -> str:
        """Get relevant context for a query"""
        results = self.similarity_search(query, limit=max_chunks)
        
        if not results:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_part = f"[Document {i}: {result['filename']} (Score: {result['score']:.3f})]\n{result['content']}"
            context_parts.append(context_part)
        
        return "\n\n" + "="*50 + "\n\n".join(context_parts)

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                'name': self.collection_name,
                'vectors_count': collection_info.vectors_count,
                'indexed_vectors_count': collection_info.indexed_vectors_count,
                'points_count': collection_info.points_count,
                'status': collection_info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {}

    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")

    def clear_collection(self):
        """Clear all points from the collection"""
        try:
            # Delete all points
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter()
            )
            logger.info(f"Cleared all points from collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")

if __name__ == "__main__":
    # Example usage
    try:
        rag = QdrantRAGPipeline()
        
        # Get collection info
        info = rag.get_collection_info()
        print(f"Collection info: {info}")
        
        # Ingest documents if the collection is empty
        if info.get('points_count', 0) == 0:
            print("Collection is empty. Ingesting documents...")
            count = rag.ingest_documents_from_directory("data")
            print(f"Ingested {count} document chunks")
        
        # Test search
        query = "What are the requirements for student visa?"
        context = rag.get_context_for_query(query, max_chunks=3)
        print(f"\nQuery: {query}")
        print(f"\nRelevant context:\n{context}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure to set QDRANT_URL and QDRANT_API_KEY environment variables")
