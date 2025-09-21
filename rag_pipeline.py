import os
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter
from qdrant_client.http.exceptions import ResponseHandlingException
import PyPDF2
import docx
from loguru import logger
import time
import httpx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import pickle

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class LightweightRAGPipeline:
    def __init__(self, 
                 collection_name: str = "stepbeyond_docs",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        """
        Initialize Lightweight RAG Pipeline with TF-IDF embeddings for Render free tier
        """
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Get Qdrant credentials from environment
        self.qdrant_url = os.getenv('QDRANT_URL')
        self.qdrant_api_key = os.getenv('QDRANT_API_KEY')
        
        if not self.qdrant_url or not self.qdrant_api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in environment variables")
        
        # Initialize TF-IDF vectorizer (lightweight alternative to sentence-transformers)
        self.vectorizer = TfidfVectorizer(
            max_features=384,  # Match common embedding dimensions but lighter
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            max_df=0.95,
            min_df=2
        )
        
        self.embedding_dim = 384  # TF-IDF feature dimension
        self.is_fitted = False
        
        # Initialize Qdrant client
        logger.info(f"Connecting to Qdrant at: {self.qdrant_url}")
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            timeout=60
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

    def simple_text_splitter(self, text: str) -> List[str]:
        """Simple text splitting using NLTK"""
        try:
            # Split into sentences first
            sentences = sent_tokenize(text)
            
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                # If adding this sentence would exceed chunk_size, start new chunk
                if len(current_chunk) + len(sentence) > self.chunk_size:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
            
            # Add the last chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                
            return chunks
            
        except Exception as e:
            logger.error(f"Error in text splitting: {str(e)}")
            # Fallback to simple splitting
            words = text.split()
            chunks = []
            for i in range(0, len(words), self.chunk_size//10):  # Rough word count
                chunk = " ".join(words[i:i+self.chunk_size//10])
                if chunk.strip():
                    chunks.append(chunk.strip())
            return chunks

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
                text_chunks = self.simple_text_splitter(doc['content'])
                
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
        """Create TF-IDF embeddings for text chunks"""
        if not texts:
            return []
            
        logger.info(f"Creating TF-IDF embeddings for {len(texts)} chunks")
        
        try:
            if not self.is_fitted:
                # Fit the vectorizer on all texts
                self.vectorizer.fit(texts)
                self.is_fitted = True
                logger.info("TF-IDF vectorizer fitted")
            
            # Transform texts to vectors
            tfidf_matrix = self.vectorizer.transform(texts)
            
            # Convert sparse matrix to dense and then to list
            embeddings = tfidf_matrix.toarray().tolist()
            
            # Pad or truncate to exact dimension
            processed_embeddings = []
            for embedding in embeddings:
                if len(embedding) < self.embedding_dim:
                    # Pad with zeros
                    embedding.extend([0.0] * (self.embedding_dim - len(embedding)))
                elif len(embedding) > self.embedding_dim:
                    # Truncate
                    embedding = embedding[:self.embedding_dim]
                processed_embeddings.append(embedding)
            
            logger.info(f"Created {len(processed_embeddings)} embeddings")
            return processed_embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            # Fallback to random embeddings (for testing)
            return [[0.0] * self.embedding_dim for _ in texts]

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

    def similarity_search(self, query: str, limit: int = 5, score_threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Search for similar documents in Qdrant"""
        try:
            # Create query embedding
            if not self.is_fitted:
                logger.warning("Vectorizer not fitted, cannot search")
                return []
            
            query_vector = self.vectorizer.transform([query]).toarray()[0].tolist()
            
            # Pad or truncate to exact dimension
            if len(query_vector) < self.embedding_dim:
                query_vector.extend([0.0] * (self.embedding_dim - len(query_vector)))
            elif len(query_vector) > self.embedding_dim:
                query_vector = query_vector[:self.embedding_dim]
            
            # Search in Qdrant
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
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

# Alias for backward compatibility
QdrantRAGPipeline = LightweightRAGPipeline
