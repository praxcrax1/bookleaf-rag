import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class Config:
    # API Keys
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY")
    google_api_key: str = os.getenv("GOOGLE_API_KEY")
    
    # Pinecone settings
    index_name: str = os.getenv("INDEX_NAME", "google-doc-rag")
    dimension: int = int(os.getenv("DIMENSION", "768"))  # Gemini embedding dimension
    
    # Processing settings
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    batch_size: int = 100
    
    # Multi-size chunking configuration for scattered knowledge
    LARGE_CHUNK_SIZE: int = 1500
    LARGE_CHUNK_OVERLAP: int = 300
    MEDIUM_CHUNK_SIZE: int = 800
    MEDIUM_CHUNK_OVERLAP: int = 200
    SMALL_CHUNK_SIZE: int = 400
    SMALL_CHUNK_OVERLAP: int = 100
    CROSS_REF_WINDOW: int = 2

    # Retrieval settings
    top_k: int = int(os.getenv("TOP_K", "5"))
    score_threshold: float = float(os.getenv("SCORE_THRESHOLD", "0.7"))
    
    # Model settings
    model_name: str = (os.getenv("MODEL_NAME", "gemini-2.5-flash"))  # Gemini chat model
    embedding_model: str = (os.getenv("EMBEDDING_MODEL", "models/embedding-001"))  # Gemini embedding model


    # JWT/Auth settings
    jwt_secret: str = os.getenv("JWT_SECRET", "your-secret-key")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")

    # MongoDB settings
    mongo_uri: str = os.getenv("MONGO_URI", "mongodb+srv://<username>:<password>@<cluster-url>/books?retryWrites=true&w=majority")
    db_name: str = os.getenv("DB_NAME", "books")
    collection_name: str = os.getenv("COLLECTION_NAME", "books")

config = Config()
