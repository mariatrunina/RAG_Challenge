import os

class Config:
    # Пути к файлам
    PDF_DIR = r"e:\Desktop\RAG\RAG_Challenge\photo\pdfs"
    INDEX_PATH = "./faiss_index"
    QUESTIONS_PATH = "./questions.json"
    SUBMISSION_PATH = "./submission_trunina_v1.json"
    CACHE_DIR = "./cache"
    
    # Эмбеддинги
    EMBEDDING_MODEL = "intfloat/multilingual-e5-large" 
    EMBEDDING_DEVICE = "cpu" 
    EMBEDDING_BATCH_SIZE = 16 
    
    TOP_K_RESULTS = 7  
    
    # Параметры текста 
    CHUNK_SIZE = 2000   
    CHUNK_OVERLAP = 200
    MAX_PAGES = 1000

    # LLM
    LLM_MODEL = "llama-3.1-8b-instant"
    import os
    api_key = os.getenv("GROQ_API_KEY")
config = Config()