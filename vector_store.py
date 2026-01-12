# vector_store.py - исправленная версия
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
import time
from tqdm import tqdm
from configuration import config

class VectorStore:
    def __init__(self):
        print(f"Инициализация модели эмбеддингов: {config.EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(
            config.EMBEDDING_MODEL, 
            device=config.EMBEDDING_DEVICE
        )
        self.index = None
        self.chunks = []  
        
    def build_index(self, chunks):
        if not chunks:
            print("Нет чанков для индексации")
            return
            
        valid_chunks = []
        texts = []
        
        for chunk in chunks:
            if isinstance(chunk, dict) and "text" in chunk:
                text = chunk["text"].strip()
                if text and len(text) >= 20:  
                    chunk_with_all_fields = {
                        "text": text,
                        "pdf_sha1": chunk.get("pdf_sha1", ""),
                        "page_index": chunk.get("page_index", 1),
                        "source": chunk.get("source", "")
                    }
                    valid_chunks.append(chunk_with_all_fields)
                    texts.append(text)
        
        if not texts:
            print("Нет валидных текстов для индексации")
            return
            
        print(f"Валидных чанков для индексации: {len(texts)}")
        
        print("Генерация эмбеддингов")
        embeddings = self.embedder.encode(
            texts, 
            batch_size=config.EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype(np.float32))
        
        self.chunks = valid_chunks
        
        self.save()
        
        print(f"Индекс построен: {self.index.ntotal} векторов")
    
    def save(self):
        os.makedirs(config.INDEX_PATH, exist_ok=True)
        
        faiss.write_index(self.index, os.path.join(config.INDEX_PATH, "index.faiss"))
        
        data_to_save = {
            "chunks": self.chunks,
            "config": {
                "embedding_model": config.EMBEDDING_MODEL,
                "chunk_size": config.CHUNK_SIZE
            }
        }
        
        with open(os.path.join(config.INDEX_PATH, "metadata.pkl"), "wb") as f:
            pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Индекс сохранен в {config.INDEX_PATH}")
    
    def load(self):
        index_path = os.path.join(config.INDEX_PATH, "index.faiss")
        metadata_path = os.path.join(config.INDEX_PATH, "metadata.pkl")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError("Индекс не найден")
        
        self.index = faiss.read_index(index_path)
        
        if os.path.exists(metadata_path):
            with open(metadata_path, "rb") as f:
                data = pickle.load(f)
                
                if isinstance(data, dict) and 'chunks' in data:
                    self.chunks = data['chunks']
                    print(f"Загружено {len(self.chunks)} чанков")
                elif isinstance(data, list):
                    self.chunks = data
                    print(f"Загружено {len(self.chunks)} чанков (старый формат)")
                else:
                    print(f"Неизвестный формат метаданных: {type(data)}")
                    self.chunks = []
        else:
            print("Файл метаданных не найден")
            self.chunks = []
        
        print(f"Индекс загружен: {self.index.ntotal} векторов")
    
    def search(self, query: str, k: int = None, min_score: float = 0.15):
        if k is None:
            k = config.TOP_K_RESULTS
        
        if not self.index or self.index.ntotal == 0:
            print("Индекс пуст")
            return []
        
        query_embedding = self.embedder.encode([query], normalize_embeddings=True)
        
        search_k = min(k * 3, self.index.ntotal)
        
        scores, indices = self.index.search(query_embedding.astype(np.float32), search_k)
        
        results = []
        seen_texts = set()
        
        for idx, score in zip(indices[0], scores[0]):
            if 0 <= idx < len(self.chunks) and score >= min_score:
                chunk = self.chunks[idx]
                text = chunk.get("text", "").strip()
                if not text or len(text) < 20:
                    continue
                if text in seen_texts:
                    continue
                    
                seen_texts.add(text)
                
                results.append({
                    "text": text,
                    "pdf_sha1": chunk.get("pdf_sha1", ""),
                    "page_index": chunk.get("page_index", 1),
                    "source": chunk.get("source", ""),
                    "score": float(score)  # Inner Product score (выше = лучше)
                })
                
                if len(results) >= k:
                    break
        
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results
    
    @property
    def metadata(self):
        return self.chunks