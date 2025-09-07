from qdrant_client import QdrantClient
from qdrant_client import models
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple
from md_parser import get_chunks
import pickle


class VectorDB:
    def __init__(self, collection_name: str = 'fastapi_docs'):
        self.collection_name = collection_name
        self.client = QdrantClient('http://localhost:6333') # Подключение к запущенному контейнеру
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config= models.VectorParams(
                    size=768, # Размерность модели
                    distance=Distance.COSINE
                )
            )
            print(f'Коллекция {collection_name} создана')
        except Exception as e:
            print(f'Коллекция {collection_name} уже существует или ошибка {e}')
        
    def add_chunks(self, chunks: List[Tuple[str, str]] = None):
        if  not chunks:            
            try:
                with open('chunks.pkl', 'rb') as f:
                    chunks = pickle.load(f)
                    print(f'Чанки загружены из файла {f}')
            except Exception as e:
                print(f'Ошибка {e} чтения файла {f} ')
                
        chunks = get_chunks()

        points = []
        for idx, (header, content) in enumerate(chunks):
            embedding = self.embedding_model.encode(content).tolist()
            point = models.PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    'content': content,
                    'header': header,
                    'source': collection_name,
                    'text_lenght': len(content)
                }
            )
            points.append(point)

        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
            
            
        
            
                
                

