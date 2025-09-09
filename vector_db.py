from qdrant_client import QdrantClient
from qdrant_client import models
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple
from md_parser import get_chunks
import pickle
import os
from tqdm import tqdm

class VectorDB:
    def __init__(self, collection_name: str = 'fastapi_docs'):
        self.collection_name = collection_name
        self.client = QdrantClient('http://localhost:6333') # Подключение к запущенному контейнеру
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
        # 'qilowoq/paraphrase-multilingual-mpnet-base-v2-en-ru'
        # self.embedding_model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

        if self.client.collection_exists(collection_name):
            print(f'Коллекция {collection_name} уже существует')
        else:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config= models.VectorParams(
                    size=768, # Размерность модели
                    distance=models.Distance.COSINE
                )
            )
            print(f'Коллекция {self.collection_name} создана')
        
    def add_chunks(self, chunks: List[Tuple[str, str]]):
        
        if os.path.isfile('dumps/points.pkl'):
            pass
                
        else:
            print(f'Добавление точек в {self.collection_name}')
            points = []
            for idx, (header, content) in tqdm(enumerate(chunks)):
                embedding = self.embedding_model.encode(content).tolist()
                point = models.PointStruct(
                    id=idx,
                    vector=embedding,
                    payload={
                        'content': content,
                        'header': header,
                        'source': self.collection_name,
                        'text_lenght': len(content)
                    }
                )
                points.append(point)
            with open('dumps/points.pkl', 'wb') as f:
                pickle.dump(points, f)

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )



    def search(self, query:str, limit=5):
        '''Поиск похожих чанков'''
        query_embedding = self.embedding_model.encode(query).tolist()

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=limit,
            with_vectors=False
        )

        return results
            
            
        
if __name__ == '__main__':

    client = VectorDB()

    if os.path.isfile('dumps/chunks.pkl'):
        with open('dumps/chunks.pkl', 'rb') as f:
            chunks = pickle.load(f)
            print(f'Чанки загружены из файла {f.name}')
    else:
        chunks = get_chunks()
    client.add_chunks(chunks)
    while True:
        print("Для завершения работы, введите 'выход'")
        query = str(input('Введите вопрос: ')).lower()
        if query == 'выход':
            break
        limit = int(input('Введите лимит: '))
        result = client.search(query, limit)

        for result in result.points:
            print(f"Score: {result.score:.3f}")
            print(f"Header: {result.payload['header']}")
            print(f"Content: {result.payload['content'][:200]}...")
            print("---")

                

