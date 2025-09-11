from gigachat import GigaChat
from vector_db import VectorDB
from dotenv import load_dotenv
import os
load_dotenv()
from dotenv import load_dotenv
API_KEY = os.getenv('GIGACHAT_API_KEY')

class RAGSystem:
    def __init__(self):
        self.vector_db = VectorDB()
        self.giga_client = GigaChat(
            credentials=API_KEY,
            verify_ssl_certs=False
        )
    def ask(self, question: str):
        # 1. Ищем релевантные чанки
        results = self.vector_db.search(question, limit=6)
        # 2. Формируем контекст
        context = '\n\n'.join([
            f'ИСточник: {r.payload['header']}\n{r.payload['content']}'
            for r in results.points
        ])
        # 3. Промт для GigaChat
        prompt = f"""
        Ответь на вопрос пользователя, основываясь на  предоставленный контекст.
        Если ответа нет в контексте, ответь своими словами или скажи "Не могу найти информацию в документации".

        Контекст:
        {context}

        Вопрос: {question}
        Ответ:
        """
        # 4. Ответ GigaChat
        response = self.giga_client.chat(prompt)
        return response.choices[0].message.content

