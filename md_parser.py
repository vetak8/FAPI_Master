import os
import re
import pickle
from langchain.docstore.document import  Document
from langchain.text_splitter import MarkdownHeaderTextSplitter
from tqdm import tqdm
from typing import List, Tuple
PATH = './docs/'

def get_chunks(path: str = PATH) -> List[Tuple[str, str]]:
    headers_to_split_on = [
        ('#', 'Header 1'),
        ('##', 'Header 2'),
        ('###', 'Header 3'),
    ]

    all_chunks = []
    for root, _, files in os.walk(path):
        for file in tqdm(files):
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                try:
                    # Загрузка файла
                    with open(file_path, 'r', encoding='utf-8') as f:
                        markdown_content = f.read()
                        
                    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
                    doc_chunks = markdown_splitter.split_text(markdown_content)
                    
                    for chunk in doc_chunks:                        
                        content = clean_text(chunk.page_content) # Чистим текст
                        all_chunks.append((list(chunk.metadata.values())[0], content)) # на выходе список кортежей вида (заголовок, текст)
                   
                except Exception as e:
                    print(f'Ошибка обработки файла {file_path}: {e}')
                    continue

    print(f'\nВсего создано {len(all_chunks)} чанков из {len(files)} файлов')
    return all_chunks
                


def clean_text(text: str) -> str:
    '''Чистка текста от HTML-тегов, лишних символов и сусора с сохранением структуры и содержания'''
    
    if not text or not text.strip():
        return ""

    cleaned = text.strip() 

    # Удаляем теги HTML-теги
    cleaned = re.sub(r'<a\s+href="([^"]*)"[^>]*>([^<]*)</a>', r'\2 (\1)', cleaned)
    cleaned = re.sub(r'<[^>]+>', '', cleaned)

    # Символы типа '///'
    cleaned = re.sub(r'///.*?///', '', cleaned, flags=re.DOTALL)

    # Заголовки
    cleaned = re.sub(r'#+\s*', '', cleaned)
    
    # Жирный текст
    cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned) 
    
    # Курсив
    cleaned = re.sub(r'\*(.*?)\*', r'\1', cleaned)  
    
    # Удаляем лишние пробелы и переносы строк
    cleaned = re.sub(r'\s+', ' ', cleaned)
    cleaned = cleaned.strip().lower()

    return cleaned

if __name__ == '__main__':

    print(f'\n\nПарсинг Markdown файлов в директории {PATH}\n\n')
    chunks = get_chunks(PATH)

    with open('dumps/chunks.pkl', 'wb') as f:
        print(f'Сохранение файла {f.name}')
        pickle.dump(chunks, f)