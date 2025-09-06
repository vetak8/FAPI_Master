import os
from langchain.docstore.document import  Document
from langchain.text_splitter import MarkdownHeaderTextSplitter
from tqdm import tqdm
from typing import List, Tuple
PATH = './docs/'

def get_chunks(path) -> List[Tuple[str, str]]:
    headers_to_split_on = [
        ('#', 'Header 1'),
        ('##', 'Header 2'),
        ('###', 'Header 3'),
        # ('###', 'Header 4')
    ]

    all_chunks = []
    for root, _, files in os.walk(path):
        for file in tqdm(files):
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                try:
                    # Загрузка файла
                    with open(file_path) as f:
                        markdown_content = f.read()
                        
                    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
                    doc_chunks = markdown_splitter.split_text(markdown_content)
                    
                    for chunk in doc_chunks:
                        all_chunks.append((list(chunk.metadata.values())[0], chunk.page_content)) # на выходе список кортежей вида (заголовок, текст)
                   
                except Exception as e:
                    print(f'Ошибка обработки файла {file_path}: {e}')
                    continue

    print(f'Всего создано {len(all_chunks)} чанков из {len(files)} файлов')
    return all_chunks
                

def chunk_cleaner(chunk):
    