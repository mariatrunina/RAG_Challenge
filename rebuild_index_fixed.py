import os
from configuration import config
from pdf_parser import PDFParser
from vector_store import VectorStore

def rebuild_index():
    print("Перестроение индекса с исправленным форматом")
    parser = PDFParser()
    
    pdf_files = []
    if not os.path.exists(config.PDF_DIR):
        print(f"Ошибка: Путь {config.PDF_DIR} не найден")
        return

    for filename in os.listdir(config.PDF_DIR):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(config.PDF_DIR, filename)
            pdf_files.append(file_path)
    
    print(f"Парсинг {len(pdf_files)} PDF файлов")
    
    all_chunks = []
    for file_path in pdf_files:
        chunks = parser.parse_single(file_path)
        all_chunks.extend(chunks)
        print(f"  {os.path.basename(file_path)}: {len(chunks)} чанков")
    
    print(f"Всего извлечено чанков: {len(all_chunks)}")
    
    if not all_chunks:
        print("Нет чанков для индексации. Проверьте содержимое PDF или путь к папке.")
        return
    
    # 2. Строим индекс
    print("\nПостроение индекса")
    vector_store = VectorStore()
    vector_store.build_index(all_chunks)
    
    print("Индекс перестроен")
    
    # 3. Тестируем поиск
    print("\n Тестируем поиск")
    test_questions = ["Выручка", "Компания", "ESG"]
    
    for question in test_questions:
        print(f"\nПоиск: '{question}'")
        results = vector_store.search(question, k=2)
        if results:
            for res in results:
                print(f"  [Score: {res['score']:.3f}] {res['text'][:100]}...")
        else:
            print("Ничего не найдено")

if __name__ == "__main__":
    rebuild_index()