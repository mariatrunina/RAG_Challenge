#python main.py --mode query
import json
import sys
from configuration import config
from vector_store import VectorStore
from query_processor import QueryProcessor

def run_rag_pipeline():
    vs = VectorStore()
    vs.load()
    
    processor = QueryProcessor(vs)
    
    print(f"Обработка {len(processor.questions)} вопросов через Groq API...")
    
    final_results = processor.process_all()

    submission = {
    "team_email": "mashatrunina2508@gmail.com",
    "submission_name": "trunina_v1",
    "answers": final_results
}

    try:
        with open(config.SUBMISSION_PATH, 'w', encoding='utf-8') as f:
            json.dump(submission, f, ensure_ascii=False, indent=2)
        print(f"\n Готово: {config.SUBMISSION_PATH}")
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and "--mode" in sys.argv:
        mode = sys.argv[sys.argv.index("--mode") + 1]
        if mode == "query":
            run_rag_pipeline()
        else:
            print(f"Режим {mode} не поддерживается в данной версии.")
    else:
        run_rag_pipeline()