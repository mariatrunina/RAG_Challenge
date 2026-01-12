import json
import re
import os
import time
from groq import Groq
from typing import Dict, List, Any
from configuration import config

class QueryProcessor:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.questions = self._load_questions()
        self.client = Groq(api_key=config.GROQ_API_KEY)
        self.model_name = config.LLM_MODEL
        self.temp_file = "temp_results.json" 

    def _load_questions(self):
        with open(config.QUESTIONS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _call_llm(self, prompt: str) -> str:
        wait_times = [15, 30, 60] 
        for attempt in range(len(wait_times)):
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a precise data extractor. Give only the value."},
                        {"role": "user", "content": prompt}
                    ],
                    model=self.model_name,
                    temperature=0.0,
                    max_tokens=100,
                )
                return chat_completion.choices[0].message.content.strip()
            except Exception as e:
                err_msg = str(e).lower()
                if "413" in err_msg or "tokens" in err_msg or "rate_limit" in err_msg:
                    print(f"Ожидание {wait_times[attempt]} сек...")
                    time.sleep(wait_times[attempt])
                else:
                    print(f"Ошибка API: {e}")
                    break
        return "N/A"

    def _create_prompt(self, question: Dict, context: List[Dict]) -> str:
        context_parts = []
        for i, res in enumerate(context):
            clean_text = " ".join(res['text'].split())
            context_parts.append(f"CHUNK {i+1}: {clean_text}")
        
        context_text = "\n\n".join(context_parts)
        kind = question.get('kind', 'text')

        return f"""You are a professional financial analyst. Extract the exact answer from the context.

CONTEXT:
{context_text}

INSTRUCTIONS:
1. Provide only the specific value for the question.
2. If Kind is 'number', output ONLY the digit (e.g., 1500000). Convert 'million/billion' to zeros.
3. If Kind is 'boolean', output only 'true' or 'false'.
4. If the information is missing, respond exactly with 'N/A'.
5. No conversational filler or explanations.

QUESTION: {question['text']}
EXPECTED TYPE: {kind}

ANSWER:"""
    def _format_value(self, value: str, kind: str) -> Any:
        val = value.strip().lower().split('\n')[0].replace("'", "").replace('"', '')
        if any(x in val for x in ["n/a", "no information", "not found"]): return "N/A"

        if kind == 'boolean':
            if any(x in val for x in ['true', 'yes', '1']): return True
            if any(x in val for x in ['false', 'no', '0']): return False
            return "N/A"

        if kind == 'number':
            clean_val = re.sub(r'(?<=\d)\s(?=\d)', '', val)
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", clean_val.replace(',', '.'))
            if nums:
                n = float(nums[0])
                return int(n) if n.is_integer() else n
            return "N/A"
        return value.strip()

    def process_all(self) -> List[Dict]:
        results = []
        if os.path.exists(self.temp_file):
            with open(self.temp_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            print(f" {len(results)} ответов")

        K_SEARCH = 6 

        for i in range(len(results), len(self.questions)):
            q = self.questions[i]
            time.sleep(5.0)
            
            print(f"[{i+1}/{len(self.questions)}] {q['text'][:60]}...")
            search_results = self.vector_store.search(q['text'], k=K_SEARCH)
            
            if not search_results:
                final_val, ref = "N/A", []
            else:
                prompt = self._create_prompt(q, search_results)
                raw_answer = self._call_llm(prompt)
                final_val = self._format_value(raw_answer, q.get('kind', 'text'))
                ref = [{"pdf_sha1": search_results[0]["pdf_sha1"], "page_index": search_results[0]["page_index"]}] if final_val != "N/A" else []

            results.append({
    "question_text": q['text'],
    "value": final_val, 
    "references": ref
})
            print(f" {final_val}")

            with open(self.temp_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        if len(results) == len(self.questions) and os.path.exists(self.temp_file):
            os.remove(self.temp_file)
            
        return results