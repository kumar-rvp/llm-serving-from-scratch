import torch
import multiprocessing as mp
import logging
import uuid
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, Depends
from pydantic import BaseModel

logger = logging.getLogger(__name__)
app = FastAPI()



class GenerateResponse(BaseModel):
    generated_text: str

class GenerateRequest(BaseModel):
    prompt : str

class Sequence:
    def __init__(self, seq_id: str, prompt: str, max_tokens, stopping_criteria):
        self.seq_id = seq_id
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.stopping_criteria = stopping_criteria



class ModelWorker:
    def __init__(self, model_name: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

    @staticmethod
    def run(model_name: str, task_queue: mp.Queue, result_queue: mp.Queue):
        worker = ModelWorker(model_name)
        while True:
            logger.debug("ModelWorker: Waiting for task")
            request = task_queue.get()
            try:
                generated_text = worker.generate(request)
                # FIX (12): put() takes ONE item – use a dict
                result_queue.put({"status": "completed", "generated_text": generated_text})
            except Exception as e:
                result_queue.put({"status": "error", "generated_text": str(e)})
                logger.debug("ModelWorker: Task error")
            logger.debug("ModelWorker: Task completed")
    
    def generate(self, sequence: Sequence, max_new_tokens: int = 200) -> str:
        inputs = self.tokenizer(sequence.prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class ModelManager:
    def __init__(self):
        pass


class ModelExecutor:
    def __init__(self):
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()

    def setup_worker(self, model_name: str):
        self.worker_process = mp.Process(target= ModelWorker.run, args=(model_name, self.task_queue, self.result_queue))
        self.worker_process.start()

    def execute(self, sequence : Sequence) -> dict:
        self.task_queue.put(sequence)
        result = self.result_queue.get()
        return result
    
    """def execute_batch(self, prompt: str):
        self.task_queue.put((prompts, False))
        results = self.result_queue.get()
        return results """ 


class WorkloadManager:
    def __init__(self):
        pass


class LLMEngine:
    def __init__(self):
        self.workload_manager = WorkloadManager()
        #self.model_manager = ModelManager()
        self.model_executor = ModelExecutor()
        self.max_tokens = 200
    
        #self.model_executor.setup_worker("facebook/opt-125m")
        #TinyLlama/TinyLlama-1.1B-Chat-v1.0
        self.model_executor.setup_worker("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    def basic_generate(self, prompt: str):
        sequnce = Sequence(str(uuid.uuid4()), prompt, None, None)
        results = self.model_executor.execute(sequnce)
        return results['generated_text']

_llm_engine : LLMEngine | None = None

def get_llm() -> LLMEngine:
    global _llm_engine
    if _llm_engine is None:
        _llm_engine = LLMEngine()
    return _llm_engine




@app.post("/basic_generate", response_model= GenerateResponse)
async def basic_generate(request: GenerateRequest, llm: LLMEngine = Depends(get_llm)):
    generated_text = llm.basic_generate(request.prompt)
    return GenerateResponse(generated_text=generated_text)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
