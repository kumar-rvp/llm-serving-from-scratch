import uuid
import logging
import torch
import multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, Depends
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class BasicGenerateRequest(BaseModel):
    prompt: str


class BasicGenerateResponse(BaseModel):
    generated_text: str


# ---------------------------------------------------------------------------
# Sequence (placeholder – fill in fields the book defines)
# ---------------------------------------------------------------------------

class Sequence:
    def __init__(self, seq_id: str, prompt: str, max_tokens, stopping_criteria):
        self.seq_id = seq_id
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.stopping_criteria = stopping_criteria


# ---------------------------------------------------------------------------
# WorkloadManager (stub – wire up when the book covers scheduling)
# ---------------------------------------------------------------------------

class WorkloadManager:
    pass


# ---------------------------------------------------------------------------
# ModelManager (kept for reference; logic is now also in ModelWorker)
# ---------------------------------------------------------------------------

class ModelManager:
    def load_model(self, model_name: str = "facebook/opt-125m"):
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer


# ---------------------------------------------------------------------------
# ModelWorker  – runs inside a child process
# ---------------------------------------------------------------------------

class ModelWorker:
    def __init__(self, model_name: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # FIX (10): load model + tokenizer here so generate() can use them
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

    # FIX (11): staticmethod – creates its own worker instance internally
    @staticmethod
    def run(model_name: str, task_queue: mp.Queue, result_queue: mp.Queue):
        worker = ModelWorker(model_name)
        while True:
            logger.debug("ModelWorker: waiting for task")
            request = task_queue.get()
            try:
                generated_text = worker.generate(request)
                # FIX (12): put() takes ONE item – use a dict
                result_queue.put({"status": "completed", "generated_text": generated_text})
            except Exception as e:
                result_queue.put({"status": "error", "generated_text": str(e)})
            logger.debug("ModelWorker: task completed")

    # FIX (13): define the missing generate() method
    def generate(self, sequence: Sequence, max_new_tokens: int = 20) -> str:
        inputs = self.tokenizer(sequence.prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# ModelExecutor  – owns the child process and the two queues
# ---------------------------------------------------------------------------

class ModelExecutor:
    def __init__(self):
        # FIX (6): both lines belong inside __init__
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()

    def setup_worker(self, model_name: str):
        self.worker_process = mp.Process(
            target=ModelWorker.run,
            args=(model_name, self.task_queue, self.result_queue),
        )
        # FIX (6): start() belongs inside setup_worker, not at class body level
        self.worker_process.start()

    # FIX (7): restore a clean execute() that the engine can call
    def execute(self, sequence: Sequence) -> dict:
        self.task_queue.put(sequence)
        result = self.result_queue.get()   # blocks until the worker responds
        return result

    # execute_batch kept for later; FIX (8/9): parameter name made consistent
    def execute_batch(self, sequences) -> dict:
        self.task_queue.put((sequences, False))
        result = self.result_queue.get()
        return result


# ---------------------------------------------------------------------------
# LLMEngine  – top-level orchestrator
# ---------------------------------------------------------------------------

class LLMEngine:
    def __init__(self):
        self.workload_manager = WorkloadManager()
        self.model_executor = ModelExecutor()
        self.max_tokens = 20
        # FIX (1): call is now inside __init__ where it belongs
        # FIX (2): method name corrected to setup_worker (was setupworker)
        self.model_executor.setup_worker("facebook/opt-125m")

    def basic_generate(self, prompt: str) -> str:
        # FIX (3): balanced parentheses on str(uuid.uuid4())
        # FIX (4): uuid is now imported at the top
        sequence = Sequence(str(uuid.uuid4()), prompt, None, None)
        result = self.model_executor.execute(sequence)
        # FIX (5): removed the erroneous dot before the bracket
        return result["generated_text"]


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

_llm_engine: LLMEngine | None = None


def get_llm() -> LLMEngine:
    # FIX (14): get_llm defined so Depends(get_llm) works
    global _llm_engine
    if _llm_engine is None:
        _llm_engine = LLMEngine()
    return _llm_engine


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

# FIX (15): response_model and the returned class now both use BasicGenerateResponse
# FIX (16): request/response classes inherit from pydantic BaseModel (defined above)
@app.post("/basic_generate", response_model=BasicGenerateResponse)
async def basic_generate(
    request: BasicGenerateRequest,
    llm: LLMEngine = Depends(get_llm),
):
    generated_text = llm.basic_generate(request.prompt)
    return BasicGenerateResponse(generated_text=generated_text)