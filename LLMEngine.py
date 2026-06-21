import torch
import multiprocessing as mp
import logging
import uuid
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from typing import List

logger = logging.getLogger(__name__)
app = FastAPI()


# --- Request / Response models ---

class GenerateRequest(BaseModel):
    prompt: str

class GenerateResponse(BaseModel):
    generated_text: str

class BatchGenerateRequest(BaseModel):
    prompts: List[str]

class BatchGenerateResponse(BaseModel):
    generated_texts: List[str]


# --- Sequence ---

class Sequence:
    def __init__(self, seq_id: str, prompt: str, max_tokens, stopping_criteria):
        self.seq_id = seq_id
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.stopping_criteria = stopping_criteria


# --- Model Worker (runs in a subprocess) ---

class ModelWorker:
    def __init__(self, model_name: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        # LLaMA-family models have no pad token by default; reuse EOS.
        # Left-padding is required for decoder-only models so that all real
        # tokens stay flush against the generation boundary.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

    @staticmethod
    def run(model_name: str, task_queue: mp.Queue, result_queue: mp.Queue):
        worker = ModelWorker(model_name)
        while True:
            logger.debug("ModelWorker: Waiting for task")
            sequences = task_queue.get()   # always a List[Sequence]
            try:
                generated = worker.generate(sequences)
                result_queue.put({"status": "completed", "generated": generated})
            except Exception as e:
                logger.exception("ModelWorker: Task error")
                result_queue.put({"status": "error", "generated": str(e)})
            logger.debug("ModelWorker: Task completed")

    def generate(self, sequences: List[Sequence], max_new_tokens: int = 200) -> dict:
        prompt_txts = [seq.prompt for seq in sequences]
        prompt_ids  = [seq.seq_id  for seq in sequences]

        logger.info(f"batch_size={len(prompt_txts)}")

        # padding=True handles variable-length prompts in a batch.
        inputs = self.tokenizer(
            prompt_txts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        logger.info(f"input_ids shape={inputs['input_ids'].shape}")

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
            )

        logger.info(f"outputs shape={outputs.shape}")

        # Slice off the echoed prompt tokens so we only decode the new text.
        new_tokens = outputs[:, input_len:]

        logger.info(f"new_tokens shape={new_tokens.shape}")
        generated_texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        return {req_id: text for req_id, text in zip(prompt_ids, generated_texts)}


# --- Model Manager (placeholder) ---

class ModelManager:
    def __init__(self):
        pass


# --- Model Executor ---

class ModelExecutor:
    def __init__(self):
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()

    def setup_worker(self, model_name: str):
        self.worker_process = mp.Process(
            target=ModelWorker.run,
            args=(model_name, self.task_queue, self.result_queue),
        )
        self.worker_process.start()

    def execute(self, sequences: List[Sequence]) -> dict:
        """Send a batch (one or more sequences) and block until results arrive."""
        self.task_queue.put(sequences)
        return self.result_queue.get()


# --- Workload Manager (placeholder) ---

class WorkloadManager:
    def __init__(self):
        pass


# --- LLM Engine ---

class LLMEngine:
    def __init__(self):
        self.workload_manager = WorkloadManager()
        self.model_executor = ModelExecutor()
        self.max_tokens = 200
        self.model_executor.setup_worker("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    def basic_generate(self, prompt: str) -> str:
        sequence = Sequence(str(uuid.uuid4()), prompt, None, None)
        result = self.model_executor.execute([sequence])
        return result["generated"][sequence.seq_id]

    def batch_generate(self, prompts: List[str]) -> List[str]:
        sequences = [Sequence(str(uuid.uuid4()), p, None, None) for p in prompts]
        result = self.model_executor.execute(sequences)
        # Preserve original prompt order.
        return [result["generated"][seq.seq_id] for seq in sequences]


# --- Dependency ---

_llm_engine: LLMEngine | None = None

def get_llm() -> LLMEngine:
    global _llm_engine
    if _llm_engine is None:
        _llm_engine = LLMEngine()
    return _llm_engine


# --- Routes ---

@app.post("/basic_generate", response_model=GenerateResponse)
async def basic_generate(request: GenerateRequest, llm: LLMEngine = Depends(get_llm)):
    generated_text = llm.basic_generate(request.prompt)
    return GenerateResponse(generated_text=generated_text)


@app.post("/batch_generate", response_model=BatchGenerateResponse)
async def batch_generate(request: BatchGenerateRequest, llm: LLMEngine = Depends(get_llm)):
    generated_texts = llm.batch_generate(request.prompts)
    return BatchGenerateResponse(generated_texts=generated_texts)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
