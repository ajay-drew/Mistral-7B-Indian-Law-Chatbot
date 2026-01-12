"""FastAPI service exposing the fine-tuned Mistral chat endpoint with RAG.

Run with:
    uvicorn backend.main:app --reload --port 2347

Environment variables:
    BASE_MODEL_NAME      default "mistralai/Mistral-7B-v0.1"
    ADAPTER_PATH         default "./mistral-indian-law-final"
    DEVICE_MAP           default "auto"
    MAX_NEW_TOKENS       default 256
    TEMPERATURE          default 0.7
    TOP_P                default 0.9
    RAG_EMBEDDING_MODEL  default "sentence-transformers/all-MiniLM-L6-v2"
    RAG_PERSIST_DIR      default "./data/chroma_db"
    RAG_TOP_K            default 3
    RAG_CHUNK_SIZE       default 1500 (increased for legal docs)
    RAG_CHUNK_OVERLAP    default 400 (increased for context)
    RAG_MIN_RELEVANCE    default 0.35 (minimum relevance threshold)
    RAG_HYBRID_ALPHA     default 0.6 (semantic vs BM25 weight)
    RAG_USE_RERANKER     default true (cross-encoder reranking)
    CORS_ORIGINS         default "http://localhost:5173,http://localhost:3000"
    API_KEY              default "" (No auth if empty)
    GENERATION_TIMEOUT   default 60 (seconds)
"""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import sys
import time
import traceback
import uuid
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Header, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

from backend.document_store import DocumentStore
from backend.exceptions import DocumentError, ValidationError
from backend.pdf_processor import extract_text_with_pages, chunk_text_with_pages
from backend.rag import RAGSystem
from backend.validation import (
    validate_pdf_file,
    validate_file_size,
)


# Configure extended logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# Set specific log levels for verbose modules
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("peft").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)

# System prompt for Indian Law Assistant
SYSTEM_PROMPT = """You are a professional Indian Law Assistant providing informational answers about Indian legal matters—constitutional law, criminal law, civil law, procedures, and jurisprudence.

ANSWERING QUESTIONS:
When document context is provided above, prioritize it as your primary source. Cite the document by referencing specific provisions or sections mentioned in the context. If the context is insufficient, supplement with your knowledge of Indian law.

When no document context is provided, answer based on your training in Indian legal matters. Cite relevant sections, acts, or provisions (e.g., "Section 302 IPC", "Article 21 of the Constitution").

RESPONSE FORMAT:
- Answer the question directly and completely
- Use proper legal terminology with citations
- For complex answers, use bullet points or numbered lists
- STOP immediately after providing the answer and disclaimer
- NEVER generate additional questions, queries, or new content after your response
- NEVER start a new line with "Query:", "Question:", or "User Question:"

GREETING BEHAVIOR:
When users greet you, respond: "Hello! I'm your Indian Law Assistant. I can help with questions about Indian legal matters including constitutional law, criminal law, civil law, and legal procedures. What would you like to know?"

IMPORTANT RULES:
- Never include system tags, role indicators, or metadata in responses
- Never provide legal advice as professional counsel—only informational answers
- Acknowledge uncertainty when you don't know something
- Redirect off-topic questions to Indian legal matters
- Maintain a professional, respectful tone

MANDATORY DISCLAIMER:
End every response with: "I may be incorrect. For accurate and verified legal advice, please consult a qualified lawyer."
"""


class Message(BaseModel):
    role: str = Field(pattern=r"^(system|assistant|user)$")
    content: str


class ChatRequest(BaseModel):
    prompt: Optional[str] = None
    messages: List[Message] = Field(default_factory=list)
    max_new_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)


class SourceCitation(BaseModel):
    """Detailed source citation for RAG responses."""
    filename: str
    page: int
    snippet: str
    relevance: float


class ChatResponse(BaseModel):
    reply: str
    sources: Optional[List[SourceCitation]] = None


class ServiceConfig(BaseModel):
    base_model_name: str = "mistralai/Mistral-7B-v0.1"
    adapter_path: str = "./mistral-indian-law-final"
    max_new_tokens: int = 256
    temperature: float = 0.2  # Lower for legal accuracy
    top_p: float = 0.9
    repetition_penalty: float = 1.25  # Increased to prevent Q&A pattern repetition
    rag_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_persist_dir: str = "./data/chroma_db"
    rag_top_k: int = 3
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200
    cors_origins: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    api_key: Optional[str] = None
    generation_timeout: int = 60  # seconds

    @property
    def quant_config(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    
    def get_device_map(self) -> str:
        """Get device map - 'auto' for low VRAM (<8GB), direct GPU otherwise."""
        if not torch.cuda.is_available():
            return "cpu"
        
        # Check available VRAM
        try:
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            # Use 'auto' for low VRAM systems to allow GPU/CPU split
            if total_vram_gb < 8.0:
                return "auto"
            return "cuda:0"
        except Exception:
            return "auto"
    
    def get_max_memory(self) -> dict:
        """Calculate dynamic memory limits based on actual available VRAM."""
        if not torch.cuda.is_available():
            return {}
        
        try:
            # Get actual VRAM
            props = torch.cuda.get_device_properties(0)
            total_vram_gb = props.total_memory / (1024**3)
            
            # Reserve 1.5GB safety margin for:
            # - CUDA overhead (~500MB)
            # - Inference activations (~500MB)
            # - System/display (~500MB)
            safety_margin_gb = 1.5
            usable_vram_gb = max(total_vram_gb - safety_margin_gb, 2.0)
            
            # Set CPU memory limit for offloading
            # Model loading requires significant virtual memory (RAM + page file)
            # We set a minimum of 10GB to ensure loading can complete
            cpu_memory_gb = 10.0  # Minimum needed for 7B model loading
            
            return {
                0: f"{usable_vram_gb:.1f}GiB",
                "cpu": f"{cpu_memory_gb:.1f}GiB"
            }
        except Exception as e:
            logger.warning(f"Could not calculate memory limits: {e}")
            return {0: "5GiB", "cpu": "10GiB"}  # Safe fallback


@lru_cache
def get_config() -> ServiceConfig:
    """Load and log service configuration from environment variables."""
    logger.info("Loading service configuration...")
    defaults = ServiceConfig()
    
    # Parse CORS origins
    cors_env = os.getenv("CORS_ORIGINS")
    cors_origins = cors_env.split(",") if cors_env else defaults.cors_origins

    config = ServiceConfig(
        base_model_name=os.getenv("BASE_MODEL_NAME", defaults.base_model_name),
        adapter_path=os.getenv("ADAPTER_PATH", defaults.adapter_path),
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", defaults.max_new_tokens)),
        temperature=float(os.getenv("TEMPERATURE", defaults.temperature)),
        top_p=float(os.getenv("TOP_P", defaults.top_p)),
        repetition_penalty=float(os.getenv("REPETITION_PENALTY", defaults.repetition_penalty)),
        rag_embedding_model=os.getenv("RAG_EMBEDDING_MODEL", defaults.rag_embedding_model),
        rag_persist_dir=os.getenv("RAG_PERSIST_DIR", defaults.rag_persist_dir),
        rag_top_k=int(os.getenv("RAG_TOP_K", defaults.rag_top_k)),
        rag_chunk_size=int(os.getenv("RAG_CHUNK_SIZE", defaults.rag_chunk_size)),
        rag_chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", defaults.rag_chunk_overlap)),
        cors_origins=cors_origins,
        api_key=os.getenv("API_KEY"),
        generation_timeout=int(os.getenv("GENERATION_TIMEOUT", defaults.generation_timeout))
    )
    
    logger.info(f"Config: model={config.base_model_name}, adapter={config.adapter_path}")
    logger.info(f"Device: {'cuda:0' if torch.cuda.is_available() else 'cpu'}")
    
    return config


class StopSequenceCriteria(StoppingCriteria):
    """Stop generation when specific sequences are detected."""
    def __init__(self, tokenizer, stop_sequences: List[str]):
        self.tokenizer = tokenizer
        # Convert stop sequences to token IDs
        self.stop_token_ids = []
        for seq in stop_sequences:
            tokens = tokenizer.encode(seq, add_special_tokens=False)
            if tokens:
                self.stop_token_ids.append(tokens)
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check if any stop sequence appears at the end
        for stop_tokens in self.stop_token_ids:
            if input_ids.shape[1] >= len(stop_tokens):
                # Get the last N tokens
                last_tokens = input_ids[0, -len(stop_tokens):].cpu().tolist()
                stop_tokens_list = stop_tokens if isinstance(stop_tokens, list) else stop_tokens.tolist() if hasattr(stop_tokens, 'tolist') else list(stop_tokens)
                if last_tokens == stop_tokens_list:
                    return True
        return False


class ModelBundle:
    def __init__(self, config: ServiceConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._load_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
        self._loading_failed = False  # Prevent retry loops on persistent failures
        
        # Stop sequences to prevent Q&A chain reactions
        self._stop_sequences = [
            "\n\nQuery:",
            "\nQuery:",
            "Query:",
            "\n\nUser Question:",
            "\nUser Question:",
            "User Question:",
            "\n\nQuestion:",
            "\nQuestion:",
            "Question:",
            "\n\nUser:",
            "\nUser:",
            "User:",
            "[/INST]",
            "\n\nAnswer:",
            "\nAnswer:",
            "Answer:",
            "\n\nStop answering",
            "\nStop answering",
            "Stop answering",
            "\n\nStop.",
            "\nStop.",
            "Stop.",
        ]
    
    def _log_gpu_memory(self, stage: str) -> None:
        """Log GPU memory usage at various stages."""
        if not torch.cuda.is_available():
            return
        try:
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            free = total - allocated
            logger.info(f"[{stage}] GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {free:.2f}GB free / {total:.2f}GB total")
        except Exception as e:
            logger.debug(f"Could not log GPU memory: {e}")
    
    def _cleanup_memory(self) -> None:
        """Comprehensive memory cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()  # Clear IPC memory
            except Exception:
                pass

    async def ensure_loaded(self) -> None:
        """Ensure model and tokenizer are loaded with comprehensive diagnostics."""
        if self._model is not None and self._tokenizer is not None:
            return
        
        # Prevent retry loops on persistent failures
        if self._loading_failed:
            raise RuntimeError("Model loading previously failed. Restart the service to retry.")
        
        async with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return
            
            load_start_time = time.time()
            
            try:
                # ===== PRE-LOAD DIAGNOSTICS =====
                logger.info("=" * 60)
                logger.info("Starting model loading sequence...")
                logger.info("=" * 60)
                
                # Log system info
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    logger.info(f"GPU: {gpu_name} ({total_vram:.1f}GB VRAM)")
                else:
                    logger.info("GPU: Not available, using CPU only")
                
                # Validate paths before loading
                adapter_path = Path(self.config.adapter_path)
                if not adapter_path.exists():
                    raise FileNotFoundError(f"Adapter path not found: {adapter_path.absolute()}")
                
                # Ensure offload directory exists
                offload_dir = Path("offload_dir")
                offload_dir.mkdir(exist_ok=True)
                
                # ===== STEP 1: CLEAR MEMORY =====
                logger.info("[1/5] Clearing memory...")
                self._cleanup_memory()
                self._log_gpu_memory("Pre-load")
                
                # ===== STEP 2: LOAD TOKENIZER =====
                logger.info("[2/5] Loading tokenizer...")
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(self.config.adapter_path)
                    logger.info(f"Loaded tokenizer from adapter path")
                except Exception:
                    self._tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)
                    logger.info(f"Loaded tokenizer from base model")
                
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                
                # ===== STEP 3: LOAD BASE MODEL =====
                logger.info("[3/5] Loading base model with 4-bit quantization...")
                device_map = self.config.get_device_map()
                max_memory = self.config.get_max_memory()
                
                logger.info(f"Device map: {device_map}")
                if max_memory:
                    logger.info(f"Memory limits: GPU={max_memory.get(0, 'auto')}, CPU={max_memory.get('cpu', 'auto')}")
                
                # Enable PyTorch optimizations
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                logger.info("PyTorch optimizations enabled (TF32, cuDNN benchmark)")
                
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model_name,
                    quantization_config=self.config.quant_config,
                    dtype=torch.float16,
                    device_map=device_map,
                    max_memory=max_memory if max_memory else None,
                    low_cpu_mem_usage=True,
                    offload_folder="offload_dir",
                    trust_remote_code=False,
                )
                self._log_gpu_memory("After base model")
                
                # ===== STEP 4: LOAD ADAPTER =====
                logger.info("[4/5] Loading LoRA adapter...")
                self._model = PeftModel.from_pretrained(base_model, self.config.adapter_path)
                self._model.eval()
                
                # Try to compile model for optimization
                try:
                    self._model = torch.compile(self._model, mode="reduce-overhead")
                    logger.info("Model compiled with torch.compile for optimization")
                except Exception as e:
                    logger.warning(f"torch.compile failed (non-critical): {e}")
                    # Continue without compilation
                
                self._log_gpu_memory("After adapter")
                
                # ===== STEP 5: WARMUP INFERENCE =====
                logger.info("[5/5] Running warmup inference...")
                try:
                    warmup_input = self._tokenizer("Test", return_tensors="pt")
                    device = next(self._model.parameters()).device
                    warmup_input = {k: v.to(device) for k, v in warmup_input.items()}
                    with torch.inference_mode():
                        self._model.generate(**warmup_input, max_new_tokens=1, pad_token_id=self._tokenizer.eos_token_id)
                    self._cleanup_memory()
                    logger.info("Warmup successful - model ready for inference")
                except Exception as e:
                    logger.warning(f"Warmup failed (non-critical): {e}")
                
                # ===== LOAD COMPLETE =====
                load_time = time.time() - load_start_time
                self._log_gpu_memory("Final")
                logger.info("=" * 60)
                logger.info(f"Model loaded successfully in {load_time:.1f}s")
                logger.info("=" * 60)
                
            except torch.cuda.OutOfMemoryError as e:
                self._loading_failed = True
                self._log_gpu_memory("OOM Error")
                logger.critical("\n" + "!" * 80)
                logger.critical("CUDA OUT OF MEMORY ERROR")
                logger.critical("!" * 80)
                logger.critical(f"Your GPU ran out of memory during model loading.")
                logger.critical("Possible solutions:")
                logger.critical("1. Close other GPU applications (browsers, games, other AI models)")
                logger.critical("2. Reduce max_memory in config (currently may be too high)")
                logger.critical("3. Use a smaller model or more aggressive quantization")
                logger.critical("4. The model may be too large for your GPU")
                logger.critical("!" * 80 + "\n")
                self._cleanup_failed_load()
                raise RuntimeError(f"CUDA OOM: {e}") from e
                
            except OSError as e:
                self._loading_failed = True
                error_msg = str(e)
                if "1455" in error_msg or "paging file" in error_msg.lower():
                    logger.critical("\n" + "!" * 80)
                    logger.critical("WINDOWS PAGE FILE TOO SMALL")
                    logger.critical("!" * 80)
                    logger.critical("Your system ran out of virtual memory.")
                    logger.critical("To fix this:")
                    logger.critical("1. Press Win+R, type 'sysdm.cpl', press Enter")
                    logger.critical("2. Advanced tab → Performance Settings → Advanced → Virtual Memory → Change")
                    logger.critical("3. Uncheck 'Automatically manage paging file size'")
                    logger.critical("4. Select C: drive → Custom size")
                    logger.critical("5. Initial: 24000 MB, Maximum: 32000 MB")
                    logger.critical("6. Click Set → OK → Restart computer")
                    logger.critical("!" * 80 + "\n")
                self._cleanup_failed_load()
                raise RuntimeError(f"OS Error: {e}") from e
                
            except FileNotFoundError as e:
                self._loading_failed = True
                logger.critical(f"File not found: {e}")
                logger.critical("Check that BASE_MODEL_NAME and ADAPTER_PATH are correct")
                self._cleanup_failed_load()
                raise RuntimeError(f"File not found: {e}") from e
                
            except Exception as e:
                self._loading_failed = True
                logger.error(f"Failed to load model: {e}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                self._cleanup_failed_load()
                raise RuntimeError(f"Failed to load model: {e}") from e
    
    def _cleanup_failed_load(self) -> None:
        """Clean up after a failed load attempt."""
        logger.info("Cleaning up after failed load...")
        self._model = None
        self._tokenizer = None
        self._cleanup_memory()

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Model not loaded")
        return self._model

    async def generate(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
        """Generate response from the model with timeout and concurrency control."""
        await self.ensure_loaded()
        
        async with self._inference_lock:
            try:
                return await asyncio.wait_for(
                    self._generate_internal(prompt, max_new_tokens, temperature, top_p),
                    timeout=self.config.generation_timeout
                )
            except asyncio.TimeoutError:
                logger.error("Generation timed out")
                raise HTTPException(status_code=504, detail="Generation timed out. Please try a shorter query.")
            except Exception as e:
                logger.error(f"Generation error: {e}")
                raise

    async def _generate_internal(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
        """Internal generation logic running in a thread to avoid blocking loop."""
        def run_inference():
            device = next(self.model.parameters()).device
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Create stopping criteria to prevent Q&A chain reactions
            stopping_criteria = StoppingCriteriaList([
                StopSequenceCriteria(self.tokenizer, self._stop_sequences)
            ])
            
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=self.config.repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria,
                )
            
            input_length = inputs["input_ids"].shape[1]
            decoded = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            
            # Post-processing: Aggressive stop marker filtering (backup)
            # Check for any continuation patterns and cut immediately
            stop_markers = self._stop_sequences + ["\n\n---\n"]
            
            # Find the earliest stop marker occurrence
            earliest_idx = len(decoded)
            for marker in stop_markers:
                idx = decoded.find(marker)
                if idx != -1 and idx < earliest_idx:
                    earliest_idx = idx
            
            # If we found a stop marker, cut everything after it
            if earliest_idx < len(decoded):
                decoded = decoded[:earliest_idx].strip()
            
            # Cut everything after disclaimer if present (hard stop)
            disclaimer_end = "qualified lawyer."
            if disclaimer_end in decoded:
                idx = decoded.find(disclaimer_end) + len(disclaimer_end)
                decoded = decoded[:idx].strip()
            
            # Final check: Remove any trailing continuation patterns
            # This catches cases where the model starts generating new content
            lines = decoded.split('\n')
            cleaned_lines = []
            for line in lines:
                line_stripped = line.strip()
                # Stop if we see a line that looks like a new query/question
                if line_stripped.startswith(('Query:', 'Question:', 'User Question:', 'User:')):
                    break
                # Stop if we see "Stop answering" or similar phrases
                if any(phrase in line_stripped.lower() for phrase in ['stop answering', 'stop.', 'stop generating']):
                    break
                cleaned_lines.append(line)
            
            decoded = '\n'.join(cleaned_lines).strip()
            
            # Final cleanup: Remove any trailing "Stop answering" or similar phrases
            stop_phrases = ['stop answering', 'stop.', 'stop generating', 'stop responding']
            for phrase in stop_phrases:
                if decoded.lower().endswith(phrase.lower()):
                    # Find the last occurrence and remove it
                    idx = decoded.lower().rfind(phrase.lower())
                    if idx > 0:
                        decoded = decoded[:idx].strip()
                        break
            
            return decoded

        # Run blocking inference in a separate thread
        decoded = await asyncio.to_thread(run_inference)
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return decoded
    
    async def generate_stream(
        self, 
        prompt: str, 
        max_new_tokens: int, 
        temperature: float, 
        top_p: float
    ):
        """Stream tokens as they're generated."""
        await self.ensure_loaded()
        
        async with self._inference_lock:
            try:
                async for token in self._generate_stream_internal(prompt, max_new_tokens, temperature, top_p):
                    yield token
            except Exception as e:
                logger.error(f"Streaming generation error: {e}")
                raise
    
    async def _generate_stream_internal(
        self, 
        prompt: str, 
        max_new_tokens: int, 
        temperature: float, 
        top_p: float
    ):
        """Internal streaming generation logic using TextIteratorStreamer."""
        from transformers import TextIteratorStreamer
        import threading
        import queue
        
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Create streamer with queue for async compatibility
        token_queue = queue.Queue()
        exception_queue = queue.Queue()
        
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        # Create stopping criteria to prevent Q&A chain reactions
        stopping_criteria = StoppingCriteriaList([
            StopSequenceCriteria(self.tokenizer, self._stop_sequences)
        ])
        
        # Generation parameters
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": self.config.repetition_penalty,
            "pad_token_id": self.tokenizer.eos_token_id,
            "streamer": streamer,
            "stopping_criteria": stopping_criteria,
        }
        
        # Run generation in thread
        def run_generation():
            try:
                with torch.inference_mode():
                    self.model.generate(**generation_kwargs)
            except Exception as e:
                logger.error(f"Generation error in stream: {e}")
                exception_queue.put(e)
                streamer.end()
        
        thread = threading.Thread(target=run_generation, daemon=True)
        thread.start()
        
        # Yield tokens as they arrive (async-friendly) with batching for better UX
        generated_text = ""
        # Rolling buffer for pattern detection (last 50 chars)
        buffer_size = 50
        text_buffer = ""
        
        # Token batching for faster SSE delivery
        token_batch = []
        batch_size = 3  # Send every 3 tokens or when batch reaches certain size
        last_yield_time = time.time()
        batch_timeout = 0.05  # Send batch every 50ms max
        
        try:
            while True:
                # Check for exceptions
                if not exception_queue.empty():
                    raise exception_queue.get()
                
                # Check if thread is still alive
                if not thread.is_alive() and streamer.text_queue.empty():
                    # Flush any remaining tokens in batch
                    if token_batch:
                        yield ''.join(token_batch)
                    break
                
                # Get next token from streamer (reduced timeout for faster delivery)
                try:
                    new_text = streamer.text_queue.get(timeout=0.05)  # Reduced from 0.1s to 0.05s
                    if new_text is None:  # End signal
                        # Flush batch before ending
                        if token_batch:
                            yield ''.join(token_batch)
                        break
                    
                    if not new_text:
                        continue
                    
                    # Add to batch
                    token_batch.append(new_text)
                    
                    # Update buffers for stop detection
                    generated_text += new_text
                    text_buffer += new_text
                    # Keep buffer size manageable
                    if len(text_buffer) > buffer_size:
                        text_buffer = text_buffer[-buffer_size:]
                    
                    # Real-time stop marker detection with rolling buffer
                    full_text = generated_text
                    
                    # Check for stop patterns in full text (more reliable)
                    should_stop = False
                    earliest_idx = len(full_text)
                    
                    # Quick check: Look for stop patterns in recent buffer first (faster)
                    check_text = text_buffer.lower()
                    stop_patterns_quick = ['query:', 'question:', 'user:', 'stop answering', 'stop.', 'qualified lawyer.']
                    for pattern in stop_patterns_quick:
                        if pattern in check_text:
                            # Found in buffer, now check full text for exact position
                            should_stop = True
                            break
                    
                    # If quick check found something, do full check
                    if should_stop:
                        # Check all stop sequences (case-insensitive) and find earliest occurrence
                        for marker in self._stop_sequences + ["\n\n---\n"]:
                            marker_lower = marker.lower()
                            if marker_lower in full_text.lower():
                                idx = full_text.lower().find(marker_lower)
                                if idx > 0 and idx < earliest_idx:
                                    earliest_idx = idx
                    
                        # Also check for lines starting with query/question patterns or containing stop phrases
                        lines = full_text.split('\n')
                        for i, line in enumerate(lines):
                            line_stripped = line.strip().lower()
                            # Check for query/question patterns
                            if line_stripped.startswith(('query:', 'question:', 'user question:', 'user:')):
                                line_start = sum(len(l) + 1 for l in lines[:i])
                                if line_start < earliest_idx:
                                    earliest_idx = line_start
                                    break
                            # Check for "Stop answering" or similar phrases
                            if any(phrase in line_stripped for phrase in ['stop answering', 'stop.', 'stop generating', 'stop responding']):
                                line_start = sum(len(l) + 1 for l in lines[:i])
                                if line_start < earliest_idx:
                                    earliest_idx = line_start
                                    break
                    
                    # Check for disclaimer end (hard stop after "qualified lawyer.")
                    if "qualified lawyer." in full_text:
                        idx = full_text.find("qualified lawyer.") + len("qualified lawyer.")
                        if idx < earliest_idx:
                            earliest_idx = idx
                            should_stop = True
                    
                    # Decide when to yield batch
                    current_time = time.time()
                    time_since_yield = current_time - last_yield_time
                    should_yield_batch = (
                        len(token_batch) >= batch_size or  # Batch size reached
                        time_since_yield >= batch_timeout or  # Timeout reached
                        should_stop  # Stop pattern detected
                    )
                    
                    if should_yield_batch:
                        if token_batch:
                            batch_text = ''.join(token_batch)
                            
                            if should_stop:
                                # Calculate how much of batch to send before stop marker
                                batch_start_pos = len(generated_text) - len(batch_text)
                                if earliest_idx > batch_start_pos:
                                    # Stop marker is within or after this batch
                                    send_length = earliest_idx - batch_start_pos
                                    if send_length > 0:
                                        yield batch_text[:send_length]
                                else:
                                    # Stop marker was before this batch, don't send it
                                    pass
                                return
                            
                            yield batch_text
                            token_batch = []
                            last_yield_time = current_time
                    
                except queue.Empty:
                    # If we have tokens in batch and timeout reached, flush them
                    current_time = time.time()
                    if token_batch and (current_time - last_yield_time) >= batch_timeout:
                        yield ''.join(token_batch)
                        token_batch = []
                        last_yield_time = current_time
                    # Small delay to avoid busy waiting
                    await asyncio.sleep(0.005)  # Reduced from 0.01s to 0.005s
                    continue
        finally:
            # Wait for thread to finish (with timeout)
            thread.join(timeout=5)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


config = get_config()
bundle = ModelBundle(config)

# Initialize RAG system and document store
rag_system = RAGSystem(
    embedding_model=config.rag_embedding_model,
    persist_dir=Path(config.rag_persist_dir),
    top_k=config.rag_top_k
)

document_store = DocumentStore(store_path=Path("./data/documents.json"))

app = FastAPI(
    title="Mistral Indian Law Chat API",
    version="1.0.0",
    description="FastAPI backend for fine-tuned Mistral 7B model specialized in Indian law with RAG",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Rate Limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# API Key Dependency
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Required API Key verification."""
    if not config.api_key:
        # If no API key configured, allow all (for development)
        return x_api_key
    
    if not x_api_key or x_api_key != config.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key. Please provide a valid X-API-Key header."
        )
    return x_api_key


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with detailed information."""
    request_start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    
    logger.info(f"Request: {request.method} {request.url.path} from {client_ip}")
    
    try:
        response = await call_next(request)
        request_time = time.time() - request_start_time
        logger.info(f"Response: {response.status_code} in {request_time:.3f}s")
        return response
    except Exception as e:
        request_time = time.time() - request_start_time
        logger.error(f"Request failed after {request_time:.3f}s: {str(e)}")
        raise


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Custom 404 handler with helpful message."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"Route {request.url.path} not found",
            "available_routes": ["/", "/health", "/chat", "/documents/upload"]
        }
    )


@app.on_event("startup")
async def preload_model() -> None:
    """Preload model on startup with logging."""
    logger.info("FastAPI application starting up...")
    try:
        # Check if RAG persist dir exists/is accessible
        Path(config.rag_persist_dir).mkdir(parents=True, exist_ok=True)
        
        await bundle.ensure_loaded()
        logger.info("Model preloaded successfully on startup")
        logger.info(f"RAG system initialized with {document_store.count()} documents")
    except Exception as e:
        logger.info(f"Startup warning: Failed to preload model: {str(e)}")
        # Don't crash startup, let it fail gracefully on first request if needed


@app.get("/", summary="API root")
async def root(request: Request):
    """API root endpoint with available routes information."""
    base_url = str(request.base_url).rstrip("/")
    return {
        "service": "Mistral Indian Law Chat API",
        "version": "1.0.0",
        "status": "running",
            "endpoints": {
            "health": f"{base_url}/health",
            "chat": f"{base_url}/chat",
            "chat_stream": f"{base_url}/chat/stream",
            "documents": f"{base_url}/documents",
            "upload": f"{base_url}/documents/upload",
            "docs": f"{base_url}/docs"
        },
        "documents_count": document_store.count()
    }


@app.get("/health", summary="Health check")
async def healthcheck():
    """Check if the service is running and model is loaded."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_loaded = bundle._model is not None
    
    health_info = {
        "status": "ok" if model_loaded else "loading",
        "device": device,
        "model_loaded": model_loaded,
        "documents_count": document_store.count()
    }
    
    if torch.cuda.is_available():
        try:
            health_info["cuda_memory_gb"] = round(torch.cuda.memory_allocated(0) / 1024**3, 2)
            health_info["cuda_device"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    
    return health_info


def build_prompt(payload: ChatRequest, rag_context: str = "", has_rag: bool = False) -> str:
    """Build prompt from request payload with optional RAG context."""
    parts: List[str] = [SYSTEM_PROMPT]
    
    # Extract user question
    if payload.prompt:
        user_question = payload.prompt
    elif payload.messages:
        user_question = None
        for message in reversed(payload.messages):
            if message.role == "user":
                user_question = message.content
                break
        if not user_question:
            raise HTTPException(status_code=400, detail="No user question found in messages")
    else:
        raise HTTPException(status_code=400, detail="No prompt or messages provided")
    
    # Add RAG context BEFORE question if available (better attention)
    if has_rag and rag_context:
        parts.append(f"\n\nDOCUMENT CONTEXT:\n{rag_context}")
    
    # Add user question with simple prompt (changed from "User Question:" to avoid triggering continuation)
    parts.append(f"\n\nQuery: {user_question}\n\nAnswer:")
    
    return "\n".join(parts)


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """
    Generate a response from the fine-tuned Mistral Indian Law model.
    Uses RAG if documents are available, otherwise uses base model knowledge.
    """
    chat_start_time = time.time()
    source_filenames = []
    rag_context = ""
    
    # Extract user question
    user_question = payload.prompt
    if not user_question and payload.messages:
        for message in reversed(payload.messages):
            if message.role == "user":
                user_question = message.content
                break
    
    if not user_question:
        raise HTTPException(status_code=400, detail="No user question provided")
    
    try:
        # RAG Retrieval (optional - only if documents are available)
        has_documents = document_store.count() > 0
        if has_documents:
            try:
                chunks = rag_system.search(user_question, top_k=config.rag_top_k)
                rag_context = rag_system.format_context(chunks)
                
                # Resolve source filenames
                document_ids = {chunk['metadata']['document_id'] for chunk in chunks if chunk.get('metadata') and 'document_id' in chunk['metadata']}
                for doc_id in document_ids:
                    doc = document_store.get_by_rag_id(doc_id)
                    if doc:
                        source_filenames.append(doc.get('filename', doc_id))
            except Exception as e:
                logger.warning(f"RAG retrieval failed, continuing without RAG: {e}")
                rag_context = ""
        
        prompt = build_prompt(payload, rag_context, has_rag=has_documents and rag_context)
        
        reply = await bundle.generate(
            prompt=prompt,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
        )
        
        return ChatResponse(reply=reply, sources=source_filenames if source_filenames else None)
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Unexpected error in chat: {exc}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error during response generation")


@app.post("/chat/stream", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def chat_stream(payload: ChatRequest, request: Request):
    """
    Stream response tokens as they're generated (Server-Sent Events).
    Uses RAG if documents are available, otherwise uses base model knowledge.
    """
    import json
    
    # Extract user question
    user_question = payload.prompt
    if not user_question and payload.messages:
        for message in reversed(payload.messages):
            if message.role == "user":
                user_question = message.content
                break
    
    if not user_question:
        raise HTTPException(status_code=400, detail="No user question provided")
    
    try:
        # RAG Retrieval (optional - only if documents are available)
        has_documents = document_store.count() > 0
        rag_context = ""
        source_filenames = []
        
        if has_documents:
            try:
                chunks = rag_system.search(user_question, top_k=config.rag_top_k)
                rag_context = rag_system.format_context(chunks)
                
                # Resolve source filenames
                document_ids = {chunk['metadata']['document_id'] for chunk in chunks if chunk.get('metadata') and 'document_id' in chunk['metadata']}
                for doc_id in document_ids:
                    doc = document_store.get_by_rag_id(doc_id)
                    if doc:
                        source_filenames.append(doc.get('filename', doc_id))
            except Exception as e:
                logger.warning(f"RAG retrieval failed, continuing without RAG: {e}")
                rag_context = ""
        
        prompt = build_prompt(payload, rag_context, has_rag=has_documents and rag_context)
        
        async def generate():
            """Generate and stream tokens with optimized batching."""
            full_response = ""
            try:
                async for token_batch in bundle.generate_stream(
                    prompt=prompt,
                    max_new_tokens=payload.max_new_tokens,
                    temperature=payload.temperature,
                    top_p=payload.top_p,
                ):
                    full_response += token_batch
                    # Format as Server-Sent Events (batched tokens for faster delivery)
                    yield f"data: {json.dumps({'token': token_batch, 'done': False})}\n\n"
                
                # Send sources if available
                if source_filenames:
                    yield f"data: {json.dumps({'sources': source_filenames, 'done': False})}\n\n"
                
                # Send completion signal
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Unexpected error in chat stream: {exc}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error during streaming")


# Document Management

class DocumentResponse(BaseModel):
    id: str
    filename: str
    upload_date: str
    file_size: int
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int


@app.post("/documents/upload", response_model=DocumentResponse, dependencies=[Depends(verify_api_key)])
async def upload_document(file: UploadFile = File(...)) -> DocumentResponse:
    """Upload a PDF document for RAG indexing with rollback support."""
    logger.info(f"Document upload request: {file.filename}")
    
    # Validation
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Read and validate size
    try:
        file_content = await file.read()
        validate_file_size(file_content=file_content, max_size_mb=10)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    rag_id = str(uuid.uuid4())
    doc_id = None
    
    try:
        logger.info(f"Extracting text from PDF: {file.filename} ({len(file_content)} bytes)")
        # Extract with page information preserved
        pages_data = extract_text_with_pages(file_content)
        total_chars = sum(len(p['text']) for p in pages_data)
        logger.info(f"Extracted {total_chars} characters from {len(pages_data)} pages")
        
        # Chunk with page-aware chunking
        chunks = chunk_text_with_pages(
            pages_data, 
            chunk_size=config.rag_chunk_size, 
            overlap=config.rag_chunk_overlap
        )
        logger.info(f"Created {len(chunks)} chunks from document")
        
        if not chunks:
            raise ValidationError("No text could be extracted from this document")

        # 1. Add to RAG (Expensive operation) with filename for citations
        logger.info(f"Adding {len(chunks)} chunks to RAG system with ID: {rag_id}")
        rag_system.add_documents(chunks, rag_id, filename=file.filename)
        logger.info("Successfully added chunks to RAG system")
        
        # 2. Add to Metadata Store
        try:
            doc_id = document_store.create(
                filename=file.filename,
                rag_id=rag_id,
                file_size=len(file_content),
                chunk_count=len(chunks)
            )
            logger.info(f"Document metadata saved with ID: {doc_id}")
        except Exception as e:
            # Rollback RAG if metadata save fails
            logger.error(f"Metadata save failed, rolling back RAG entry: {e}")
            rag_system.delete_document(rag_id)
            raise
            
        return DocumentResponse(
            id=doc_id,
            filename=file.filename,
            upload_date=document_store.get(doc_id)['upload_date'],
            file_size=len(file_content),
            chunk_count=len(chunks)
        )
        
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except DocumentError as e:
        logger.error(f"Document processing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload failed with exception: {e}")
        logger.error(traceback.format_exc())
        # Attempt cleanup if something failed vaguely
        if rag_id:
            try:
                rag_system.delete_document(rag_id)
                logger.info(f"Cleaned up RAG entry: {rag_id}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup RAG entry: {cleanup_error}")
        raise HTTPException(status_code=500, detail=f"Internal upload error: {str(e)}")


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    """Get a list of all uploaded documents."""
    documents = document_store.get_all()
    doc_responses = [
        DocumentResponse(
            id=doc['id'],
            filename=doc['filename'],
            upload_date=doc['upload_date'],
            file_size=doc['file_size'],
            chunk_count=doc['chunk_count']
        )
        for doc in documents
    ]
    return DocumentListResponse(documents=doc_responses, total=len(doc_responses))


@app.delete("/documents/{doc_id}", dependencies=[Depends(verify_api_key)])
async def delete_document(doc_id: str) -> dict:
    """Delete a document and remove it from the RAG system."""
    doc = document_store.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    
    try:
        # Delete from RAG system
        rag_id = doc.get('rag_id')
        if rag_id:
            rag_system.delete_document(rag_id)
        
        # Delete from document store
        document_store.delete(doc_id)
        
        return {"message": "Document deleted successfully", "id": doc_id}
    except Exception as e:
        logger.error(f"Deletion failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")