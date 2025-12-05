"""FastAPI service exposing the fine-tuned Mistral chat endpoint.

Run with:
    uvicorn backend.main:app --reload --port 2347

Environment variables:
    BASE_MODEL_NAME      default "mistralai/Mistral-7B-v0.1"
    ADAPTER_PATH         default "./mistral-indian-law-final"
    DEVICE_MAP           default "auto"
    MAX_NEW_TOKENS       default 256
    TEMPERATURE          default 0.7
    TOP_P                default 0.9
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import traceback
from functools import lru_cache
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
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
SYSTEM_PROMPT = """You are a professional Indian Law Assistant specialized in Indian legal system, laws, procedures, and jurisprudence. Your role is to provide accurate, comprehensive answers to legal questions in a question-answer format.

CORE FUNCTIONALITY:
- Answer ONLY the question that is asked - do not generate additional questions
- Provide accurate information about Indian laws, legal procedures, and jurisprudence
- Use proper legal terminology and citations when relevant
- Structure answers clearly with proper formatting
- Focus on factual information from Indian legal system
- STOP after answering the question - do not continue generating

RESPONSE FORMAT:
- Answer the question directly without preamble
- Do not include system tags, role indicators, or metadata in your response
- Provide clear, structured answers with proper legal citations when applicable
- Use bullet points or numbered lists for complex answers
- Keep responses focused and relevant to the question asked

GREETING BEHAVIOR:
- When users greet you, respond briefly: "Hello! I'm your Indian Law Assistant. I can help you with questions about Indian legal matters including constitutional law, criminal law, civil law, and legal procedures. What would you like to know?"
- Keep greeting responses to 1-2 sentences maximum
- STOP after the greeting - do not generate additional content

IMPORTANT RULES:
- NEVER include [System], [User], [Assistant] tags or any metadata in your responses
- ALWAYS answer questions directly without showing internal system messages
- ALWAYS leave one line end every response with: "I may be incorrect. For accurate and verified legal advice, please consult a qualified lawyer." 
- NEVER generate additional questions after answering - only answer what is asked
- NEVER continue generating Q&A pairs after your response
- STOP immediately after providing the answer and disclaimer
- NEVER provide legal advice as professional counsel - only provide informational answers
- ALWAYS acknowledge when you don't know something or are uncertain
- ALWAYS redirect off-topic questions back to Indian legal matters
- NEVER generate random or irrelevant responses to greetings or questions
- ALWAYS maintain professional, respectful tone
- ALWAYS cite relevant sections, acts, or legal provisions when providing specific legal information"""


class Message(BaseModel):
    role: str = Field(pattern=r"^(system|assistant|user)$")
    content: str


class ChatRequest(BaseModel):
    prompt: Optional[str] = None
    messages: List[Message] = Field(default_factory=list)
    max_new_tokens: int = Field(default_factory=lambda: get_config().max_new_tokens, ge=1, le=1024)
    temperature: float = Field(default_factory=lambda: get_config().temperature, ge=0, le=2)
    top_p: float = Field(default_factory=lambda: get_config().top_p, ge=0.1, le=1.0)


class ChatResponse(BaseModel):
    reply: str




class ServiceConfig(BaseModel):
    base_model_name: str = "mistralai/Mistral-7B-v0.1"
    adapter_path: str = "./mistral-indian-law-final"
    device_map: str = "auto"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

    @property
    def quant_config(self) -> BitsAndBytesConfig:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )


@lru_cache
def get_config() -> ServiceConfig:
    """Load and log service configuration from environment variables."""
    logger.info("Loading service configuration...")
    defaults = ServiceConfig()
    
    config = ServiceConfig(
        base_model_name=os.getenv("BASE_MODEL_NAME", defaults.base_model_name),
        adapter_path=os.getenv("ADAPTER_PATH", defaults.adapter_path),
        device_map=os.getenv("DEVICE_MAP", defaults.device_map),
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", defaults.max_new_tokens)),
        temperature=float(os.getenv("TEMPERATURE", defaults.temperature)),
        top_p=float(os.getenv("TOP_P", defaults.top_p)),
    )
    
    logger.info(f"Configuration loaded:")
    logger.info(f"  - Base Model: {config.base_model_name}")
    logger.info(f"  - Adapter Path: {config.adapter_path}")
    logger.info(f"  - Device Map: {config.device_map}")
    logger.info(f"  - Max New Tokens: {config.max_new_tokens}")
    logger.info(f"  - Temperature: {config.temperature}")
    logger.info(f"  - Top P: {config.top_p}")
    logger.info(f"  - CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  - CUDA Device Count: {torch.cuda.device_count()}")
        logger.info(f"  - CUDA Device Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"  - CUDA Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    return config


class ModelBundle:
    def __init__(self, config: ServiceConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._lock = asyncio.Lock()

    async def ensure_loaded(self) -> None:
        """Ensure model and tokenizer are loaded with detailed logging."""
        if self._model is not None and self._tokenizer is not None:
            logger.debug("Model and tokenizer already loaded, skipping initialization")
            return
        
        async with self._lock:
            if self._model is not None and self._tokenizer is not None:
                logger.debug("Model and tokenizer loaded by another coroutine")
                return
            
            load_start_time = time.time()
            logger.info("=" * 80)
            logger.info("Starting model initialization...")
            logger.info(f"Adapter path: {self.config.adapter_path}")
            logger.info(f"Base model: {self.config.base_model_name}")
            
            try:
                # Load tokenizer
                logger.info("Loading tokenizer...")
                tokenizer_start = time.time()
                self._tokenizer = AutoTokenizer.from_pretrained(self.config.adapter_path)
                tokenizer_time = time.time() - tokenizer_start
                logger.info(f"Tokenizer loaded successfully in {tokenizer_time:.2f}s")
                logger.info(f"Tokenizer vocab size: {len(self._tokenizer)}")
                
                # Load base model
                logger.info("Loading base model...")
                base_model_start = time.time()
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model_name,
                    quantization_config=self.config.quant_config,
                    device_map=self.config.device_map,
                    offload_folder="offload_dir",
                    offload_state_dict=True,
                    low_cpu_mem_usage=True,
                )
                base_model_time = time.time() - base_model_start
                logger.info(f"Base model loaded successfully in {base_model_time:.2f}s")
                
                # Log model device placement
                if hasattr(base_model, 'hf_device_map'):
                    logger.info(f"Model device map: {base_model.hf_device_map}")
                else:
                    device = next(base_model.parameters()).device
                    logger.info(f"Model device: {device}")
                
                # Load adapter
                logger.info("Loading PEFT adapter...")
                adapter_start = time.time()
                self._model = PeftModel.from_pretrained(base_model, self.config.adapter_path)
                adapter_time = time.time() - adapter_start
                logger.info(f"PEFT adapter loaded successfully in {adapter_time:.2f}s")
                
                # Set to eval mode
                self._model.eval()
                logger.info("Model set to evaluation mode")
                
                total_load_time = time.time() - load_start_time
                logger.info(f"Model initialization completed in {total_load_time:.2f}s")
                
                # Log memory usage
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
                    logger.info(f"CUDA Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")
                
                logger.info("=" * 80)
                
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                logger.error(f"Error type: {type(e).__name__}")
                logger.error(f"Traceback:\n{traceback.format_exc()}")
                raise

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
        """Generate response with comprehensive logging."""
        generation_start_time = time.time()
        
        logger.debug("Starting generation process...")
        logger.debug(f"Generation parameters:")
        logger.debug(f"  - Max new tokens: {max_new_tokens}")
        logger.debug(f"  - Temperature: {temperature}")
        logger.debug(f"  - Top P: {top_p}")
        logger.debug(f"  - Prompt length: {len(prompt)} characters")
        
        await self.ensure_loaded()

        device = next(self.model.parameters()).device
        logger.debug(f"Using device: {device}")
        
        # Tokenize input
        tokenize_start = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_token_count = inputs["input_ids"].shape[1]
        tokenize_time = time.time() - tokenize_start
        logger.debug(f"Input tokenized in {tokenize_time:.3f}s - {input_token_count} tokens")
        
        # Move inputs to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate
        generate_start = time.time()
        logger.debug("Starting model generation...")
        try:
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
            generate_time = time.time() - generate_start
            logger.debug(f"Model generation completed in {generate_time:.3f}s")
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise
        
        # Decode output
        decode_start = time.time()
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_token_count = len(generated_tokens)
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        decode_time = time.time() - decode_start
        logger.debug(f"Output decoded in {decode_time:.3f}s - {generated_token_count} tokens generated")
        
        # Post-process to stop at question markers
        original_length = len(decoded)
        stop_markers = ["\n\nQuestion:", "\nQuestion:", "\n\nUser Question:", "\nUser Question:"]
        for marker in stop_markers:
            if marker in decoded:
                decoded = decoded.split(marker)[0].strip()
                logger.debug(f"Stopped generation at marker: {marker}")
                logger.debug(f"Truncated from {original_length} to {len(decoded)} characters")
                break
        
        total_generation_time = time.time() - generation_start_time
        tokens_per_second = generated_token_count / total_generation_time if total_generation_time > 0 else 0
        
        logger.info(f"Generation completed:")
        logger.info(f"  - Total time: {total_generation_time:.3f}s")
        logger.info(f"  - Input tokens: {input_token_count}")
        logger.info(f"  - Generated tokens: {generated_token_count}")
        logger.info(f"  - Output length: {len(decoded)} characters")
        logger.info(f"  - Tokens/second: {tokens_per_second:.2f}")
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            logger.debug(f"CUDA memory after generation: {memory_allocated:.2f} GB")
        
        return decoded


config = get_config()
bundle = ModelBundle(config)

app = FastAPI(
    title="Mistral Indian Law Chat API",
    version="1.0.0",
    description="FastAPI backend for fine-tuned Mistral 7B model specialized in Indian law",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with detailed information."""
    request_start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    client_port = request.client.port if request.client else None
    
    # Log request details
    logger.info("=" * 80)
    logger.info(f"INCOMING REQUEST")
    logger.info(f"  Method: {request.method}")
    logger.info(f"  Path: {request.url.path}")
    logger.info(f"  Query: {request.url.query if request.url.query else 'None'}")
    logger.info(f"  Client IP: {client_ip}:{client_port}" if client_port else f"  Client IP: {client_ip}")
    logger.info(f"  User-Agent: {request.headers.get('user-agent', 'Unknown')}")
    logger.info(f"  Content-Type: {request.headers.get('content-type', 'None')}")
    logger.info(f"  Content-Length: {request.headers.get('content-length', 'Unknown')} bytes")
    
    # Process request
    try:
        response = await call_next(request)
        request_time = time.time() - request_start_time
        
        # Log response details
        logger.info(f"RESPONSE")
        logger.info(f"  Status: {response.status_code}")
        logger.info(f"  Time: {request_time:.3f}s")
        if hasattr(response, 'headers'):
            logger.debug(f"  Headers: {dict(response.headers)}")
        logger.info("=" * 80)
        
        return response
    except Exception as e:
        request_time = time.time() - request_start_time
        logger.error(f"REQUEST FAILED after {request_time:.3f}s")
        logger.error(f"  Error: {str(e)}")
        logger.error(f"  Error type: {type(e).__name__}")
        logger.error(f"  Traceback:\n{traceback.format_exc()}")
        logger.info("=" * 80)
        raise


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Custom 404 handler with helpful message."""
    logger.warning(f"404 Not Found: {request.method} {request.url.path} from {request.client.host if request.client else 'unknown'}")
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"Route {request.url.path} not found",
            "available_routes": ["/", "/health", "/chat", "/docs", "/redoc"]
        }
    )


@app.on_event("startup")
async def preload_model() -> None:
    """Preload model on startup with logging."""
    logger.info("=" * 80)
    logger.info("FastAPI application starting up...")
    logger.info("Preloading model and tokenizer...")
    try:
        await bundle.ensure_loaded()
        logger.info("Model preloaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to preload model on startup: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise
    logger.info("=" * 80)


@app.get("/", summary="API root")
async def root():
    """API root endpoint with available routes information."""
    return {
        "service": "Mistral Indian Law Chat API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "GET /health",
            "chat": "POST /chat",
            "docs": "GET /docs",
            "redoc": "GET /redoc"
        },
        "base_url": "http://localhost:2347"
    }


@app.get("/health", summary="Health check")
async def healthcheck():
    """Check if the service is running and model is loaded."""
    logger.debug("Health check endpoint called")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_loaded = bundle._model is not None
    tokenizer_loaded = bundle._tokenizer is not None
    
    health_info = {
        "status": "ok" if model_loaded else "loading",
        "device": device,
        "model_loaded": model_loaded,
        "tokenizer_loaded": tokenizer_loaded
    }
    
    if torch.cuda.is_available():
        health_info["cuda_memory_gb"] = round(torch.cuda.memory_allocated(0) / 1024**3, 2)
        health_info["cuda_device"] = torch.cuda.get_device_name(0)
    
    logger.debug(f"Health check result: {health_info}")
    return health_info


def build_prompt(payload: ChatRequest) -> str:
    """Build prompt from request payload with logging."""
    logger.debug("Building prompt from request payload...")
    parts: List[str] = [SYSTEM_PROMPT]
    
    if payload.prompt:
        logger.debug(f"Using single prompt field (length: {len(payload.prompt)} chars)")
        parts.append(f"\n\nUser Question: {payload.prompt}\n\nProvide a direct answer to ONLY this question. Answer it completely and then STOP. Do not generate any additional questions, answers, or content after your response.\n\nAnswer:")
        prompt_text = "\n".join(parts)
        logger.debug(f"Built prompt with single question (total length: {len(prompt_text)} chars)")
        return prompt_text

    if not payload.messages:
        logger.error("No prompt or messages provided in request")
        raise HTTPException(status_code=400, detail="No prompt or messages provided")

    logger.debug(f"Using messages array ({len(payload.messages)} messages)")
    # Add conversation history
    parts.append("\n\n")
    for i, message in enumerate(payload.messages):
        logger.debug(f"  Message {i+1}: role={message.role}, content_length={len(message.content)}")
        if message.role == "user":
            parts.append(f"User Question: {message.content}\n\n")
        elif message.role == "assistant":
            parts.append(f"Answer: {message.content}\n\n")
        elif message.role == "system":
            logger.debug(f"  Skipping system message at index {i}")
            continue
    
    # Add prompt for next answer with explicit stop instruction
    parts.append("Provide a direct answer to ONLY this question. Answer it completely and then STOP. Do not generate any additional questions, answers, or content after your response.\n\nAnswer:")
    
    prompt_text = "\n".join(parts)
    logger.debug(f"Built prompt from messages array (total length: {len(prompt_text)} chars)")
    return prompt_text


@app.post("/chat", response_model=ChatResponse, summary="Chat with the fine-tuned model")
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """
    Generate a response from the fine-tuned Mistral Indian Law model.
    
    You can provide either:
    - A simple `prompt` string, or
    - A `messages` array for conversation history
    
    Generation parameters can be customized per request.
    """
    chat_start_time = time.time()
    client_ip = request.client.host if request.client else "unknown"
    
    logger.info("=" * 80)
    logger.info("CHAT REQUEST RECEIVED")
    logger.info(f"  Client: {client_ip}")
    logger.info(f"  Request type: {'prompt' if payload.prompt else 'messages'}")
    
    if payload.prompt:
        logger.info(f"  Prompt: {payload.prompt[:100]}{'...' if len(payload.prompt) > 100 else ''}")
    else:
        logger.info(f"  Messages count: {len(payload.messages)}")
        for i, msg in enumerate(payload.messages):
            logger.info(f"    [{i+1}] {msg.role}: {msg.content[:50]}{'...' if len(msg.content) > 50 else ''}")
    
    logger.info(f"  Generation parameters:")
    logger.info(f"    - max_new_tokens: {payload.max_new_tokens}")
    logger.info(f"    - temperature: {payload.temperature}")
    logger.info(f"    - top_p: {payload.top_p}")
    
    try:
        prompt = build_prompt(payload)
        logger.debug(f"Built prompt (length: {len(prompt)} chars)")
        
        reply = await bundle.generate(
            prompt=prompt,
            max_new_tokens=payload.max_new_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
        )
        
        logger.info(f"Generation successful - Response length: {len(reply)} chars")
        logger.debug(f"Response preview: {reply[:200]}{'...' if len(reply) > 200 else ''}")
        
    except RuntimeError as exc:
        chat_time = time.time() - chat_start_time
        logger.error(f"RUNTIME ERROR during generation (after {chat_time:.3f}s)")
        logger.error(f"  Error: {str(exc)}")
        logger.error(f"  Error type: {type(exc).__name__}")
        logger.error(f"  Traceback:\n{traceback.format_exc()}")
        logger.info("=" * 80)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except HTTPException:
        # Re-raise HTTP exceptions without logging (they're expected)
        raise
    except Exception as exc:
        chat_time = time.time() - chat_start_time
        logger.error(f"UNEXPECTED ERROR during generation (after {chat_time:.3f}s)")
        logger.error(f"  Error: {str(exc)}")
        logger.error(f"  Error type: {type(exc).__name__}")
        logger.error(f"  Traceback:\n{traceback.format_exc()}")
        logger.info("=" * 80)
        raise HTTPException(status_code=500, detail=f"Generation error: {str(exc)}") from exc

    # Add disclaimer to every response
    disclaimer = "\n\nPlease note: I am an AI assistant and may provide incorrect information. Please consult a qualified professional lawyer for accurate legal advice and verification."
    
    # Check if disclaimer is already in the response (to avoid duplication)
    if disclaimer.strip() not in reply:
        reply = reply + disclaimer
        logger.debug("Disclaimer appended to response")
    else:
        logger.debug("Disclaimer already present in response, skipping append")
    
    total_chat_time = time.time() - chat_start_time
    logger.info(f"CHAT REQUEST COMPLETED in {total_chat_time:.3f}s")
    logger.info(f"  Final response length: {len(reply)} chars")
    logger.info("=" * 80)
    
    return ChatResponse(reply=reply)

