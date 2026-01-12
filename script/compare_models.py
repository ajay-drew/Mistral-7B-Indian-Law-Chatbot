"""Compare base model vs fine-tuned model performance.

This script loads both the base Mistral-7B model and the fine-tuned model
(with LoRA adapter) and runs the same queries through both to compare
their responses, generation times, and quality.
"""

import json
import time
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# Test queries for Indian Law
TEST_QUERIES = [
    "What is Article 21 of the Indian Constitution?",
    "Explain Section 302 of the Indian Penal Code.",
    "What is the doctrine of basic structure?",
    "What are the fundamental rights in India?",
    "Explain the difference between murder and culpable homicide.",
    "What is the procedure for filing a writ petition?",
    "What is the difference between civil and criminal law?",
    "Explain the concept of judicial review in India.",
]


# System prompt (same as in backend/main.py)
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


def get_quant_config() -> BitsAndBytesConfig:
    """Get quantization configuration."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def get_device_map() -> str:
    """Get device map configuration."""
    if not torch.cuda.is_available():
        return "cpu"
    
    try:
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if total_vram_gb < 8.0:
            return "auto"
        return "cuda:0"
    except Exception:
        return "auto"


def get_max_memory() -> dict:
    """Get memory limits for model loading."""
    if not torch.cuda.is_available():
        return {}
    
    try:
        props = torch.cuda.get_device_properties(0)
        total_vram_gb = props.total_memory / (1024**3)
        safety_margin_gb = 1.5
        usable_vram_gb = max(total_vram_gb - safety_margin_gb, 2.0)
        cpu_memory_gb = 10.0
        
        return {
            0: f"{usable_vram_gb:.1f}GiB",
            "cpu": f"{cpu_memory_gb:.1f}GiB"
        }
    except Exception:
        return {0: "5GiB", "cpu": "10GiB"}


def load_base_model():
    """Load base model only (no adapter)."""
    print("=" * 80)
    print("Loading BASE MODEL (Mistral-7B-v0.1)...")
    print("=" * 80)
    
    base_model_name = "mistralai/Mistral-7B-v0.1"
    
    # Enable PyTorch optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.benchmark = True
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=get_quant_config(),
        device_map=get_device_map(),
        max_memory=get_max_memory() if get_max_memory() else None,
        low_cpu_mem_usage=True,
        offload_folder="offload_dir",
        trust_remote_code=False,
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✓ Base model loaded successfully!")
    print()
    return model, tokenizer


def load_finetuned_model():
    """Load base model with LoRA adapter."""
    print("=" * 80)
    print("Loading FINE-TUNED MODEL (Base + LoRA Adapter)...")
    print("=" * 80)
    
    base_model_name = "mistralai/Mistral-7B-v0.1"
    adapter_path = "./mistral-indian-law-final"
    
    if not Path(adapter_path).exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    
    # Enable PyTorch optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.benchmark = True
    
    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=get_quant_config(),
        device_map=get_device_map(),
        max_memory=get_max_memory() if get_max_memory() else None,
        low_cpu_mem_usage=True,
        offload_folder="offload_dir",
        trust_remote_code=False,
    )
    
    # Load adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        print("Loaded tokenizer from adapter path")
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        print("Loaded tokenizer from base model")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✓ Fine-tuned model loaded successfully!")
    print()
    return model, tokenizer


def build_prompt(user_question: str) -> str:
    """Build full prompt with system prompt."""
    parts = [
        SYSTEM_PROMPT,
        f"\n\nUser Question: {user_question}\n\nProvide a direct answer to ONLY this question based on your knowledge of Indian law. Answer it completely and then STOP. Do not generate any additional questions, answers, or content after your response.\n\nAnswer:"
    ]
    return "\n".join(parts)


def generate_response(model, tokenizer, prompt: str, max_tokens: int = 256):
    """Generate response from model."""
    full_prompt = build_prompt(prompt)
    
    inputs = tokenizer(full_prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    start_time = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    generation_time = time.time() - start_time
    
    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    # Clean up stop markers
    stop_markers = ["\n\nQuestion:", "\nQuestion:", "User:", "\nUser:", "[/INST]"]
    for marker in stop_markers:
        if marker in response:
            response = response.split(marker)[0].strip()
    
    return response, generation_time


def compare_models():
    """Compare base model vs fine-tuned model."""
    print("=" * 80)
    print("MODEL COMPARISON: Base vs Fine-tuned")
    print("=" * 80)
    print()
    print("This will compare the base Mistral-7B model against the")
    print("fine-tuned model (with LoRA adapter) on Indian Law queries.")
    print()
    print(f"Test queries: {len(TEST_QUERIES)}")
    print()
    
    results: List[Dict] = []
    
    try:
        # Load base model
        base_model, base_tokenizer = load_base_model()
        
        # Load fine-tuned model
        finetuned_model, finetuned_tokenizer = load_finetuned_model()
        
        # Test each query
        print("=" * 80)
        print("RUNNING COMPARISONS")
        print("=" * 80)
        print()
        
        for i, query in enumerate(TEST_QUERIES, 1):
            print(f"[{i}/{len(TEST_QUERIES)}] Query: {query}")
            print("-" * 80)
            
            # Generate with base model
            print("  Generating with base model...", end=" ", flush=True)
            try:
                base_response, base_time = generate_response(
                    base_model, base_tokenizer, query
                )
                print(f"✓ ({base_time:.2f}s)")
            except Exception as e:
                print(f"✗ Error: {e}")
                base_response = f"Error: {str(e)}"
                base_time = 0.0
            
            # Generate with fine-tuned model
            print("  Generating with fine-tuned model...", end=" ", flush=True)
            try:
                finetuned_response, finetuned_time = generate_response(
                    finetuned_model, finetuned_tokenizer, query
                )
                print(f"✓ ({finetuned_time:.2f}s)")
            except Exception as e:
                print(f"✗ Error: {e}")
                finetuned_response = f"Error: {str(e)}"
                finetuned_time = 0.0
            
            # Store results
            result = {
                "query": query,
                "base_model": {
                    "response": base_response,
                    "generation_time": base_time,
                    "response_length": len(base_response),
                    "word_count": len(base_response.split()),
                },
                "finetuned_model": {
                    "response": finetuned_response,
                    "generation_time": finetuned_time,
                    "response_length": len(finetuned_response),
                    "word_count": len(finetuned_response.split()),
                }
            }
            results.append(result)
            
            # Print side-by-side comparison
            print()
            print("  BASE MODEL RESPONSE:")
            print("  " + "-" * 76)
            base_preview = base_response[:300] + "..." if len(base_response) > 300 else base_response
            for line in base_preview.split('\n'):
                print(f"  {line}")
            print()
            print("  FINE-TUNED MODEL RESPONSE:")
            print("  " + "-" * 76)
            finetuned_preview = finetuned_response[:300] + "..." if len(finetuned_response) > 300 else finetuned_response
            for line in finetuned_preview.split('\n'):
                print(f"  {line}")
            print()
            print("=" * 80)
            print()
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Save results
        output_file = "model_comparison_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "test_queries": len(TEST_QUERIES),
                "results": results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Results saved to: {output_file}")
        print()
        
        # Summary statistics
        print("=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print()
        
        successful_results = [r for r in results if r["base_model"]["generation_time"] > 0 and r["finetuned_model"]["generation_time"] > 0]
        
        if successful_results:
            base_avg_time = sum(r["base_model"]["generation_time"] for r in successful_results) / len(successful_results)
            finetuned_avg_time = sum(r["finetuned_model"]["generation_time"] for r in successful_results) / len(successful_results)
            base_avg_length = sum(r["base_model"]["response_length"] for r in successful_results) / len(successful_results)
            finetuned_avg_length = sum(r["finetuned_model"]["response_length"] for r in successful_results) / len(successful_results)
            base_avg_words = sum(r["base_model"]["word_count"] for r in successful_results) / len(successful_results)
            finetuned_avg_words = sum(r["finetuned_model"]["word_count"] for r in successful_results) / len(successful_results)
            
            print("BASE MODEL:")
            print(f"  Average generation time: {base_avg_time:.2f}s")
            print(f"  Average response length: {base_avg_length:.0f} characters")
            print(f"  Average word count: {base_avg_words:.0f} words")
            print()
            print("FINE-TUNED MODEL:")
            print(f"  Average generation time: {finetuned_avg_time:.2f}s")
            print(f"  Average response length: {finetuned_avg_length:.0f} characters")
            print(f"  Average word count: {finetuned_avg_words:.0f} words")
            print()
            
            # Performance comparison
            time_diff = ((finetuned_avg_time - base_avg_time) / base_avg_time) * 100
            length_diff = ((finetuned_avg_length - base_avg_length) / base_avg_length) * 100 if base_avg_length > 0 else 0
            
            print("COMPARISON:")
            if abs(time_diff) < 5:
                print(f"  Generation time: Similar ({time_diff:+.1f}%)")
            elif time_diff > 0:
                print(f"  Generation time: Fine-tuned is {time_diff:.1f}% slower")
            else:
                print(f"  Generation time: Fine-tuned is {abs(time_diff):.1f}% faster")
            
            if abs(length_diff) < 10:
                print(f"  Response length: Similar ({length_diff:+.1f}%)")
            elif length_diff > 0:
                print(f"  Response length: Fine-tuned is {length_diff:.1f}% longer")
            else:
                print(f"  Response length: Fine-tuned is {abs(length_diff):.1f}% shorter")
            print()
        
        print("=" * 80)
        print("Comparison complete!")
        print("=" * 80)
        print()
        print("Review the detailed responses in the output above and")
        print(f"check {output_file} for full results in JSON format.")
        
    except KeyboardInterrupt:
        print("\n\nComparison interrupted by user.")
    except Exception as e:
        print(f"\n\nError during comparison: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    compare_models()
