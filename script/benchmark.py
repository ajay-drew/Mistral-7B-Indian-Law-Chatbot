"""Performance benchmark script for the Mistral Indian Law Chat API.

Measures latency, throughput, and tokens/second for various query types.
"""

import asyncio
import json
import statistics
import time
from typing import List, Dict, Any

import httpx


API_BASE_URL = "http://localhost:2347"
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"
API_KEY = None  # Set if required


async def benchmark_request(
    client: httpx.AsyncClient,
    prompt: str,
    iterations: int = 10
) -> List[Dict[str, Any]]:
    """Run benchmark requests for a given prompt."""
    results = []
    
    for i in range(iterations):
        start_time = time.time()
        
        try:
            headers = {}
            if API_KEY:
                headers["X-API-Key"] = API_KEY
            
            response = await client.post(
                CHAT_ENDPOINT,
                json={
                    "prompt": prompt,
                    "max_new_tokens": 256,
                    "temperature": 0.7,
                    "top_p": 0.9
                },
                headers=headers,
                timeout=60.0
            )
            
            if response.status_code != 200:
                print(f"  Request {i+1} failed: {response.status_code}")
                continue
            
            data = response.json()
            duration = time.time() - start_time
            
            reply = data.get("reply", "")
            token_count = len(reply.split())
            tokens_per_second = token_count / duration if duration > 0 else 0
            
            results.append({
                "duration": duration,
                "tokens": token_count,
                "tokens_per_second": tokens_per_second,
                "success": True
            })
            
            print(f"  Request {i+1}/{iterations}: {duration:.2f}s, {token_count} tokens, {tokens_per_second:.1f} tokens/s")
            
        except Exception as e:
            print(f"  Request {i+1} error: {e}")
            results.append({
                "duration": None,
                "tokens": 0,
                "tokens_per_second": 0,
                "success": False,
                "error": str(e)
            })
        
        # Small delay between requests
        await asyncio.sleep(0.5)
    
    return results


async def run_benchmarks():
    """Run comprehensive benchmarks."""
    print("=" * 70)
    print("Mistral Indian Law Chat API - Performance Benchmarks")
    print("=" * 70)
    print()
    
    # Test queries of varying complexity
    test_queries = [
        {
            "name": "Simple Query",
            "prompt": "What is Article 21 of the Indian Constitution?",
            "expected_tokens": 50
        },
        {
            "name": "Medium Query",
            "prompt": "Explain the doctrine of basic structure in Indian constitutional law.",
            "expected_tokens": 150
        },
        {
            "name": "Complex Query",
            "prompt": "What are the key provisions of the Indian Penal Code regarding murder and what are the different types of murder recognized under Section 300?",
            "expected_tokens": 250
        }
    ]
    
    all_results = []
    
    async with httpx.AsyncClient() as client:
        # Check health first
        try:
            health_response = await client.get(f"{API_BASE_URL}/health", timeout=5.0)
            if health_response.status_code == 200:
                health_data = health_response.json()
                print(f"Backend Status: {health_data.get('status', 'unknown')}")
                print(f"Device: {health_data.get('device', 'unknown')}")
                if 'cuda_device' in health_data:
                    print(f"GPU: {health_data['cuda_device']}")
                if 'cuda_memory_gb' in health_data:
                    print(f"GPU Memory: {health_data['cuda_memory_gb']} GB")
                print()
            else:
                print("Warning: Health check failed")
                print()
        except Exception as e:
            print(f"Warning: Could not check health: {e}")
            print()
        
        # Run benchmarks for each query type
        for query_info in test_queries:
            print(f"Benchmarking: {query_info['name']}")
            print(f"Query: {query_info['prompt'][:60]}...")
            print()
            
            results = await benchmark_request(client, query_info['prompt'], iterations=10)
            
            if not results:
                print("  No successful requests")
                print()
                continue
            
            successful_results = [r for r in results if r.get('success', False)]
            
            if not successful_results:
                print("  All requests failed")
                print()
                continue
            
            # Calculate statistics
            durations = [r['duration'] for r in successful_results]
            token_counts = [r['tokens'] for r in successful_results]
            tokens_per_second = [r['tokens_per_second'] for r in successful_results]
            
            stats = {
                "query_name": query_info['name'],
                "query": query_info['prompt'],
                "iterations": len(results),
                "successful": len(successful_results),
                "avg_duration": statistics.mean(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "p50_duration": statistics.median(durations),
                "p95_duration": statistics.quantiles(durations, n=20)[18] if len(durations) >= 20 else max(durations),
                "p99_duration": max(durations),
                "avg_tokens": statistics.mean(token_counts),
                "avg_tokens_per_second": statistics.mean(tokens_per_second),
                "min_tokens_per_second": min(tokens_per_second),
                "max_tokens_per_second": max(tokens_per_second),
            }
            
            all_results.append(stats)
            
            # Print summary
            print(f"  Summary:")
            print(f"    Successful: {stats['successful']}/{stats['iterations']}")
            print(f"    Avg Latency: {stats['avg_duration']:.2f}s")
            print(f"    P50 Latency: {stats['p50_duration']:.2f}s")
            print(f"    P95 Latency: {stats['p95_duration']:.2f}s")
            print(f"    Avg Tokens: {stats['avg_tokens']:.0f}")
            print(f"    Avg Tokens/s: {stats['avg_tokens_per_second']:.1f}")
            print()
    
    # Print overall summary
    if all_results:
        print("=" * 70)
        print("Overall Summary")
        print("=" * 70)
        print()
        
        overall_avg_latency = statistics.mean([r['avg_duration'] for r in all_results])
        overall_avg_tokens_per_second = statistics.mean([r['avg_tokens_per_second'] for r in all_results])
        
        print(f"Average Latency (across all queries): {overall_avg_latency:.2f}s")
        print(f"Average Throughput: {overall_avg_tokens_per_second:.1f} tokens/second")
        print()
        
        # Save results to file
        output_file = "benchmark_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "overall": {
                    "avg_latency": overall_avg_latency,
                    "avg_tokens_per_second": overall_avg_tokens_per_second
                },
                "queries": all_results
            }, f, indent=2)
        
        print(f"Detailed results saved to: {output_file}")
        print()
    
    print("Benchmark complete!")


if __name__ == "__main__":
    asyncio.run(run_benchmarks())
