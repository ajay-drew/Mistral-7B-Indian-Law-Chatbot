"""Quick test script to verify model loading works with the fixes."""
import asyncio
import sys
import os
sys.path.insert(0, '.')

from backend.main import ModelBundle, get_config

async def test_model_loading():
    """Test model loading with the new configuration."""
    print("=" * 80)
    print("Testing Model Loading After Paging File Configuration")
    print("=" * 80)
    
    try:
        config = get_config()
        bundle = ModelBundle(config)
        
        print("\nAttempting to load model...")
        print("This may take several minutes...")
        
        await bundle.ensure_loaded()
        
        print("\n" + "=" * 80)
        print("SUCCESS: Model loaded successfully!")
        print("=" * 80)
        
        # Test a simple generation
        print("\nTesting generation...")
        test_prompt = "What is Indian law?"
        response = await bundle.generate(
            prompt=test_prompt,
            max_new_tokens=50,
            temperature=0.7,
            top_p=0.9
        )
        print(f"\nTest prompt: {test_prompt}")
        print(f"Response: {response[:200]}...")
        
        print("\n" + "=" * 80)
        print("All tests passed!")
        print("=" * 80)
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"ERROR: {type(e).__name__}: {str(e)}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_model_loading())
