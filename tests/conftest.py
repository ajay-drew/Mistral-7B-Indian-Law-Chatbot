"""Pytest configuration and shared fixtures."""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import torch

try:
    from fastapi.testclient import TestClient
except ImportError:
    from starlette.testclient import TestClient

os.environ.setdefault("BASE_MODEL_NAME", "mistralai/Mistral-7B-v0.1")
os.environ.setdefault("ADAPTER_PATH", "./mistral-indian-law-final")
os.environ.setdefault("DEVICE_MAP", "cpu")


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.eval.return_value = None
    model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    model.parameters.return_value = iter([torch.tensor([1.0])])
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    tokenizer.decode.return_value = "Test response"
    tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    tokenizer.eos_token_id = 2
    
    def tokenize(text, **kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3]])}
    
    tokenizer.side_effect = tokenize
    return tokenizer


@pytest.fixture
def sample_prompt():
    """Sample prompt for testing."""
    return "What is the doctrine of basic structure?"


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        {"role": "user", "content": "What is Section 439 of CrPC?"},
        {"role": "assistant", "content": "Section 439 deals with..."},
        {"role": "user", "content": "Tell me more"}
    ]


@pytest.fixture(scope="module")
def test_client():
    """Create a test client for FastAPI."""
    from backend.main import app, bundle
    import httpx
    from httpx._transports.asgi import ASGITransport
    
    async def mock_generate(prompt, max_new_tokens, temperature, top_p):
        return "Mock generated response for: " + prompt
    
    bundle._model = MagicMock()
    bundle._tokenizer = MagicMock()
    bundle.generate = AsyncMock(side_effect=mock_generate)
    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    bundle._model.parameters.return_value = iter([mock_param])
    bundle.tokenizer.decode.return_value = "Mock response"
    bundle.tokenizer.eos_token_id = 2
    
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass
    
    class CompatibleTestClient:
        """Test client compatible with current httpx/starlette versions."""
        def __init__(self, app, base_url="http://testserver"):
            self.app = app
            self.base_url = base_url
            self._async_client = None
        
        def _get_client(self):
            """Get or create async client."""
            if self._async_client is None:
                transport = ASGITransport(app=self.app)
                self._async_client = httpx.AsyncClient(
                    transport=transport,
                    base_url=self.base_url
                )
            return self._async_client
        
        def _run_async(self, coro):
            """Run async coroutine, handling event loop properly."""
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    return asyncio.run(coro)
                except (ImportError, RuntimeError):
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, coro)
                        return future.result()
            except RuntimeError:
                return asyncio.run(coro)
        
        def get(self, url, **kwargs):
            """Synchronous GET request."""
            client = self._get_client()
            response = self._run_async(client.get(url, **kwargs))
            return self._make_response(response)
        
        def post(self, url, **kwargs):
            """Synchronous POST request."""
            client = self._get_client()
            response = self._run_async(client.post(url, **kwargs))
            return self._make_response(response)
        
        def options(self, url, **kwargs):
            """Synchronous OPTIONS request."""
            client = self._get_client()
            response = self._run_async(client.options(url, **kwargs))
            return self._make_response(response)
        
        def _make_response(self, async_response):
            """Convert async response to sync-like response object."""
            class SyncResponse:
                def __init__(self, async_resp):
                    self._async_resp = async_resp
                    self.status_code = async_resp.status_code
                    self.headers = async_resp.headers
                    self._json = None
                    self._text = None
                    self._content = None
                
                def json(self):
                    if self._json is None:
                        content = self.content
                        import json
                        self._json = json.loads(content)
                    return self._json
                
                def text(self):
                    if self._text is None:
                        self._text = self.content.decode()
                    return self._text
                
                @property
                def content(self):
                    if self._content is None:
                        import asyncio
                        try:
                            loop = asyncio.get_running_loop()
                            try:
                                import nest_asyncio
                                nest_asyncio.apply()
                                self._content = asyncio.run(self._async_resp.aread())
                            except (ImportError, RuntimeError):
                                import concurrent.futures
                                with concurrent.futures.ThreadPoolExecutor() as executor:
                                    future = executor.submit(asyncio.run, self._async_resp.aread())
                                    self._content = future.result()
                        except RuntimeError:
                            self._content = asyncio.run(self._async_resp.aread())
                    return self._content
            
            return SyncResponse(async_response)
    
    client = CompatibleTestClient(app)
    
    yield client
    
    if client._async_client:
        try:
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                    asyncio.run(client._async_client.aclose())
                except (ImportError, RuntimeError):
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        executor.submit(asyncio.run, client._async_client.aclose())
            except RuntimeError:
                asyncio.run(client._async_client.aclose())
        except Exception:
            pass


@pytest.fixture
def skip_if_no_model():
    """Skip test if model files don't exist."""
    adapter_path = os.getenv("ADAPTER_PATH", "./mistral-indian-law-final")
    if not os.path.exists(adapter_path):
        pytest.skip(f"Model adapter not found at {adapter_path}")


@pytest.fixture
def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables before each test."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)
