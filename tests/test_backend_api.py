"""Tests for FastAPI backend endpoints."""

import pytest
from fastapi import status
from unittest.mock import patch, MagicMock, AsyncMock


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_endpoint_returns_ok(self, test_client):
        """Test health endpoint returns 200 OK."""
        response = test_client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data
        assert "device" in data
        assert data["device"] in ["cuda", "cpu"]
    
    def test_health_endpoint_structure(self, test_client):
        """Test health endpoint response structure."""
        response = test_client.get("/health")
        data = response.json()
        assert isinstance(data["status"], str)
        assert isinstance(data["device"], str)
        assert "model_loaded" in data


class TestChatEndpoint:
    """Tests for /chat endpoint."""
    
    def test_chat_with_prompt(self, test_client, sample_prompt):
        """Test chat endpoint with simple prompt."""
        response = test_client.post(
            "/chat",
            json={"prompt": sample_prompt}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "reply" in data
        assert isinstance(data["reply"], str)
    
    def test_chat_with_messages(self, test_client, sample_messages):
        """Test chat endpoint with messages array."""
        response = test_client.post(
            "/chat",
            json={"messages": sample_messages}
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "reply" in data
    
    def test_chat_with_custom_parameters(self, test_client):
        """Test chat endpoint with custom generation parameters."""
        response = test_client.post(
            "/chat",
            json={
                "prompt": "Test prompt",
                "max_new_tokens": 100,
                "temperature": 0.5,
                "top_p": 0.8
            }
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "reply" in data
    
    def test_chat_without_prompt_or_messages(self, test_client):
        """Test chat endpoint fails without prompt or messages."""
        response = test_client.post("/chat", json={})
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_chat_with_empty_messages(self, test_client):
        """Test chat endpoint fails with empty messages array."""
        response = test_client.post("/chat", json={"messages": []})
        assert response.status_code == status.HTTP_400_BAD_REQUEST
    
    def test_chat_invalid_role(self, test_client):
        """Test chat endpoint validates message roles."""
        response = test_client.post(
            "/chat",
            json={
                "messages": [{"role": "invalid_role", "content": "test"}]
            }
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_chat_max_tokens_validation(self, test_client):
        """Test chat endpoint validates max_new_tokens range."""
        response = test_client.post(
            "/chat",
            json={"prompt": "test", "max_new_tokens": 0}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        response = test_client.post(
            "/chat",
            json={"prompt": "test", "max_new_tokens": 2000}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_chat_temperature_validation(self, test_client):
        """Test chat endpoint validates temperature range."""
        response = test_client.post(
            "/chat",
            json={"prompt": "test", "temperature": -1}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        response = test_client.post(
            "/chat",
            json={"prompt": "test", "temperature": 3}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_chat_top_p_validation(self, test_client):
        """Test chat endpoint validates top_p range."""
        response = test_client.post(
            "/chat",
            json={"prompt": "test", "top_p": 0.05}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        response = test_client.post(
            "/chat",
            json={"prompt": "test", "top_p": 1.5}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestCORS:
    """Tests for CORS configuration."""
    
    def test_cors_headers_present(self, test_client):
        """Test CORS headers are present in responses."""
        response = test_client.options("/chat")
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_405_METHOD_NOT_ALLOWED]


class TestErrorHandling:
    """Tests for error handling."""
    
    def test_chat_handles_generation_error(self, test_client):
        """Test chat endpoint handles generation errors gracefully."""
        from backend.main import bundle
        original_generate = bundle.generate
        
        async def error_generate(prompt, max_new_tokens, temperature, top_p):
            raise RuntimeError("Generation failed")
        
        bundle.generate = error_generate
        
        try:
            response = test_client.post(
                "/chat",
                json={"prompt": "test"}
            )
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            data = response.json()
            assert "detail" in data
        finally:
            bundle.generate = original_generate


class TestRequestValidation:
    """Tests for request validation."""
    
    def test_invalid_json(self, test_client):
        """Test endpoint handles invalid JSON."""
        response = test_client.post(
            "/chat",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_missing_content_type(self, test_client):
        """Test endpoint requires Content-Type header."""
        response = test_client.post(
            "/chat",
            json={"prompt": "test"},
            headers={}
        )
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_422_UNPROCESSABLE_ENTITY]

