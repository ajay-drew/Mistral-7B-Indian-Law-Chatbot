"""Tests for model loading and generation functionality."""

import pytest
import torch
import os
from unittest.mock import patch, MagicMock, AsyncMock
from backend.main import ModelBundle, ServiceConfig, get_config


class TestServiceConfig:
    """Tests for ServiceConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ServiceConfig()
        assert config.base_model_name == "mistralai/Mistral-7B-v0.1"
        assert config.adapter_path == "./mistral-indian-law-final"
        assert config.device_map == "auto"
        assert config.max_new_tokens == 256
        assert config.temperature == 0.7
        assert config.top_p == 0.9
    
    def test_config_from_environment(self):
        """Test configuration loads from environment variables."""
        os.environ["BASE_MODEL_NAME"] = "test-model"
        os.environ["MAX_NEW_TOKENS"] = "512"
        os.environ["TEMPERATURE"] = "0.9"
        
        get_config.cache_clear()
        config = get_config()
        
        assert config.base_model_name == "test-model"
        assert config.max_new_tokens == 512
        assert config.temperature == 0.9
        
        del os.environ["BASE_MODEL_NAME"]
        del os.environ["MAX_NEW_TOKENS"]
        del os.environ["TEMPERATURE"]
        get_config.cache_clear()
    
    def test_quant_config_property(self):
        """Test quantization config is properly created."""
        config = ServiceConfig()
        quant_config = config.quant_config
        
        assert quant_config.load_in_4bit is True
        assert quant_config.bnb_4bit_compute_dtype == torch.float16
        assert quant_config.bnb_4bit_use_double_quant is True
        assert quant_config.bnb_4bit_quant_type == "nf4"


class TestModelBundle:
    """Tests for ModelBundle class."""
    
    @pytest.mark.asyncio
    async def test_model_bundle_initialization(self):
        """Test ModelBundle initializes correctly."""
        config = ServiceConfig()
        bundle = ModelBundle(config)
        
        assert bundle.config == config
        assert bundle._model is None
        assert bundle._tokenizer is None
    
    @pytest.mark.asyncio
    async def test_ensure_loaded_creates_model(self, mock_model, mock_tokenizer):
        """Test ensure_loaded loads model and tokenizer."""
        config = ServiceConfig()
        bundle = ModelBundle(config)
        
        with patch("backend.main.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("backend.main.AutoModelForCausalLM.from_pretrained", return_value=mock_model), \
             patch("backend.main.PeftModel.from_pretrained", return_value=mock_model):
            
            await bundle.ensure_loaded()
            
            assert bundle._model is not None
            assert bundle._tokenizer is not None
    
    @pytest.mark.asyncio
    async def test_ensure_loaded_idempotent(self, mock_model, mock_tokenizer):
        """Test ensure_loaded can be called multiple times safely."""
        config = ServiceConfig()
        bundle = ModelBundle(config)
        
        with patch("backend.main.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("backend.main.AutoModelForCausalLM.from_pretrained", return_value=mock_model), \
             patch("backend.main.PeftModel.from_pretrained", return_value=mock_model):
            
            await bundle.ensure_loaded()
            first_model = bundle._model
            
            await bundle.ensure_loaded()
            second_model = bundle._model
            
            assert first_model is second_model
    
    def test_model_property_raises_when_not_loaded(self):
        """Test model property raises error when not loaded."""
        config = ServiceConfig()
        bundle = ModelBundle(config)
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            _ = bundle.model
    
    def test_tokenizer_property_raises_when_not_loaded(self):
        """Test tokenizer property raises error when not loaded."""
        config = ServiceConfig()
        bundle = ModelBundle(config)
        
        with pytest.raises(RuntimeError, match="Tokenizer not loaded"):
            _ = bundle.tokenizer
    
    @pytest.mark.asyncio
    async def test_generate_extracts_response_correctly(self, mock_model, mock_tokenizer):
        """Test generate method extracts only new tokens."""
        config = ServiceConfig()
        bundle = ModelBundle(config)
        bundle._model = mock_model
        bundle._tokenizer = mock_tokenizer
        
        def tokenize_side_effect(text, return_tensors, **kwargs):
            return {"input_ids": torch.tensor([[1, 2, 3]])}
        
        mock_tokenizer.side_effect = tokenize_side_effect
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        mock_model.generate.return_value = input_ids
        
        def mock_tokenizer_call(text, return_tensors=None, **kwargs):
            return {"input_ids": torch.tensor([[1, 2, 3]])}
        
        mock_tokenizer.side_effect = mock_tokenizer_call
        mock_tokenizer.decode.return_value = "Generated response"
        
        mock_param = MagicMock()
        mock_param.device = torch.device("cpu")
        mock_model.parameters.return_value = iter([mock_param])
        
        with patch("torch.inference_mode"):
            result = await bundle.generate("test", 10, 0.7, 0.9)
            
            assert mock_tokenizer.decode.called
            assert isinstance(result, str)


class TestPromptBuilding:
    """Tests for prompt building logic."""
    
    def test_build_prompt_with_prompt_field(self):
        """Test build_prompt uses prompt field when provided and includes system prompt."""
        from backend.main import build_prompt, ChatRequest, SYSTEM_PROMPT
        
        request = ChatRequest(prompt="Direct prompt")
        result = build_prompt(request)
        assert SYSTEM_PROMPT in result
        assert "Question: Direct prompt" in result
        assert "Answer:" in result
    
    def test_build_prompt_with_messages(self):
        """Test build_prompt builds from messages array."""
        from backend.main import build_prompt, ChatRequest, Message
        
        messages = [
            Message(role="user", content="Question 1"),
            Message(role="assistant", content="Answer 1"),
            Message(role="user", content="Question 2")
        ]
        request = ChatRequest(messages=messages)
        result = build_prompt(request)
        
        assert "Question: Question 1" in result
        assert "Answer: Answer 1" in result
        assert "Question: Question 2" in result
        assert "Answer:" in result
    
    def test_build_prompt_empty_raises_error(self):
        """Test build_prompt raises error with no prompt or messages."""
        from backend.main import build_prompt, ChatRequest
        from fastapi import HTTPException
        
        request = ChatRequest()
        with pytest.raises(HTTPException):
            build_prompt(request)

