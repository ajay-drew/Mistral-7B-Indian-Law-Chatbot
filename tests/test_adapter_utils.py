"""Tests for adapter utility functions."""

import json
import pytest
import tempfile
import shutil
from pathlib import Path

from backend.adapter_utils import (
    validate_adapter_path,
    check_adapter_files,
    load_adapter_config,
    validate_adapter_config,
    get_adapter_metadata,
    validate_adapter,
    get_adapter_info,
    get_required_adapter_files,
    get_optional_adapter_files,
    AdapterValidationError,
    AdapterNotFoundError,
)


class TestAdapterPathValidation:
    """Tests for adapter path validation."""
    
    def test_validate_adapter_path_exists(self, tmp_path):
        """Test validation passes for existing directory."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        
        result = validate_adapter_path(adapter_dir)
        assert result is True
    
    def test_validate_adapter_path_not_exists(self, tmp_path):
        """Test validation fails for non-existent path."""
        adapter_dir = tmp_path / "nonexistent"
        
        with pytest.raises(AdapterNotFoundError):
            validate_adapter_path(adapter_dir)
    
    def test_validate_adapter_path_not_directory(self, tmp_path):
        """Test validation fails for file instead of directory."""
        adapter_file = tmp_path / "adapter.txt"
        adapter_file.write_text("test")
        
        with pytest.raises(AdapterNotFoundError):
            validate_adapter_path(adapter_file)


class TestAdapterFiles:
    """Tests for adapter file checking."""
    
    def test_get_required_adapter_files(self):
        """Test required files list."""
        required = get_required_adapter_files()
        assert "adapter_config.json" in required
        assert "adapter_model.safetensors" in required
    
    def test_get_optional_adapter_files(self):
        """Test optional files list."""
        optional = get_optional_adapter_files()
        assert "tokenizer_config.json" in optional
        assert "tokenizer.json" in optional
    
    def test_check_adapter_files_all_exist(self, tmp_path):
        """Test file check when all files exist."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        
        # Create required files
        (adapter_dir / "adapter_config.json").write_text("{}")
        (adapter_dir / "adapter_model.safetensors").write_text("dummy")
        
        all_exist, missing, existing = check_adapter_files(adapter_dir)
        assert all_exist is True
        assert len(missing) == 0
        assert len(existing) >= 2
    
    def test_check_adapter_files_missing_required(self, tmp_path):
        """Test file check when required files are missing."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        
        # Create only one required file
        (adapter_dir / "adapter_config.json").write_text("{}")
        
        all_exist, missing, existing = check_adapter_files(adapter_dir)
        assert all_exist is False
        assert "adapter_model.safetensors" in missing


class TestAdapterConfig:
    """Tests for adapter configuration loading and validation."""
    
    def test_load_adapter_config_valid(self, tmp_path):
        """Test loading valid adapter config."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        
        config_data = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "mistralai/Mistral-7B-v0.1",
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"]
        }
        
        config_path = adapter_dir / "adapter_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f)
        
        config = load_adapter_config(adapter_dir)
        assert config["peft_type"] == "LORA"
        assert config["r"] == 16
    
    def test_load_adapter_config_not_found(self, tmp_path):
        """Test loading config when file doesn't exist."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        
        with pytest.raises(AdapterNotFoundError):
            load_adapter_config(adapter_dir)
    
    def test_load_adapter_config_invalid_json(self, tmp_path):
        """Test loading invalid JSON config."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        
        config_path = adapter_dir / "adapter_config.json"
        config_path.write_text("invalid json {")
        
        with pytest.raises(AdapterValidationError):
            load_adapter_config(adapter_dir)
    
    def test_validate_adapter_config_valid_lora(self):
        """Test validation of valid LoRA config."""
        config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "mistralai/Mistral-7B-v0.1",
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"]
        }
        
        is_valid, errors = validate_adapter_config(config)
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_adapter_config_missing_required_fields(self):
        """Test validation fails for missing required fields."""
        config = {
            "peft_type": "LORA"
        }
        
        is_valid, errors = validate_adapter_config(config)
        assert is_valid is False
        assert any("task_type" in error for error in errors)
        assert any("base_model_name_or_path" in error for error in errors)
    
    def test_validate_adapter_config_invalid_peft_type(self):
        """Test validation fails for invalid PEFT type."""
        config = {
            "peft_type": "INVALID",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "test"
        }
        
        is_valid, errors = validate_adapter_config(config)
        assert is_valid is False
        assert any("Invalid peft_type" in error for error in errors)
    
    def test_validate_adapter_config_invalid_lora_r(self):
        """Test validation fails for invalid LoRA r value."""
        config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "test",
            "r": -1,
            "lora_alpha": 32,
            "target_modules": ["q_proj"]
        }
        
        is_valid, errors = validate_adapter_config(config)
        assert is_valid is False
        assert any("r must be a positive integer" in error for error in errors)


class TestAdapterMetadata:
    """Tests for adapter metadata functions."""
    
    def test_get_adapter_metadata_complete(self, tmp_path):
        """Test getting metadata for complete adapter."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        
        # Create required files
        config_data = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "mistralai/Mistral-7B-v0.1",
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"]
        }
        
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config_data))
        (adapter_dir / "adapter_model.safetensors").write_text("dummy")
        
        metadata = get_adapter_metadata(adapter_dir)
        
        assert metadata["exists"] is True
        assert metadata["files"]["all_required_exist"] is True
        assert metadata["config"]["is_valid"] is True
        assert metadata["config"]["peft_type"] == "LORA"
    
    def test_get_adapter_metadata_missing_files(self, tmp_path):
        """Test getting metadata when files are missing."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        
        metadata = get_adapter_metadata(adapter_dir)
        
        assert metadata["exists"] is True
        assert metadata["files"]["all_required_exist"] is False
        assert len(metadata["files"]["missing_files"]) > 0
    
    def test_get_adapter_metadata_nonexistent(self, tmp_path):
        """Test getting metadata for non-existent adapter."""
        adapter_dir = tmp_path / "nonexistent"
        
        metadata = get_adapter_metadata(adapter_dir)
        
        assert metadata["exists"] is False


class TestAdapterValidation:
    """Tests for comprehensive adapter validation."""
    
    def test_validate_adapter_valid(self, tmp_path):
        """Test validation passes for valid adapter."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        
        config_data = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "mistralai/Mistral-7B-v0.1",
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"]
        }
        
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config_data))
        (adapter_dir / "adapter_model.safetensors").write_text("dummy")
        
        is_valid, errors = validate_adapter(adapter_dir, raise_on_error=False)
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_adapter_missing_files(self, tmp_path):
        """Test validation fails when files are missing."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        
        is_valid, errors = validate_adapter(adapter_dir, raise_on_error=False)
        assert is_valid is False
        assert len(errors) > 0
    
    def test_validate_adapter_raises_on_error(self, tmp_path):
        """Test validation raises exception when raise_on_error=True."""
        adapter_dir = tmp_path / "nonexistent"
        
        with pytest.raises(AdapterNotFoundError):
            validate_adapter(adapter_dir, raise_on_error=True)


class TestAdapterInfo:
    """Tests for adapter info function."""
    
    def test_get_adapter_info_valid(self, tmp_path):
        """Test getting info for valid adapter."""
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        
        config_data = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "mistralai/Mistral-7B-v0.1",
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"]
        }
        
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config_data))
        (adapter_dir / "adapter_model.safetensors").write_text("dummy")
        
        info = get_adapter_info(adapter_dir)
        
        assert info["status"] == "valid"
        assert info["peft_type"] == "LORA"
        assert info["base_model"] == "mistralai/Mistral-7B-v0.1"
        assert "lora_rank" in info

