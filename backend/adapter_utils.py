"""Adapter utility functions for PEFT adapter management and validation.

This module provides utilities for:
- Validating adapter files and configurations
- Loading adapter metadata
- Checking adapter health and compatibility
- Adapter file verification
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from peft import PeftConfig

logger = logging.getLogger(__name__)


class AdapterValidationError(Exception):
    """Raised when adapter validation fails."""
    pass


class AdapterNotFoundError(Exception):
    """Raised when adapter files are not found."""
    pass


def validate_adapter_path(adapter_path: str | Path) -> bool:
    """Validate that adapter path exists and is a directory.
    
    Args:
        adapter_path: Path to the adapter directory
        
    Returns:
        True if path exists and is a directory
        
    Raises:
        AdapterNotFoundError: If adapter path doesn't exist
    """
    adapter_path = Path(adapter_path)
    
    if not adapter_path.exists():
        raise AdapterNotFoundError(
            f"Adapter path does not exist: {adapter_path}"
        )
    
    if not adapter_path.is_dir():
        raise AdapterNotFoundError(
            f"Adapter path is not a directory: {adapter_path}"
        )
    
    return True


def get_required_adapter_files() -> List[str]:
    """Get list of required adapter files.
    
    Returns:
        List of required file names
    """
    return [
        "adapter_config.json",
        "adapter_model.safetensors",
    ]


def get_optional_adapter_files() -> List[str]:
    """Get list of optional adapter files.
    
    Returns:
        List of optional file names
    """
    return [
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
    ]


def check_adapter_files(adapter_path: str | Path) -> Tuple[bool, List[str], List[str]]:
    """Check which adapter files exist.
    
    Args:
        adapter_path: Path to the adapter directory
        
    Returns:
        Tuple of (all_required_exist, missing_files, existing_files)
    """
    adapter_path = Path(adapter_path)
    required_files = get_required_adapter_files()
    optional_files = get_optional_adapter_files()
    all_files = required_files + optional_files
    
    missing_files = []
    existing_files = []
    
    for filename in all_files:
        file_path = adapter_path / filename
        if file_path.exists():
            existing_files.append(filename)
        else:
            if filename in required_files:
                missing_files.append(filename)
    
    all_required_exist = len([f for f in required_files if f in existing_files]) == len(required_files)
    
    return all_required_exist, missing_files, existing_files


def load_adapter_config(adapter_path: str | Path) -> Dict:
    """Load adapter configuration from adapter_config.json.
    
    Args:
        adapter_path: Path to the adapter directory
        
    Returns:
        Dictionary containing adapter configuration
        
    Raises:
        AdapterNotFoundError: If adapter_config.json doesn't exist
        AdapterValidationError: If config file is invalid
    """
    adapter_path = Path(adapter_path)
    config_path = adapter_path / "adapter_config.json"
    
    if not config_path.exists():
        raise AdapterNotFoundError(
            f"Adapter config file not found: {config_path}"
        )
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        raise AdapterValidationError(
            f"Invalid JSON in adapter config: {e}"
        ) from e
    except Exception as e:
        raise AdapterValidationError(
            f"Error loading adapter config: {e}"
        ) from e


def validate_adapter_config(config: Dict) -> Tuple[bool, List[str]]:
    """Validate adapter configuration structure and values.
    
    Args:
        config: Adapter configuration dictionary
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check required fields
    required_fields = ["peft_type", "task_type", "base_model_name_or_path"]
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Validate PEFT type
    if "peft_type" in config:
        valid_peft_types = ["LORA", "ADALORA", "PROMPT_TUNING", "P_TUNING", "PREFIX_TUNING", "IA3"]
        if config["peft_type"] not in valid_peft_types:
            errors.append(f"Invalid peft_type: {config['peft_type']}. Must be one of {valid_peft_types}")
    
    # Validate task type
    if "task_type" in config:
        valid_task_types = ["CAUSAL_LM", "SEQ_2_SEQ_LM", "QUESTION_ANS", "FEATURE_EXTRACTION", "TOKEN_CLS", "SEQ_CLS"]
        if config["task_type"] not in valid_task_types:
            errors.append(f"Invalid task_type: {config['task_type']}. Must be one of {valid_task_types}")
    
    # Validate LoRA-specific fields if applicable
    if config.get("peft_type") == "LORA":
        if "r" not in config:
            errors.append("Missing required LoRA field: r")
        elif not isinstance(config["r"], int) or config["r"] <= 0:
            errors.append("LoRA r must be a positive integer")
        
        if "lora_alpha" not in config:
            errors.append("Missing required LoRA field: lora_alpha")
        elif not isinstance(config["lora_alpha"], (int, float)) or config["lora_alpha"] <= 0:
            errors.append("LoRA lora_alpha must be a positive number")
        
        if "target_modules" not in config:
            errors.append("Missing required LoRA field: target_modules")
        elif not isinstance(config["target_modules"], list) or len(config["target_modules"]) == 0:
            errors.append("LoRA target_modules must be a non-empty list")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def get_adapter_metadata(adapter_path: str | Path) -> Dict:
    """Get comprehensive metadata about an adapter.
    
    Args:
        adapter_path: Path to the adapter directory
        
    Returns:
        Dictionary containing adapter metadata
    """
    adapter_path = Path(adapter_path)
    metadata = {
        "adapter_path": str(adapter_path),
        "exists": adapter_path.exists(),
        "is_directory": adapter_path.is_dir() if adapter_path.exists() else False,
    }
    
    if not adapter_path.exists():
        return metadata
    
    # Check files
    all_required_exist, missing_files, existing_files = check_adapter_files(adapter_path)
    metadata["files"] = {
        "all_required_exist": all_required_exist,
        "missing_files": missing_files,
        "existing_files": existing_files,
    }
    
    # Load and validate config
    try:
        config = load_adapter_config(adapter_path)
        is_valid, errors = validate_adapter_config(config)
        metadata["config"] = {
            "exists": True,
            "is_valid": is_valid,
            "errors": errors,
            "peft_type": config.get("peft_type"),
            "task_type": config.get("task_type"),
            "base_model": config.get("base_model_name_or_path"),
        }
        
        # Add LoRA-specific metadata
        if config.get("peft_type") == "LORA":
            metadata["config"]["lora"] = {
                "r": config.get("r"),
                "lora_alpha": config.get("lora_alpha"),
                "lora_dropout": config.get("lora_dropout"),
                "target_modules": config.get("target_modules"),
            }
    except Exception as e:
        metadata["config"] = {
            "exists": False,
            "error": str(e),
        }
    
    # Get file sizes
    file_sizes = {}
    for filename in existing_files:
        file_path = adapter_path / filename
        if file_path.exists():
            file_sizes[filename] = file_path.stat().st_size
    metadata["file_sizes"] = file_sizes
    
    return metadata


def validate_adapter(adapter_path: str | Path, raise_on_error: bool = True) -> Tuple[bool, List[str]]:
    """Comprehensive adapter validation.
    
    Args:
        adapter_path: Path to the adapter directory
        raise_on_error: If True, raise exceptions on validation errors
        
    Returns:
        Tuple of (is_valid, error_messages)
        
    Raises:
        AdapterNotFoundError: If adapter path doesn't exist (when raise_on_error=True)
        AdapterValidationError: If adapter validation fails (when raise_on_error=True)
    """
    errors = []
    
    try:
        # Validate path
        validate_adapter_path(adapter_path)
    except AdapterNotFoundError as e:
        errors.append(str(e))
        if raise_on_error:
            raise
        return False, errors
    
    # Check required files
    all_required_exist, missing_files, existing_files = check_adapter_files(adapter_path)
    if not all_required_exist:
        errors.append(f"Missing required adapter files: {', '.join(missing_files)}")
    
    # Validate config
    try:
        config = load_adapter_config(adapter_path)
        is_valid, config_errors = validate_adapter_config(config)
        if not is_valid:
            errors.extend(config_errors)
    except Exception as e:
        errors.append(f"Error validating adapter config: {str(e)}")
    
    is_valid = len(errors) == 0
    
    if raise_on_error and not is_valid:
        raise AdapterValidationError(f"Adapter validation failed: {'; '.join(errors)}")
    
    return is_valid, errors


def get_adapter_info(adapter_path: str | Path) -> Dict:
    """Get human-readable adapter information.
    
    Args:
        adapter_path: Path to the adapter directory
        
    Returns:
        Dictionary with adapter information
    """
    metadata = get_adapter_metadata(adapter_path)
    
    info = {
        "path": metadata["adapter_path"],
        "status": "valid" if metadata.get("config", {}).get("is_valid", False) else "invalid",
        "exists": metadata["exists"],
    }
    
    if metadata.get("config", {}).get("exists"):
        config_info = metadata["config"]
        info["peft_type"] = config_info.get("peft_type")
        info["task_type"] = config_info.get("task_type")
        info["base_model"] = config_info.get("base_model")
        
        if "lora" in config_info:
            lora_info = config_info["lora"]
            info["lora_rank"] = lora_info.get("r")
            info["lora_alpha"] = lora_info.get("lora_alpha")
            info["target_modules"] = lora_info.get("target_modules")
    
    if metadata.get("files"):
        info["files_status"] = "complete" if metadata["files"]["all_required_exist"] else "incomplete"
        info["missing_files"] = metadata["files"]["missing_files"]
    
    return info

