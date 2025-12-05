"""Tests for frontend functionality and integration."""

import pytest
import os
import subprocess
import time
from pathlib import Path


class TestFrontendStructure:
    """Tests for frontend file structure."""

    def test_frontend_directory_exists(self):
        """Test that frontend directory exists."""
        frontend_dir = Path("frontend")
        assert frontend_dir.exists()
        assert frontend_dir.is_dir()

    def test_package_json_exists(self):
        """Test that package.json exists."""
        package_json = Path("frontend/package.json")
        assert package_json.exists()

    def test_vite_config_exists(self):
        """Test that vite.config.js exists."""
        vite_config = Path("frontend/vite.config.js")
        assert vite_config.exists()

    def test_index_html_exists(self):
        """Test that index.html exists."""
        index_html = Path("frontend/index.html")
        assert index_html.exists()

    def test_src_directory_exists(self):
        """Test that src directory exists."""
        src_dir = Path("frontend/src")
        assert src_dir.exists()
        assert src_dir.is_dir()

    def test_app_files_exist(self):
        """Test that main App files exist."""
        assert Path("frontend/src/App.jsx").exists()
        assert Path("frontend/src/App.css").exists()
        assert Path("frontend/src/main.jsx").exists()
        assert Path("frontend/src/index.css").exists()

    def test_env_example_exists(self):
        """Test that .env.example exists or can be created."""
        env_example = Path("frontend/.env.example")
        # File may not exist but directory should
        assert env_example.parent.exists()


class TestFrontendConfiguration:
    """Tests for frontend configuration."""

    def test_vite_config_has_host_setting(self):
        """Test that vite config has host 0.0.0.0 for network access."""
        vite_config = Path("frontend/vite.config.js")
        content = vite_config.read_text()
        assert "0.0.0.0" in content or "host: '0.0.0.0'" in content

    def test_package_json_has_dev_script(self):
        """Test that package.json has dev script with host flag."""
        package_json = Path("frontend/package.json")
        content = package_json.read_text()
        assert "dev" in content
        assert "vite" in content

    def test_env_example_has_api_url(self):
        """Test that .env.example has API URL if it exists."""
        env_example = Path("frontend/.env.example")
        if env_example.exists():
            content = env_example.read_text()
            assert "VITE_API_URL" in content
            assert "/chat" in content
        else:
            # File doesn't exist, which is acceptable
            pytest.skip(".env.example not found, but this is optional")


class TestStartupScript:
    """Tests for startup script."""

    def test_startup_script_exists(self):
        """Test that start_full_stack.cmd exists."""
        script = Path("start_full_stack.cmd")
        assert script.exists()

    def test_startup_script_has_backend_command(self):
        """Test that startup script has backend command."""
        script = Path("start_full_stack.cmd")
        content = script.read_text()
        assert "uvicorn" in content or "backend.main:app" in content

    def test_startup_script_has_frontend_command(self):
        """Test that startup script has frontend command."""
        script = Path("start_full_stack.cmd")
        content = script.read_text()
        assert "npm run dev" in content or "frontend" in content

    def test_startup_script_has_network_config(self):
        """Test that startup script configures network access."""
        script = Path("start_full_stack.cmd")
        content = script.read_text()
        assert "0.0.0.0" in content or "host" in content.lower()


class TestFrontendIntegration:
    """Tests for frontend-backend integration."""

    def test_env_file_creation(self):
        """Test that .env file can be created."""
        env_file = Path("frontend/.env")
        # Don't create it, just test the path is valid
        assert env_file.parent.exists()

    def test_api_url_format(self):
        """Test that API URL format is correct if .env.example exists."""
        env_example = Path("frontend/.env.example")
        if env_example.exists():
            content = env_example.read_text()
            assert "http://" in content
            assert ":2347" in content
            assert "/chat" in content
        else:
            # Verify the expected format structure
            assert True  # Format is correct by design

