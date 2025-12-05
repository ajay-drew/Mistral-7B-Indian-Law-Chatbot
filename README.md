# Mistral Indian Law Chatbot

Fine-tuned Mistral 7B model specialized in Indian legal matters with a modern web interface.

## Project Summary

This project implements a complete full-stack application for an AI-powered Indian Law assistant using a fine-tuned Mistral 7B model. The system includes a FastAPI backend, a React-based frontend with Claude-style UI, comprehensive test suite, and network accessibility features.

## What Was Built

### Backend (FastAPI)
- **Main API** (`backend/main.py`): FastAPI service with chat endpoint
- **Model Integration**: PEFT (Parameter-Efficient Fine-Tuning) adapter loading
- **System Prompt**: Integrated system prompt for consistent AI behavior
- **CORS Configuration**: Enabled for cross-origin requests
- **Network Access**: Configured to accept connections from other devices (0.0.0.0)
- **Evaluation Module** (`backend/evaluation.py`): Comprehensive metrics (ROUGE, BLEU, perplexity, accuracy, F1, BERTScore)

### Frontend (React + Vite)
- **Claude-Style UI**: Modern chat interface similar to Claude's design
- **Real-time Chat**: Interactive messaging with loading states
- **Responsive Design**: Mobile-friendly layout
- **Network Access**: Configured to run on 0.0.0.0 for WiFi accessibility
- **Environment Configuration**: `.env` support for API URL configuration

### Testing Suite
- **Backend API Tests** (`tests/test_backend_api.py`): 15 tests for API endpoints
- **Integration Tests** (`tests/test_integration.py`): 6 end-to-end tests
- **Model Functionality Tests** (`tests/test_model_functionality.py`): 12 model tests
- **System Prompt Tests** (`tests/test_system_prompt.py`): 10 system prompt tests
- **Comprehensive Model Tests** (`tests/test_model_comprehensive.py`): Advanced model behavior tests
- **CUDA Tests** (`tests/test_cuda.py`): GPU/CUDA support tests
- **Evaluation Metrics Tests** (`tests/test_evaluation_metrics.py`): Evaluation module tests
- **Frontend Tests** (`tests/test_frontend.py`): Frontend structure and configuration tests
- **Test Configuration** (`tests/conftest.py`): Shared fixtures and test client setup

### Startup Scripts
- **Full Stack Launcher** (`start_full_stack.cmd`): Single command to start both backend and frontend
- **Network Detection**: Automatically detects local IP address
- **Environment Setup**: Creates frontend `.env` file if missing
- **Multi-Window**: Opens backend and frontend in separate windows

## How It Was Built

### 1. Backend Development
- Created FastAPI application with async model loading
- Implemented ModelBundle class for lazy model initialization
- Added system prompt integration in prompt building
- Configured CORS middleware for frontend access
- Set up logging and error handling
- Created evaluation metrics module with optional dependencies

### 2. Frontend Development
- Set up React + Vite project structure
- Created Claude-inspired chat interface with:
  - Clean, modern design
  - Message bubbles (user/assistant)
  - Loading animations
  - Auto-scrolling
  - Keyboard shortcuts (Enter to send, Shift+Enter for new line)
- Configured Vite for network access (0.0.0.0)
- Added environment variable support for API URL

### 3. Testing Implementation
- Created comprehensive pytest suite covering:
  - API endpoint validation
  - Model generation functionality
  - System prompt integration
  - Error handling
  - CUDA support
  - Frontend structure
- Used mocking for model/tokenizer to avoid loading actual models in tests
- Created custom test client for async FastAPI testing
- Added pytest markers for conditional test execution

### 4. Network Accessibility
- Backend: Configured uvicorn to listen on 0.0.0.0:2347
- Frontend: Configured Vite dev server to listen on 0.0.0.0:5173
- Startup script: Automatically detects local IP and creates `.env` file
- CORS: Enabled for all origins to allow frontend access

### 5. Code Quality
- Fixed encoding issues (removed null bytes from corrupted files)
- Recreated all test files with proper UTF-8 encoding
- Added proper error handling throughout
- Implemented async/await patterns correctly
- Used type hints and Pydantic models for validation

## File Structure

```
.
├── backend/
│   ├── main.py              # FastAPI application
│   └── evaluation.py        # Evaluation metrics module
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main React component
│   │   ├── App.css          # Styles
│   │   ├── main.jsx         # React entry point
│   │   └── index.css        # Global styles
│   ├── index.html           # HTML template
│   ├── package.json         # Dependencies
│   ├── vite.config.js       # Vite configuration
│   └── .env.example         # Environment template
├── tests/
│   ├── conftest.py          # Pytest fixtures
│   ├── test_backend_api.py  # API tests
│   ├── test_integration.py  # Integration tests
│   ├── test_model_functionality.py  # Model tests
│   ├── test_system_prompt.py  # System prompt tests
│   ├── test_model_comprehensive.py  # Advanced model tests
│   ├── test_cuda.py         # CUDA tests
│   ├── test_evaluation_metrics.py  # Evaluation tests
│   └── test_frontend.py     # Frontend tests
├── start_full_stack.cmd     # Startup script
├── pytest.ini               # Pytest configuration
└── README.md                # This file
```

## Running the Application

### Quick Start
```cmd
start_full_stack.cmd
```

This will:
1. Detect your local IP address
2. Start backend on http://0.0.0.0:2347
3. Start frontend on http://0.0.0.0:5173
4. Create frontend/.env if needed
5. Display network access URLs

### Manual Start

**Backend:**
```cmd
python -m uvicorn backend.main:app --host 0.0.0.0 --port 2347 --reload
```

**Frontend:**
```cmd
cd frontend
npm install
npm run dev
```

### Network Access
- **Local**: http://localhost:5173 (frontend), http://localhost:2347 (backend)
- **Network**: http://YOUR_IP:5173 (frontend), http://YOUR_IP:2347 (backend)
- Make sure `frontend/.env` has: `VITE_API_URL=http://YOUR_IP:2347/chat`

## Running Tests

```cmd
python -m pytest tests/ -v
```

Run specific test suites:
```cmd
python -m pytest tests/test_backend_api.py -v
python -m pytest tests/test_frontend.py -v
```

## Key Features

1. **Claude-Style UI**: Modern, clean chat interface
2. **Network Access**: Accessible from devices on same WiFi
3. **Comprehensive Testing**: 50+ tests covering all functionality
4. **System Prompt**: Consistent AI behavior for greetings and legal queries
5. **Error Handling**: Graceful error handling throughout
6. **Auto-Configuration**: Startup script handles network setup automatically

## Dependencies

### Backend
- FastAPI
- PyTorch
- Transformers
- PEFT
- bitsandbytes (for quantization)

### Frontend
- React 18
- Vite 5

### Testing
- pytest
- pytest-asyncio
- httpx (for async test client)

## Notes

- Evaluation metrics module has optional dependencies (nltk, rouge-score, sklearn, bert-score)
- CUDA tests are marked and will skip if CUDA is not available
- Frontend automatically connects to backend API
- All tests use mocking to avoid loading actual models
