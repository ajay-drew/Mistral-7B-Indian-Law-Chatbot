"""Tests for system prompt integration."""

import pytest
from backend.main import build_prompt, SYSTEM_PROMPT, ChatRequest, Message


class TestSystemPromptIntegration:
    """Tests for system prompt integration in build_prompt function."""
    
    def test_system_prompt_included_with_prompt(self):
        """Test that system prompt is included when using prompt field."""
        request = ChatRequest(prompt="hi")
        result = build_prompt(request)
        
        assert SYSTEM_PROMPT in result
        assert "Question: hi" in result
        assert "Answer:" in result
    
    def test_system_prompt_included_with_messages(self):
        """Test that system prompt is included when using messages array."""
        request = ChatRequest(
            messages=[Message(role="user", content="What is CrPC?")]
        )
        result = build_prompt(request)
        
        assert SYSTEM_PROMPT in result
        assert "Question: What is CrPC?" in result
        assert "Answer:" in result
    
    def test_system_prompt_with_conversation_history(self):
        """Test system prompt with multi-turn conversation."""
        request = ChatRequest(
            messages=[
                Message(role="user", content="What is Section 439?"),
                Message(role="assistant", content="Section 439 deals with..."),
                Message(role="user", content="Tell me more")
            ]
        )
        result = build_prompt(request)
        
        assert SYSTEM_PROMPT in result
        assert "Question: What is Section 439?" in result
        assert "Answer: Section 439 deals with..." in result
        assert "Question: Tell me more" in result
        assert "Answer:" in result
    
    def test_system_prompt_structure(self):
        """Test that system prompt has correct structure."""
        request = ChatRequest(prompt="test")
        result = build_prompt(request)
        
        assert SYSTEM_PROMPT in result
        assert "Question: test" in result
        assert "Answer:" in result
    
    def test_system_prompt_contains_important_rules(self):
        """Test that system prompt contains important rules."""
        assert "NEVER generate random" in SYSTEM_PROMPT or "NEVER generate" in SYSTEM_PROMPT
        assert "consult" in SYSTEM_PROMPT.lower() or "professional lawyer" in SYSTEM_PROMPT.lower()
        assert "AI assistant" in SYSTEM_PROMPT or "AI" in SYSTEM_PROMPT
    
    def test_system_prompt_contains_greeting_guidance(self):
        """Test that system prompt contains greeting instructions."""
        assert "greeting" in SYSTEM_PROMPT.lower()
        assert "briefly" in SYSTEM_PROMPT.lower() or "brief" in SYSTEM_PROMPT.lower()


class TestSystemPromptBehavior:
    """Tests for system prompt behavior with actual API calls."""
    
    def test_greeting_response_structure(self, test_client):
        """Test that greeting responses follow system prompt guidelines."""
        response = test_client.post(
            "/chat",
            json={"prompt": "hi", "max_new_tokens": 50, "temperature": 0.5}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "reply" in data
        assert isinstance(data["reply"], str)
        assert len(data["reply"]) > 0
        assert "[System]" in data["reply"] or "Indian Law Assistant" in data["reply"]
    
    def test_legal_question_gets_detailed_response(self, test_client):
        """Test that legal questions get comprehensive responses."""
        response = test_client.post(
            "/chat",
            json={
                "prompt": "What is Section 439 of CrPC?",
                "max_new_tokens": 200,
                "temperature": 0.7
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "reply" in data
        reply = data["reply"]
        assert len(reply) > 50
        assert isinstance(reply, str)
    
    def test_system_prompt_prevents_random_greeting_responses(self, test_client):
        """Test that system prompt helps prevent random responses to greetings."""
        greetings = ["hi", "hello", "hey", "greetings"]
        
        for greeting in greetings:
            response = test_client.post(
                "/chat",
                json={"prompt": greeting, "max_new_tokens": 50, "temperature": 0.3}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data["reply"], str)
            assert len(data["reply"]) > 0
            assert "[System]" in data["reply"] or "Indian Law Assistant" in data["reply"]
    
    def test_system_prompt_with_messages_array(self, test_client):
        """Test system prompt works with messages array format."""
        response = test_client.post(
            "/chat",
            json={
                "messages": [
                    {"role": "user", "content": "hi"}
                ],
                "max_new_tokens": 50,
                "temperature": 0.5
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "reply" in data
        assert isinstance(data["reply"], str)
        assert len(data["reply"]) > 0

