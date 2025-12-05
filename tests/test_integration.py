"""Integration tests for end-to-end functionality."""

import pytest


class TestFullIntegration:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_complete_chat_flow(self, test_client):
        """Test complete chat flow from request to response."""
        response = test_client.post(
            "/chat",
            json={
                "prompt": "What is the Constitution of India?",
                "max_new_tokens": 50,
                "temperature": 0.7
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "reply" in data
        assert isinstance(data["reply"], str)
        assert len(data["reply"]) > 0
    
    def test_multiple_sequential_requests(self, test_client):
        """Test handling multiple sequential chat requests."""
        prompts = [
            "What is Section 439?",
            "Explain the doctrine of basic structure",
            "What is the Limitation Act?"
        ]
        
        for prompt in prompts:
            response = test_client.post(
                "/chat",
                json={"prompt": prompt, "max_new_tokens": 30}
            )
            assert response.status_code == 200
            data = response.json()
            assert "reply" in data
    
    def test_conversation_history(self, test_client):
        """Test conversation with message history."""
        response1 = test_client.post(
            "/chat",
            json={
                "messages": [
                    {"role": "user", "content": "What is CrPC?"}
                ]
            }
        )
        assert response1.status_code == 200
        reply1 = response1.json()["reply"]
        
        response2 = test_client.post(
            "/chat",
            json={
                "messages": [
                    {"role": "user", "content": "What is CrPC?"},
                    {"role": "assistant", "content": reply1},
                    {"role": "user", "content": "Tell me more about it"}
                ]
            }
        )
        assert response2.status_code == 200
        assert "reply" in response2.json()


class TestPerformance:
    """Performance and load tests."""
    
    def test_response_time_reasonable(self, test_client):
        """Test that responses are generated in reasonable time."""
        import time
        
        start = time.time()
        response = test_client.post(
            "/chat",
            json={"prompt": "Test", "max_new_tokens": 10}
        )
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 60
    
    def test_concurrent_requests(self, test_client):
        """Test handling concurrent requests."""
        import concurrent.futures
        
        def make_request():
            return test_client.post(
                "/chat",
                json={"prompt": "Test", "max_new_tokens": 10}
            )
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request) for _ in range(3)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        assert all(r.status_code == 200 for r in results)


class TestErrorRecovery:
    """Tests for error recovery and resilience."""
    
    def test_recovery_after_error(self, test_client):
        """Test system recovers after an error."""
        response1 = test_client.post(
            "/chat",
            json={"prompt": "Valid request"}
        )
        assert response1.status_code == 200
        
        response2 = test_client.post(
            "/chat",
            json={}
        )
        assert response2.status_code == 400
        
        response3 = test_client.post(
            "/chat",
            json={"prompt": "Another valid request"}
        )
        assert response3.status_code == 200

