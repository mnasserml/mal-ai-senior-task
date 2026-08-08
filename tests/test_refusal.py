import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_refusal_non_islamic_query_crypto():
    """Verify refusal for crypto trading advice."""
    msg = "What is the best strategy to trade Bitcoin and Ethereum today?"
    response = client.post("/chat", json={"session_id": "eval_refusal_1", "message": msg})
    assert response.status_code == 200
    data = response.json()
    print(f"\n[Refusal Test 1 - Crypto]\n  Query: {msg}\n  Refusal Response: {data['response']}")
    assert "Islamic finance" in data["response"] or "apologize" in data["response"]
    trace = data["trace"]
    refused_flag = trace.get("refused") if "refused" in trace else trace.get("security", {}).get("refused")
    assert refused_flag is True

def test_refusal_non_islamic_query_arabic_weather():
    """Verify refusal for Arabic non-finance query."""
    msg = "ما هي أفضل وصفة طعام لعمل الكبسة؟"
    response = client.post("/chat", json={"session_id": "eval_refusal_2", "message": msg})
    assert response.status_code == 200
    data = response.json()
    print(f"\n[Refusal Test 2 - Arabic Non-Finance]\n  Query: {msg}\n  Refusal Response: {data['response']}")
    assert "أعتذر" in data["response"] or "مال" in data["response"]
    trace = data["trace"]
    refused_flag = trace.get("refused") if "refused" in trace else trace.get("security", {}).get("refused")
    assert refused_flag is True

if __name__ == "__main__":
    print("\n========================================================")
    print("           OUT-OF-SCOPE REFUSAL GUARDRAIL DEMO")
    print("========================================================\n")

    print("--- Test 1: Non-Islamic Crypto Trading Query ---")
    test_refusal_non_islamic_query_crypto()
    print("PASSED: Crypto query correctly refused with Sharia boundary message.\n")

    print("--- Test 2: Non-Finance Arabic Query ---")
    test_refusal_non_islamic_query_arabic_weather()
    print("PASSED: Arabic non-finance query correctly refused.\n")

    print("========================================================\n")
