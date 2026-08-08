import json
from fastapi.testclient import TestClient
from src.api.main import create_app

def run_multi_turn_test():
    app = create_app()
    client = TestClient(app)
    session_id = "multi_turn_demo_session"

    turns = [
        {
            "turn": 1,
            "message": "Hello, my name is Ahmed and my IBAN is AE01234567890123456789012. What is Murabaha financing?",
            "note": "Turn 1: Initial query with PII (Name & UAE IBAN) asking about Murabaha"
        },
        {
            "turn": 2,
            "message": "What happens if I delay my payment under this Murabaha contract?",
            "note": "Turn 2: Follow-up query relying on session context ('this Murabaha contract')"
        },
        {
            "turn": 3,
            "message": "من يلتزم بالصيانة الهيكلية في الإجارة بدلاً من ذلك؟",
            "note": "Turn 3: Arabic language turn asking about structural maintenance in Ijara"
        }
    ]

    print("\n========================================================")
    print("      MAL ISLAMIC FINANCE ASSISTANT - MULTI-TURN DEMO   ")
    print("========================================================\n")

    for turn in turns:
        print(f"--- TURN {turn['turn']}: {turn['note']} ---")
        print(f"User Message: {turn['message']}")

        response = client.post(
            "/chat",
            json={
                "session_id": session_id,
                "message": turn["message"]
            }
        )

        assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"
        data = response.json()

        print(f"Session ID  : {data['session_id']}")
        print(f"Assistant   :\n{data['response']}\n")
        print(f"Trace Logs  : {json.dumps(data['trace'], indent=2)}\n")
        print("-" * 60 + "\n")

def test_multi_turn_conversation():
    run_multi_turn_test()

if __name__ == "__main__":
    run_multi_turn_test()
