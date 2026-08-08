import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.infrastructure.evaluation.llm_judge import LLMGroundingJudge

client = TestClient(app)

def _get_judge() -> LLMGroundingJudge:
    """Instantiate LLM Judge using container's LLM provider."""
    llm_provider = app.container.base_llm_provider()
    return LLMGroundingJudge(llm_provider=llm_provider)

def _get_retrieved_context_text(chunk_ids: list[str]) -> str:
    """Retrieve full context text for the given chunk IDs from vector store."""
    vector_store = app.container.base_vector_store()
    matching_chunks = [c for c in vector_store.documents if c.chunk_id in chunk_ids]
    if not matching_chunks:
        matching_chunks = vector_store.documents[:3]
    return "\n\n".join([f"[{c.doc_name}]: {c.content}" for c in matching_chunks])

def _evaluate_query_grounding(query: str, session_id: str):
    response = client.post("/chat", json={"session_id": session_id, "message": query})
    assert response.status_code == 200
    data = response.json()

    answer = data["response"]
    chunk_ids = data["trace"].get("retrieved_chunk_ids", [])
    context_text = _get_retrieved_context_text(chunk_ids)

    judge = _get_judge()
    eval_result = judge.evaluate_grounding(query=query, retrieved_context=context_text, response_text=answer)

    return eval_result, query, answer


def test_grounding_english_murabaha():
    """Verify English query about Murabaha rules using LLM-as-a-Judge."""
    query = "What happens if a customer delays payment in a Murabaha contract?"
    eval_result, _, _ = _evaluate_query_grounding(query, "eval_grounding_1")
    assert eval_result.is_grounded, f"LLM Judge rejected grounding: {eval_result.reasoning}"
    assert eval_result.overall_score >= 0.70, f"Score too low: {eval_result.overall_score}"


def test_grounding_arabic_ijara():
    """Verify Arabic query about Ijara maintenance using LLM-as-a-Judge."""
    query = "من الذي يتحمل الصيانة الهيكلية في الإجارة المنتهية بالتمليك؟"
    eval_result, _, _ = _evaluate_query_grounding(query, "eval_grounding_2")
    assert eval_result.is_grounded, f"LLM Judge rejected grounding: {eval_result.reasoning}"
    assert eval_result.overall_score >= 0.70, f"Score too low: {eval_result.overall_score}"


if __name__ == "__main__":
    print("\n========================================================")
    print("      LLM-AS-A-JUDGE RAG GROUNDING EVALUATION DEMO")
    print("========================================================\n")

    print("--- Test 1: English Murabaha Payment Delay Rules ---")
    res1, q1, ans1 = _evaluate_query_grounding("What happens if a customer delays payment in a Murabaha contract?", "demo_g1")
    print(f"Query    : {q1}")
    print(f"Response : {ans1}")
    print(f"Grounded : {'✅ YES' if res1.is_grounded else '❌ NO'}")
    print(f"Relevant : {'✅ YES' if res1.is_relevant else '❌ NO'}")
    print(f"Scores   : Groundedness={res1.groundedness_score:.2f} | Relevance={res1.relevance_score:.2f} | Overall={res1.overall_score:.2f}")
    print(f"Judge Reasoning: {res1.reasoning}\n")

    print("--- Test 2: Arabic Ijara Structural Maintenance Rules ---")
    res2, q2, ans2 = _evaluate_query_grounding("من الذي يتحمل الصيانة الهيكلية في الإجارة المنتهية بالتمليك؟", "demo_g2")
    print(f"Query    : {q2}")
    print(f"Response : {ans2}")
    print(f"Grounded : {'✅ YES' if res2.is_grounded else '❌ NO'}")
    print(f"Relevant : {'✅ YES' if res2.is_relevant else '❌ NO'}")
    print(f"Scores   : Groundedness={res2.groundedness_score:.2f} | Relevance={res2.relevance_score:.2f} | Overall={res2.overall_score:.2f}")
    print(f"Judge Reasoning: {res2.reasoning}\n")

    print("========================================================\n")
