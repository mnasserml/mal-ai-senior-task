import pytest
from src.infrastructure.container import Container


@pytest.fixture(scope="module")
def pii_redactor():
    """Shared adapter instance wired via DI container."""
    container = Container()
    return container.base_pii_adapter()


# ── Emirates ID ─────────────────────────────────────────
def test_emirates_id(pii_redactor):
    text = "My Emirates ID is 784-1990-1234567-1 please verify my account."
    result = pii_redactor.redact(text)
    print(f"\n[Emirates ID Test]\n  Original: {text}\n  Redacted: {result.redacted_text}\n  Entities: {[e['entity_type'] for e in result.detected_entities]}")

    assert "784-1990-1234567-1" not in result.redacted_text
    assert "<EMIRATES_ID>" in result.redacted_text
    assert result.redacted_count >= 1


# ── UAE IBAN ────────────────────────────────────────────
def test_uae_iban(pii_redactor):
    text = "Please transfer funds to my IBAN AE123456789012345678901."
    result = pii_redactor.redact(text)
    print(f"\n[UAE IBAN Test]\n  Original: {text}\n  Redacted: {result.redacted_text}\n  Entities: {[e['entity_type'] for e in result.detected_entities]}")

    assert "AE123456789012345678901" not in result.redacted_text
    assert "<UAE_IBAN>" in result.redacted_text
    assert result.redacted_count >= 1


# ── Account Number ──────────────────────────────────────
def test_account_number(pii_redactor):
    text = "My Mal bank account number is 987654321012."
    result = pii_redactor.redact(text)
    print(f"\n[Account Number Test]\n  Original: {text}\n  Redacted: {result.redacted_text}\n  Entities: {[e['entity_type'] for e in result.detected_entities]}")

    assert "987654321012" not in result.redacted_text
    assert "<ACCOUNT_NUMBER>" in result.redacted_text
    assert result.redacted_count >= 1


# ── UAE Phone Number ────────────────────────────────────
def test_uae_phone_number(pii_redactor):
    text = "Contact me at +971501234567 or 0501234567 for details."
    result = pii_redactor.redact(text)
    print(f"\n[UAE Phone Test]\n  Original: {text}\n  Redacted: {result.redacted_text}\n  Entities: {[e['entity_type'] for e in result.detected_entities]}")

    assert "+971501234567" not in result.redacted_text
    assert "<PHONE_NUMBER>" in result.redacted_text
    assert result.redacted_count >= 1


# ── Email Address ───────────────────────────────────────
def test_email_address(pii_redactor):
    text = "Send contract to customer.support@malbank.ae for review."
    result = pii_redactor.redact(text)
    print(f"\n[Email Address Test]\n  Original: {text}\n  Redacted: {result.redacted_text}\n  Entities: {[e['entity_type'] for e in result.detected_entities]}")

    assert "customer.support@malbank.ae" not in result.redacted_text
    assert "<EMAIL_ADDRESS>" in result.redacted_text
    assert result.redacted_count >= 1


# ── English Person Name ─────────────────────────────────
def test_english_person_name(pii_redactor):
    text = "My name is John Smith and I want to ask about Sukuk investment."
    result = pii_redactor.redact(text)
    print(f"\n[English Name Test]\n  Original: {text}\n  Redacted: {result.redacted_text}\n  Entities: {[e['entity_type'] for e in result.detected_entities]}")

    assert "<PERSON>" in result.redacted_text
    assert result.redacted_count >= 1


# ── Arabic Compound Name ────────────────────────────────
def test_arabic_compound_name(pii_redactor):
    text = "مرحباً، أنا محمد بن عبد العزيز الهاشمي ورقم حسابي هو AE987654321098765432109"
    result = pii_redactor.redact(text)
    print(f"\n[Arabic Name Test]\n  Original: {text}\n  Redacted: {result.redacted_text}\n  Entities: {[e['entity_type'] for e in result.detected_entities]}")

    assert "<PERSON>" in result.redacted_text
    assert "<UAE_IBAN>" in result.redacted_text
    assert result.redacted_count >= 2


# ── Arabic Name + Phone ─────────────────────────────────
def test_arabic_name_and_phone(pii_redactor):
    text = "اسمي أحمد علي ورقم هاتفي +971501234567 أود الاستفسار عن حساب التوفير."
    result = pii_redactor.redact(text)
    print(f"\n[Arabic Name + Phone Test]\n  Original: {text}\n  Redacted: {result.redacted_text}\n  Entities: {[e['entity_type'] for e in result.detected_entities]}")

    assert "+971501234567" not in result.redacted_text
    assert result.redacted_count >= 1


# ── Combined English Name + Emirates ID ─────────────────
def test_english_name_and_emirates_id(pii_redactor):
    text = "My name is Ahmed, Emirates ID 784-1990-1234567-1. What happens if I delay payment in Murabaha?"
    result = pii_redactor.redact(text)
    print(f"\n[EN Name + Emirates ID Test]\n  Original: {text}\n  Redacted: {result.redacted_text}\n  Entities: {[e['entity_type'] for e in result.detected_entities]}")

    assert "784-1990-1234567-1" not in result.redacted_text
    assert "<EMIRATES_ID>" in result.redacted_text
    assert "<PERSON>" in result.redacted_text
    assert result.redacted_count >= 2


# ── CLI entry point ─────────────────────────────────────
if __name__ == "__main__":
    print("\n========================================================")
    print("       COMPREHENSIVE PII REDACTION TEST SUITE")
    print("========================================================\n")

    adapter = PresidioPIIAdapter()

    test_cases = [
        ("Emirates ID",         "My Emirates ID is 784-1990-1234567-1 please verify my account."),
        ("UAE IBAN",             "Please transfer funds to my IBAN AE123456789012345678901."),
        ("Account Number",      "My Mal bank account number is 987654321012."),
        ("UAE Phone",           "Contact me at +971501234567 or 0501234567 for details."),
        ("Email Address",       "Send contract to customer.support@malbank.ae for review."),
        ("English Name (GLiNER)", "My name is John Smith and I want to ask about Sukuk investment."),
        ("Arabic Name (GLiNER)", "مرحباً، أنا محمد بن عبد العزيز الهاشمي ورقم حسابي هو AE987654321098765432109"),
        ("Arabic Name + Phone", "اسمي أحمد علي ورقم هاتفي +971501234567 أود الاستفسار عن حساب التوفير."),
        ("EN Name + Emirates ID", "My name is Ahmed, Emirates ID 784-1990-1234567-1. What happens if I delay payment in Murabaha?"),
    ]

    for name, text in test_cases:
        res = adapter.redact(text)
        entities = [e["entity_type"] for e in res.detected_entities]
        status = "✅" if res.redacted_count >= 1 else "❌"
        print(f"{status} {name}")
        print(f"   Original : {text}")
        print(f"   Redacted : {res.redacted_text}")
        print(f"   Entities : {entities}  (count={res.redacted_count})")
        print("-" * 60)

    print("========================================================\n")
