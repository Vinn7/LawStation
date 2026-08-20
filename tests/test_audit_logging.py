from backend.app.core.logging import redact, summary


def test_sensitive_values_are_redacted():
    value = "手机13812345678 身份证110101199001011234 邮箱test@example.com"
    cleaned = redact(value)
    assert "13812345678" not in cleaned
    assert "110101199001011234" not in cleaned
    assert "test@example.com" not in cleaned


def test_secret_keys_and_summary_limit():
    cleaned = redact({"api_key": "secret", "query": "合同纠纷"})
    assert cleaned["api_key"] == "***"
    assert summary("a" * 300, 20) == "a" * 20
