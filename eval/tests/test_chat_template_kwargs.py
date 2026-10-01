import json

import httpx
import pytest

from eval.adapters import OpenAICompatibleAdapter
from eval.config import AdapterType, MODEL_REGISTRY, ModelConfig


def adapter(monkeypatch, extra):
    config = ModelConfig(
        name="nemotron-test", adapter_type=AdapterType.OPENAI_COMPATIBLE,
        base_url="https://example.invalid/v1", api_key_env="NEMOTRON_TEST_API_KEY",
        model_id="fixture-model", max_tokens=32768, extra=extra,
    )
    monkeypatch.setenv("NEMOTRON_TEST_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setitem(MODEL_REGISTRY, config.name, config)
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "fixture-response", "model": "fixture-model",
            "choices": [{"message": {"content": "{}"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        })

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("eval.adapters.httpx.Client", lambda **kwargs: client)
    return OpenAICompatibleAdapter(config), sent


@pytest.mark.parametrize("settings", [{"enable_thinking": True}, {"enable_thinking": False}, {}])
def test_chat_template_kwargs_are_preserved(monkeypatch, settings):
    model, sent = adapter(monkeypatch, {"chat_template_kwargs": settings})
    result = model.call("system fixture", "user fixture")
    assert sent[0]["chat_template_kwargs"] == settings
    assert sent[0]["model"] == "fixture-model"
    assert sent[0]["max_tokens"] == 32768
    assert json.loads(result.raw_json)["choices"][0]["finish_reason"] == "length"


def test_unspecified_chat_template_does_not_change_existing_payloads(monkeypatch):
    model, sent = adapter(monkeypatch, {})
    model.call("system fixture", "user fixture")
    assert "chat_template_kwargs" not in sent[0]


@pytest.mark.parametrize("settings", [False, "enabled", []])
def test_invalid_chat_template_kwargs_fail_before_network(monkeypatch, settings):
    model, sent = adapter(monkeypatch, {"chat_template_kwargs": settings})
    with pytest.raises(ValueError, match="chat_template_kwargs"):
        model.call("system fixture", "user fixture")
    assert sent == []
