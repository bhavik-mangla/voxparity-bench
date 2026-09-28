import pytest

from voxparity.providers.keys import KeyPool


def test_pool_prefers_plural_env(monkeypatch):
    monkeypatch.setenv("FOO_API_KEYS", "a, b ,c")
    monkeypatch.setenv("FOO_API_KEY", "z")
    p = KeyPool("FOO")
    assert p.keys == ["z", "a", "b", "c"] and p.current == "z"


def test_rotate_cycles_and_reports_exhaustion(monkeypatch):
    monkeypatch.setenv("BAR_API_KEYS", "k1,k2")
    monkeypatch.delenv("BAR_API_KEY", raising=False)
    p = KeyPool("BAR")
    assert p.rotate() and p.current == "k2"
    assert p.rotate() and p.current == "k1"


def test_single_key_cannot_rotate(monkeypatch):
    monkeypatch.setenv("BAZ_API_KEY", "only")
    monkeypatch.delenv("BAZ_API_KEYS", raising=False)
    p = KeyPool("BAZ")
    assert p.size == 1 and not p.rotate()


def test_missing_keys_raise(monkeypatch):
    monkeypatch.delenv("NOPE_API_KEY", raising=False)
    monkeypatch.delenv("NOPE_API_KEYS", raising=False)
    with pytest.raises(RuntimeError):
        KeyPool("NOPE")


def test_openai_tool_decl_conversion():
    from voxparity.providers.groq import openai_tool_decl

    g = {
        "name": "book",
        "description": "d",
        "parameters": {
            "type": "OBJECT",
            "properties": {"slot": {"type": "STRING", "description": "s"}},
            "required": ["slot"],
        },
    }
    o = openai_tool_decl(g)
    assert o["parameters"]["properties"]["slot"]["type"] == "string"
    assert o["parameters"]["required"] == ["slot"]
