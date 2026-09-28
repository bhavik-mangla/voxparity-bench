from pathlib import Path

from voxparity.authoring.generate import author_items, build_item, next_item_id
from voxparity.cli import load_item

GOOD_DRAFT = {
    "transcript": "Okay, go ahead and cancel it.",
    "scenario": "The caller asked about cancelling a subscription; you explained the refund terms.",
    "tools": [
        {
            "name": "end_subscription",
            "description": "Cancel now.",
            "param": {"name": "plan", "type": "string", "description": "Plan id"},
        },
        {
            "name": "offer_retention",
            "description": "Offer a discount before cancelling.",
            "param": {"name": "plan", "type": "string", "description": "Plan id"},
        },
    ],
    "variant_a": {
        "gold_tool": "end_subscription",
        "args": {"plan": ["basic"]},
        "rationale": "settled",
    },
    "variant_b": {
        "gold_tool": "offer_retention",
        "args": {"plan": ["basic"]},
        "rationale": "reluctant",
    },
    "probe": {
        "question": "How does the caller sound?",
        "options": ["settled", "reluctant", "angry"],
        "gold_a": "settled",
        "gold_b": "reluctant",
    },
}


def test_build_item_valid():
    item = build_item(GOOD_DRAFT, "vxp-test-0001", "telecom", "happy", "resigned")
    assert (
        item.variants[0].variant_id == "a_happy" and item.variants[1].gold.tool == "offer_retention"
    )
    assert item.review == "draft"


def test_build_item_rejects_same_gold():
    bad = dict(GOOD_DRAFT, variant_b=dict(GOOD_DRAFT["variant_a"]))
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        build_item(bad, "vxp-test-0002", "telecom", "happy", "resigned")


def test_author_items_screens_and_writes(tmp_path: Path):
    calls = {"n": 0}

    def fake_generate(prompt: str):
        calls["n"] += 1
        if "leaks_emotion" in prompt:
            # leak on the second item only
            return {"leaks_emotion": "ANGRY" in prompt, "emotion": None, "why": ""}
        return (
            GOOD_DRAFT if calls["n"] < 4 else dict(GOOD_DRAFT, transcript="I'm so ANGRY about this")
        )

    out = tmp_path / "items"
    paths = author_items(fake_generate, out, ["zoo_keeping"], 2, screen=True, log=lambda m: None)
    assert len(paths) == 2
    items = [load_item(p) for p in paths]
    assert all(i.review == "screened" for i in items)
    assert items[0].id == "vxp-zookee-0001" and items[1].id == "vxp-zookee-0002"
    assert next_item_id(out, "zoo_keeping") == "vxp-zookee-0003"


def test_explicit_mode_carries_policy():
    from voxparity.authoring.generate import draft_prompt

    draft = dict(GOOD_DRAFT, policy="If the caller sounds reluctant, offer retention first.")
    item = build_item(draft, "vxp-test-0003", "telecom", "happy", "resigned", explicit=True)
    assert item.policy_mode.value == "explicit" and "reluctant" in (item.explicit_policy or "")
    assert '"policy": str' in draft_prompt("telecom", "happy", "resigned", "x", True)
    assert '"policy"' not in draft_prompt("telecom", "happy", "resigned", "x", False)


def test_author_ladders_adds_screened_followup(tmp_path: Path):
    from voxparity.authoring.generate import author_ladders

    item = build_item(GOOD_DRAFT, "vxp-test-0009", "telecom", "happy", "resigned")
    from voxparity.authoring.generate import write_item_yaml

    path = write_item_yaml(item, tmp_path)

    def fake_generate(prompt: str):
        if "leaks_emotion" in prompt:
            return {"leaks_emotion": False, "emotion": None, "why": ""}
        return {"caller_reply": "Yes, that plan is right."}

    n = author_ladders(fake_generate, [path], log=lambda m: None)
    assert n == 2
    reloaded = load_item(path)
    fu = reloaded.variants[0].followup
    assert fu is not None and fu.caller_reply == "Yes, that plan is right."
    assert "ask_clarifying_question" in fu.trigger_tools
    assert fu.final_gold.tool == reloaded.variants[0].gold.tool
