"""FLAG-006: concurrent store writers must not clobber each other's rows."""

from pathlib import Path

from voxparity.stimuli.store import StimulusRecord, StimulusStore


def _rec(item: str, gates=None) -> StimulusRecord:
    return StimulusRecord(
        item_id=item,
        variant_id="a",
        sha256="",
        engine="gemini",
        model="m",
        voice="v",
        prompt="p",
        gates=gates or {},
    )


def test_two_writers_merge_instead_of_clobber(tmp_path: Path):
    s1 = StimulusStore(tmp_path)
    s1.put(b"RIFF0000WAVE", _rec("vxp-one-0001"))
    # A second process loads the store now...
    s2 = StimulusStore(tmp_path)
    # ...then the first writes a NEW row s2 has never seen...
    s1.put(b"RIFF0000WAVF", _rec("vxp-two-0001"))
    # ...and the second saves a gate update from its stale memory.
    s2.set_gate("vxp-one-0001", "a", "gemini", "human_check", {"passed": True})
    # Both rows must survive, and the gate must be applied.
    s3 = StimulusStore(tmp_path)
    assert s3.get("vxp-two-0001", "a", "gemini") is not None
    assert s3.get("vxp-one-0001", "a", "gemini").gates["human_check"]["passed"] is True


def test_own_changes_win_over_disk(tmp_path: Path):
    s1 = StimulusStore(tmp_path)
    s1.put(b"RIFF0000WAVE", _rec("vxp-one-0001"))
    s2 = StimulusStore(tmp_path)
    s2.set_gate("vxp-one-0001", "a", "gemini", "cue_check", {"passed": False})
    s1.set_gate("vxp-one-0001", "a", "gemini", "cue_check", {"passed": True})
    s3 = StimulusStore(tmp_path)
    assert s3.get("vxp-one-0001", "a", "gemini").gates["cue_check"]["passed"] is True


def test_stale_unchanged_rows_do_not_resurrect(tmp_path: Path):
    """D076: a long-lived writer must not write back load-time copies of rows
    another process has since replaced."""
    s1 = StimulusStore(tmp_path)
    s1.put(b"RIFF0000WAVE", _rec("vxp-one-0001"))
    s1.put(b"RIFF0000WAVF", _rec("vxp-two-0001"))
    old_sha = s1.get("vxp-one-0001", "a", "gemini").sha256
    # Long-lived process loads now (holds stale copies of BOTH rows)...
    sweep = StimulusStore(tmp_path)
    # ...a fast writer replaces row one (re-render)...
    s1.put(b"RIFF0000WAVX", _rec("vxp-one-0001"))
    new_sha = s1.get("vxp-one-0001", "a", "gemini").sha256
    assert new_sha != old_sha
    # ...then the long-lived process gates only row TWO.
    sweep.set_gate("vxp-two-0001", "a", "gemini", "cue_check", {"passed": True})
    s3 = StimulusStore(tmp_path)
    assert s3.get("vxp-one-0001", "a", "gemini").sha256 == new_sha  # not resurrected
    assert s3.get("vxp-two-0001", "a", "gemini").gates["cue_check"]["passed"] is True
