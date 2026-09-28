import math

import pytest

from voxparity.scoring.stats import (
    paired_diff_ci,
    proportion_ci,
    required_n_for_halfwidth,
    required_n_per_arm,
)


def test_halfwidth_reproduces_blueprint_figure():
    # blueprint: "~1,000 items per headline cell -> +/-3.1% 95% CI at p=0.5"
    assert required_n_for_halfwidth(0.031) == pytest.approx(1000, abs=5)


def test_unpaired_power_criterion_is_conservative():
    # unpaired two-proportion test for a 3-pt gap needs far more than 1,000 items;
    # this is why the benchmark scores models PAIRED on shared items.
    n = required_n_per_arm(0.50, 0.03, power=0.80)
    assert 4000 < n < 4700


def test_proportion_ci_basic():
    est = proportion_ci([True] * 60 + [False] * 40)
    assert est.mean == pytest.approx(0.6)
    assert est.lo < 0.6 < est.hi
    assert est.n == 100


def test_clustered_se_not_smaller_than_needed():
    # perfectly correlated within clusters -> clustered SE must exceed naive SE
    successes, clusters = [], []
    for cid in range(20):
        val = cid % 2 == 0
        for _ in range(5):
            successes.append(val)
            clusters.append(f"item-{cid}")
    naive = proportion_ci(successes)
    clustered = proportion_ci(successes, cluster_ids=clusters)
    assert clustered.clusters == 20
    assert clustered.se > naive.se


def test_clustered_requires_two_clusters():
    with pytest.raises(ValueError):
        proportion_ci([True, False], cluster_ids=["a", "a"])


def test_paired_diff_zero_when_identical():
    a = [True, False, True, True]
    est = paired_diff_ci(a, a)
    assert est.mean == 0.0
    assert math.isclose(est.se, 0.0)


def test_paired_diff_direction():
    a = [True] * 80 + [False] * 20
    b = [True] * 60 + [False] * 40
    est = paired_diff_ci(a, b)
    assert est.mean == pytest.approx(0.2)


class TestUnbiasedHitRate:
    """D040: raw accuracy is misleading under response bias, and voice-only
    emotion judgement always has one — CREMA-D listeners chose 'neutral' on 43.9%
    of trials against a 16.7% base rate."""

    def test_a_rater_who_answers_neutral_to_everything_scores_near_chance(self):
        from voxparity.scoring.stats import unbiased_hit_rates

        responses = [("neutral", "neutral")] * 10
        responses += [(e, "neutral") for e in ["sad"] * 10 + ["happy"] * 10]
        rates = unbiased_hit_rates(responses)
        # raw looks perfect on neutral...
        assert rates["neutral"]["raw"] == 1.0
        # ...but Hu discounts it by how indiscriminately the label was used
        assert rates["neutral"]["hu"] < 0.35
        assert rates["sad"]["hu"] == 0.0

    def test_a_discriminating_rater_is_not_penalised(self):
        from voxparity.scoring.stats import unbiased_hit_rates

        responses = [("neutral", "neutral")] * 10 + [("sad", "sad")] * 10
        rates = unbiased_hit_rates(responses)
        assert rates["neutral"]["hu"] == 1.0
        assert rates["sad"]["hu"] == 1.0

    def test_hu_formula(self):
        from voxparity.scoring.stats import unbiased_hit_rate

        # 5 hits, 10 stimuli of the category, label used 20 times: 25/200
        assert unbiased_hit_rate(5, 10, 20) == 0.125
        assert unbiased_hit_rate(0, 10, 0) == 0.0

    def test_rosenthal_rubin_pi_puts_option_counts_on_one_scale(self):
        from voxparity.scoring.stats import rosenthal_rubin_pi

        # chance in a 5-option task maps to 0.50
        assert abs(rosenthal_rubin_pi(0.20, 5) - 0.50) < 1e-9
        # chance in a 3-option task also maps to 0.50
        assert abs(rosenthal_rubin_pi(1 / 3, 3) - 0.50) < 1e-9
        # and a 2-option rate is unchanged
        assert abs(rosenthal_rubin_pi(0.8, 2) - 0.8) < 1e-9


class TestCueClassMapping:
    """D048: Hu needs a response category shared across stimuli, but our probe
    options are bespoke prose per item. Pooling them onto the emotion each option
    is gold for makes the bias correction actually bite — without it every
    category has n=1 and Hu silently equals raw."""

    def test_gold_options_take_their_variant_emotion(self):
        import sys

        sys.path.insert(0, "tests")
        from test_schemas import make_item
        from voxparity.scoring.stats import cue_class_map

        item = make_item()
        mapping = cue_class_map(item)
        for variant in item.variants:
            option = item.perception_probe.gold_by_variant[variant.variant_id]
            assert mapping[option] == variant.emotion.value

    def test_distractors_pool_into_one_class(self):
        import sys

        sys.path.insert(0, "tests")
        from test_schemas import make_item
        from voxparity.scoring.stats import cue_class_map

        item = make_item()
        item.perception_probe.options = [*item.perception_probe.options, "Bored", "Amused"]
        mapping = cue_class_map(item)
        # pooling matters: distractors are what a guessing rater over-uses, and
        # Hu can only see the over-use if they share a category
        assert mapping["Bored"] == "other"
        assert mapping["Amused"] == "other"


def test_probe_hu_exposes_label_collapse():
    """A judge answering one label to everything scores well raw and badly on Hu."""
    import sys

    sys.path.insert(0, "tests")
    from test_schemas import make_item
    from voxparity.harness.report import probe_hu

    item = make_item()
    neutral_option = item.perception_probe.gold_by_variant[item.variants[0].variant_id]
    records = []
    for variant in item.variants * 5:
        gold = item.perception_probe.gold_by_variant[variant.variant_id]
        records.append(
            {
                "item_id": item.id,
                "variant_id": variant.variant_id,
                "condition": "probe",
                "error": "",
                "scores": {
                    "gold": gold,
                    "answer": neutral_option,
                    "passed": gold == neutral_option,
                },
            }
        )
    result = probe_hu(records, {item.id: item})
    assert result["raw"] == 0.5  # answered one label; right half the time
    assert result["hu"] < result["raw"]  # ...but indiscriminately
    assert result["collapse_gap"] > 0
