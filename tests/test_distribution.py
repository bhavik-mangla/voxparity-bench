"""Tests for rater-distribution scoring. Pure functions, no I/O, no fixtures on disk."""

from __future__ import annotations

import math

import pytest

from voxparity.scoring.distribution import (
    consensus,
    is_contested,
    kl_divergence,
    krippendorff_alpha,
    macro_f1,
    multilabel_targets,
    rater_distribution,
)

# The four-class space from Chou & Lee's worked example (Sec. 4.4.5).
EMOTIONS = ("neutral", "angry", "sad", "happy")


# --------------------------------------------------------------------------------------
# rater_distribution
# --------------------------------------------------------------------------------------


def test_rater_distribution_matches_the_paper_worked_example() -> None:
    # Paper: five raters give angry, sad, sad, neutral, angry over (N, A, S, H)
    # -> (0.2, 0.4, 0.4, 0.0).
    answers = ["angry", "sad", "sad", "neutral", "angry"]
    dist = rater_distribution(answers, EMOTIONS)
    assert dist == {"neutral": 0.2, "angry": 0.4, "sad": 0.4, "happy": 0.0}


def test_rater_distribution_keeps_zero_vote_labels_so_c_is_readable() -> None:
    # Zero-vote labels must survive: C is read off the dict, and dropping them would
    # silently raise the 1/C threshold for everyone else.
    dist = rater_distribution(["angry", "angry"], EMOTIONS)
    assert set(dist) == set(EMOTIONS)
    assert dist["happy"] == 0.0
    assert math.isclose(math.fsum(dist.values()), 1.0)


def test_rater_distribution_preserves_label_order() -> None:
    assert list(rater_distribution(["sad"], EMOTIONS)) == list(EMOTIONS)


def test_rater_distribution_rejects_out_of_schema_answer() -> None:
    # Silently dropping would shrink the denominator and inflate the other shares.
    with pytest.raises(ValueError, match="not in the label space"):
        rater_distribution(["angry", "elated"], EMOTIONS)


def test_rater_distribution_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError):
        rater_distribution([], EMOTIONS)
    with pytest.raises(ValueError):
        rater_distribution(["angry"], [])


def test_rater_distribution_rejects_duplicate_labels() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        rater_distribution(["angry"], ("angry", "angry", "sad"))


# --------------------------------------------------------------------------------------
# multilabel_targets -- the 1/C binarisation
# --------------------------------------------------------------------------------------


def test_multilabel_targets_matches_the_paper_worked_example() -> None:
    # (0.2, 0.4, 0.4, 0.0) with threshold 1/4 = 0.25 binarises to (0, 1, 1, 0).
    dist = rater_distribution(["angry", "sad", "sad", "neutral", "angry"], EMOTIONS)
    assert multilabel_targets(dist) == {"angry", "sad"}


def test_multilabel_targets_keeps_a_genuine_secondary_emotion() -> None:
    # 2/5 angry + 2/5 sad + 1/5 neutral: majority rule would DISCARD this item entirely.
    # 1/C keeps both blended emotions -- the whole reason for the Contested track.
    dist = rater_distribution(["angry", "angry", "sad", "sad", "neutral"], EMOTIONS)
    assert multilabel_targets(dist) == {"angry", "sad"}


def test_multilabel_targets_boundary_is_strict_exceeds_not_at_least() -> None:
    # C = 4, threshold exactly 0.25. A label sitting ON the threshold is NOT a target.
    dist = {"neutral": 0.25, "angry": 0.25, "sad": 0.25, "happy": 0.25}
    assert multilabel_targets(dist) == set()

    # One tick above and one tick below the same boundary.
    dist = {"neutral": 0.30, "angry": 0.25, "sad": 0.25, "happy": 0.20}
    assert multilabel_targets(dist) == {"neutral"}


def test_multilabel_targets_boundary_survives_float_representation_error() -> None:
    # 2/6 == 1/3 mathematically but not necessarily bit-for-bit. The definition, not
    # float noise, must decide: exactly-at-threshold is excluded.
    dist = rater_distribution(["a", "a", "b", "b", "c", "c"], ("a", "b", "c"))
    assert dist["a"] != 1 / 3 or dist["a"] == 1 / 3  # either representation is fine
    assert multilabel_targets(dist) == set()


def test_multilabel_targets_unanimous_selects_exactly_one() -> None:
    dist = rater_distribution(["angry"] * 5, EMOTIONS)
    assert multilabel_targets(dist) == {"angry"}


def test_multilabel_targets_maximally_split_yields_empty_set() -> None:
    # Documented edge case: a perfectly uniform distribution has NOTHING above 1/C.
    # Such items belong to kl_divergence, not macro_f1.
    dist = rater_distribution(list(EMOTIONS), EMOTIONS)
    assert dist == {label: 0.25 for label in EMOTIONS}
    assert multilabel_targets(dist) == set()


def test_multilabel_targets_two_label_space_needs_a_real_majority() -> None:
    # C = 2 -> threshold 0.5, so the binary case degenerates to the majority rule.
    assert multilabel_targets({"yes": 0.5, "no": 0.5}) == set()
    assert multilabel_targets({"yes": 0.6, "no": 0.4}) == {"yes"}


def test_multilabel_targets_honours_an_explicit_threshold() -> None:
    dist = rater_distribution(["angry", "sad", "sad", "neutral", "angry"], EMOTIONS)
    # 0.5 is the multi-label-learning convention; nothing here reaches it.
    assert multilabel_targets(dist, threshold=0.5) == set()
    assert multilabel_targets(dist, threshold=0.1) == {"neutral", "angry", "sad"}


def test_multilabel_targets_rejects_empty_dist() -> None:
    with pytest.raises(ValueError):
        multilabel_targets({})


# --------------------------------------------------------------------------------------
# macro_f1
# --------------------------------------------------------------------------------------


def test_macro_f1_hand_computed_example() -> None:
    # HAND COMPUTATION over labels (neutral, angry, sad, happy):
    #
    #   item 1: target {angry, sad}   predicted {angry, sad}
    #   item 2: target {neutral}      predicted {neutral, happy}
    #   item 3: target {sad}          predicted {angry}
    #
    #   label     TP  FP  FN   F1 = 2TP / (2TP + FP + FN)
    #   neutral    1   0   0   2/2       = 1.0
    #   angry      1   1   0   2/3       = 0.666666...   (item 3 predicted angry wrongly)
    #   sad        1   0   1   2/3       = 0.666666...   (item 3 missed sad)
    #   happy      0   1   0   0/1       = 0.0           (item 2 hallucinated happy)
    #
    #   macro = (1.0 + 2/3 + 2/3 + 0.0) / 4 = (7/3) / 4 = 7/12 = 0.5833333...
    predicted = [{"angry", "sad"}, {"neutral", "happy"}, {"angry"}]
    targets = [{"angry", "sad"}, {"neutral"}, {"sad"}]
    assert macro_f1(predicted, targets, EMOTIONS) == pytest.approx(7 / 12)


def test_macro_f1_perfect_and_disjoint() -> None:
    targets = [{"angry", "sad"}, {"neutral"}]
    assert macro_f1(targets, targets, EMOTIONS) == pytest.approx(1.0)

    # Every prediction wrong on every active label.
    predicted = [{"happy"}, {"happy"}]
    assert macro_f1(predicted, targets, EMOTIONS) == pytest.approx(0.0)


def test_macro_f1_skips_labels_with_no_support_and_no_predictions() -> None:
    # Adding a never-used label must not move the score, or the number becomes
    # incomparable across item-bank revisions that widen the action space.
    predicted = [{"angry", "sad"}, {"neutral", "happy"}, {"angry"}]
    targets = [{"angry", "sad"}, {"neutral"}, {"sad"}]
    wider = (*EMOTIONS, "fear", "disgust")
    assert macro_f1(predicted, targets, wider) == pytest.approx(7 / 12)

    # ...but the sklearn-compatible convention is available and does move it:
    # (1.0 + 2/3 + 2/3 + 0 + 0 + 0) / 6 = 7/18.
    assert macro_f1(predicted, targets, wider, include_absent_labels=True) == pytest.approx(7 / 18)


def test_macro_f1_empty_prediction_scores_zero_not_one() -> None:
    # A model that answers nothing on a uniformly-split item must not score perfectly.
    predicted: list[set[str]] = [set(), set()]
    targets: list[set[str]] = [set(), set()]
    assert macro_f1(predicted, targets, EMOTIONS) == 0.0

    # And an empty prediction against a real target is pure false-negative.
    assert macro_f1([set()], [{"angry"}], EMOTIONS) == pytest.approx(0.0)


def test_macro_f1_empty_target_makes_every_prediction_a_false_positive() -> None:
    assert macro_f1([{"angry"}], [set()], EMOTIONS) == pytest.approx(0.0)


def test_macro_f1_rejects_mismatched_lengths_and_unknown_labels() -> None:
    with pytest.raises(ValueError, match="predicted has"):
        macro_f1([{"angry"}], [{"angry"}, {"sad"}], EMOTIONS)
    with pytest.raises(ValueError, match="outside the label space"):
        macro_f1([{"elated"}], [{"angry"}], EMOTIONS)
    with pytest.raises(ValueError, match="duplicates"):
        macro_f1([{"angry"}], [{"angry"}], ("angry", "angry"))


# --------------------------------------------------------------------------------------
# kl_divergence
# --------------------------------------------------------------------------------------


def test_kl_divergence_is_zero_for_identical_distributions() -> None:
    dist = {"neutral": 0.2, "angry": 0.4, "sad": 0.4, "happy": 0.0}
    assert kl_divergence(dist, dist) == pytest.approx(0.0, abs=1e-9)


def test_kl_divergence_hand_computed_value_in_nats() -> None:
    # KL(R || P) with R = (0.5, 0.5), P = (0.9, 0.1):
    #   0.5*ln(0.5/0.9) + 0.5*ln(0.5/0.1) = 0.5*(-0.587787) + 0.5*(1.609438) = 0.510826
    reference = {"a": 0.5, "b": 0.5}
    predicted = {"a": 0.9, "b": 0.1}
    assert kl_divergence(predicted, reference) == pytest.approx(0.5108256, abs=1e-5)


def test_kl_divergence_is_asymmetric_direction_matters() -> None:
    # Swapping the roles gives a different number; this is the symmetry-breaking that
    # makes the chosen direction, KL(reference || predicted), a real design decision.
    reference = {"a": 0.5, "b": 0.5}
    predicted = {"a": 0.9, "b": 0.1}
    forward = kl_divergence(predicted, reference)  # KL(R || P) = 0.5108
    reverse = kl_divergence(reference, predicted)  # KL(P || R) = 0.3681
    assert forward == pytest.approx(0.5108256, abs=1e-5)
    assert reverse == pytest.approx(0.3680642, abs=1e-5)
    assert forward > reverse


def test_kl_divergence_punishes_ignoring_a_human_minority() -> None:
    # THE POINT OF THE DIRECTION: humans split 60/40; a model that collapses onto the
    # modal answer is penalised more than one that keeps the minority mass.
    reference = {"proceed": 0.6, "escalate": 0.4}
    collapsed = {"proceed": 0.99, "escalate": 0.01}
    honest = {"proceed": 0.65, "escalate": 0.35}
    assert kl_divergence(collapsed, reference) > kl_divergence(honest, reference)


def test_kl_divergence_zero_in_reference_contributes_nothing() -> None:
    # 0 * log 0 == 0: a label no human chose neither rewards nor punishes the model.
    reference = {"a": 1.0, "b": 0.0}
    assert kl_divergence({"a": 1.0, "b": 0.0}, reference) == pytest.approx(0.0, abs=1e-9)
    # Predicting mass on an unvoted label still costs, via the mass taken from "a".
    assert kl_divergence({"a": 0.5, "b": 0.5}, reference) == pytest.approx(math.log(2), abs=1e-6)


def test_kl_divergence_zero_in_predicted_stays_finite_via_smoothing() -> None:
    # Unsmoothed this is +inf, which would poison any average over items.
    value = kl_divergence({"a": 1.0, "b": 0.0}, {"a": 0.5, "b": 0.5})
    assert math.isfinite(value)
    assert value > 10.0  # large, as it should be -- but reportable


def test_kl_divergence_is_never_negative() -> None:
    reference = {"a": 0.3333333333, "b": 0.3333333333, "c": 0.3333333334}
    predicted = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
    assert kl_divergence(predicted, reference) >= 0.0


def test_kl_divergence_normalises_raw_counts() -> None:
    from_counts = kl_divergence({"a": 9.0, "b": 1.0}, {"a": 5.0, "b": 5.0})
    from_props = kl_divergence({"a": 0.9, "b": 0.1}, {"a": 0.5, "b": 0.5})
    assert from_counts == pytest.approx(from_props)


def test_kl_divergence_rejects_mismatched_label_spaces_and_bad_input() -> None:
    with pytest.raises(ValueError, match="share a label space"):
        kl_divergence({"a": 1.0}, {"a": 0.5, "b": 0.5})
    with pytest.raises(ValueError, match="zero total mass"):
        kl_divergence({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 0.0})
    with pytest.raises(ValueError, match="valid probability mass"):
        kl_divergence({"a": -1.0, "b": 2.0}, {"a": 0.5, "b": 0.5})
    with pytest.raises(ValueError, match="epsilon"):
        kl_divergence({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}, epsilon=0.0)


# --------------------------------------------------------------------------------------
# consensus -- "no agreement" is a category we KEEP
# --------------------------------------------------------------------------------------


def test_consensus_unanimous() -> None:
    dist = rater_distribution(["angry"] * 5, EMOTIONS)
    assert consensus(dist) == ("angry", 1.0)


def test_consensus_unique_plurality_below_half_still_counts() -> None:
    # 2/5 angry vs 1/5 each elsewhere. Demanding >0.5 would discard a usable item.
    dist = rater_distribution(["angry", "angry", "sad", "neutral", "happy"], EMOTIONS)
    label, share = consensus(dist)
    assert label == "angry"
    assert share == pytest.approx(0.4)


def test_consensus_returns_none_when_the_top_is_tied() -> None:
    # THE NO-MAJORITY CASE. We refuse to break the tie -- Chou & Lee resolve it randomly
    # for training, which would be indefensible as a benchmark gold.
    dist = rater_distribution(["angry", "angry", "sad", "sad", "neutral"], EMOTIONS)
    label, share = consensus(dist)
    assert label is None
    assert share == pytest.approx(0.4)  # the share is still reported, not discarded


def test_consensus_returns_none_for_a_maximally_split_distribution() -> None:
    dist = rater_distribution(list(EMOTIONS), EMOTIONS)
    label, share = consensus(dist)
    assert label is None
    assert share == pytest.approx(0.25)


def test_consensus_rejects_empty_dist() -> None:
    with pytest.raises(ValueError):
        consensus({})


# --------------------------------------------------------------------------------------
# is_contested -- Core / Contested routing
# --------------------------------------------------------------------------------------


def test_is_contested_core_when_consensus_is_strong() -> None:
    dist = rater_distribution(["angry", "angry", "angry", "angry", "sad"], EMOTIONS)
    assert is_contested(dist) is False


def test_is_contested_boundary_exactly_at_threshold_is_core() -> None:
    # 3/5 = 0.6 is the natural five-rater core bar and must count as Core (>= not >).
    dist = rater_distribution(["angry", "angry", "angry", "sad", "neutral"], EMOTIONS)
    assert dist["angry"] == pytest.approx(0.6)
    assert is_contested(dist, core_threshold=0.6) is False
    # One rater fewer and it flips.
    weaker = rater_distribution(["angry", "angry", "sad", "sad", "neutral"], EMOTIONS)
    assert is_contested(weaker, core_threshold=0.6) is True


def test_is_contested_true_when_the_mode_is_tied_regardless_of_threshold() -> None:
    dist = {"angry": 0.5, "sad": 0.5, "neutral": 0.0, "happy": 0.0}
    assert is_contested(dist, core_threshold=0.5) is True


def test_is_contested_unanimous_is_core_at_any_threshold() -> None:
    dist = rater_distribution(["angry"] * 5, EMOTIONS)
    assert is_contested(dist, core_threshold=1.0) is False


def test_is_contested_rejects_a_nonsense_threshold() -> None:
    dist = rater_distribution(["angry"], EMOTIONS)
    with pytest.raises(ValueError, match="core_threshold"):
        is_contested(dist, core_threshold=0.0)
    with pytest.raises(ValueError, match="core_threshold"):
        is_contested(dist, core_threshold=1.5)


# --------------------------------------------------------------------------------------
# krippendorff_alpha
# --------------------------------------------------------------------------------------


def test_krippendorff_alpha_canonical_example() -> None:
    # Krippendorff's canonical worked example: 15 units x 3 coders, missing values,
    # 4 nominal categories. The published answer is alpha = 0.691.
    #
    #   unit :  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
    #   coder A: *  *  *  *  *  3  4  1  2  1  1  3  3  *  3
    #   coder B: 1  *  2  1  3  3  4  3  *  *  *  *  *  *  *
    #   coder C: *  *  2  1  3  4  4  *  2  1  1  3  3  *  4
    #
    # Pairable values n = 26 (units 2 and 14 are empty; unit 1 has a single value).
    # Published coincidence matrix and marginals, which this implementation reproduces:
    #   o_11 = 6, o_13 = o_31 = 1, o_22 = 4, o_33 = 7, o_34 = o_43 = 2, o_44 = 3
    #   n_1 = 7, n_2 = 4, n_3 = 10, n_4 = 5, n = 26
    # Over unordered pairs: D_o = 1 + 2 = 3 and
    #   D_e = (4*7 + 10*7 + 5*7 + 10*4 + 5*4 + 5*10)/25 = 243/25 = 9.72
    #   alpha = 1 - 3/9.72 = 0.691
    # This module sums ORDERED off-diagonal pairs, doubling both D_o (6) and D_e (19.44),
    # which leaves the ratio -- and therefore alpha -- unchanged.
    ratings = [
        ["1"],  # unit 1  -- one rating only, not pairable
        [],  # unit 2  -- no ratings at all
        ["2", "2"],  # unit 3
        ["1", "1"],  # unit 4
        ["3", "3"],  # unit 5
        ["3", "3", "4"],  # unit 6
        ["4", "4", "4"],  # unit 7
        ["1", "3"],  # unit 8
        ["2", "2"],  # unit 9
        ["1", "1"],  # unit 10
        ["1", "1"],  # unit 11
        ["3", "3"],  # unit 12
        ["3", "3"],  # unit 13
        [],  # unit 14 -- no ratings at all
        ["3", "4"],  # unit 15
    ]
    assert krippendorff_alpha(ratings) == pytest.approx(0.691, abs=0.001)


def test_krippendorff_alpha_perfect_agreement_is_one() -> None:
    ratings = [["angry", "angry", "angry"], ["sad", "sad", "sad"], ["neutral", "neutral"]]
    assert krippendorff_alpha(ratings) == pytest.approx(1.0)


def test_krippendorff_alpha_systematic_disagreement_goes_negative() -> None:
    # Every unit splits perfectly between the two categories: raters disagree MORE than
    # chance would predict, so alpha is negative. Hand computation, 4 units of [a, b]:
    #   o_ab = o_ba = 4, o_aa = o_bb = 0; n_a = n_b = 4, n = 8
    #   D_o = 8; D_e = (4*4 + 4*4)/7 = 32/7; alpha = 1 - 8/(32/7) = 1 - 1.75 = -0.75
    # Alpha is not bounded below by 0 -- a negative value is a real signal that the
    # rating protocol is broken, and must not be clamped away.
    ratings = [["a", "b"], ["b", "a"], ["a", "b"], ["b", "a"]]
    assert krippendorff_alpha(ratings) == pytest.approx(-0.75, abs=1e-9)


def test_krippendorff_alpha_ignores_rater_order_within_a_unit() -> None:
    # Nominal alpha depends only on per-unit category COUNTS, never on which coder said
    # what -- which is exactly why it cannot detect systematic per-rater bias.
    a = krippendorff_alpha([["x", "y", "x"], ["y", "y", "x"]])
    b = krippendorff_alpha([["x", "x", "y"], ["x", "y", "y"]])
    assert a == pytest.approx(b)


def test_krippendorff_alpha_single_category_sample_returns_one() -> None:
    # Degenerate: no disagreement is even possible, so there is nothing to normalise by.
    assert krippendorff_alpha([["angry", "angry"], ["angry", "angry"]]) == pytest.approx(1.0)


def test_krippendorff_alpha_requires_a_pairable_unit() -> None:
    with pytest.raises(ValueError, match="at least two raters"):
        krippendorff_alpha([["angry"], [], ["sad"]])


# --------------------------------------------------------------------------------------
# End-to-end: the Core / Contested routing this module exists to support
# --------------------------------------------------------------------------------------


def test_core_and_contested_items_route_differently() -> None:
    core = rater_distribution(["angry", "angry", "angry", "angry", "sad"], EMOTIONS)
    contested = rater_distribution(["angry", "angry", "sad", "sad", "neutral"], EMOTIONS)

    assert is_contested(core) is False
    assert consensus(core)[0] == "angry"  # scored against a single gold

    assert is_contested(contested) is True
    assert consensus(contested)[0] is None  # no gold exists to score against
    assert multilabel_targets(contested) == {"angry", "sad"}  # ...so score the set
    # A model that answers only the plurality label gets partial, not full, credit.
    partial = macro_f1([{"angry"}], [multilabel_targets(contested)], EMOTIONS)
    full = macro_f1([{"angry", "sad"}], [multilabel_targets(contested)], EMOTIONS)
    assert 0.0 < partial < full == pytest.approx(1.0)
