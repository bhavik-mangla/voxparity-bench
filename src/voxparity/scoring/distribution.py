"""Scoring against a rater DISTRIBUTION instead of a single authored gold label.

WHY THIS MODULE EXISTS
----------------------
VoxParity scores perception against one authored gold label. The affective-computing
literature shows that this throws away real signal rather than noise:

  * Chou & Lee 2025 (arXiv:2510.05934), Table 1.1: on secondary/blended emotion labels
    the majority rule discards up to 92.01% of utterances and 96.99% of the individual
    ratings. Plurality is better but still discards up to 33.72% / 78.13%.
  * The same work's "all-inclusive rule" (AR) keeps every annotated sample regardless of
    vote frequency and represents the ground truth as the vote distribution.

So: an item where humans split is not a broken item. It is an item whose answer is a
distribution. This module implements the Core/Contested dual track (D030) -- Core items
keep single-gold scoring, Contested items are scored against the human distribution.

METHOD PROVENANCE
-----------------
Verified verbatim against arXiv:2510.05934 (Chou & Lee, "Revisiting Modeling and
Evaluation Approaches in Speech Emotion Recognition: Considering Subjectivity of
Annotators and Ambiguity of Emotions"), Section 4.4.5 "Evaluation Metrics":

  * Hard-decision assessment, 1/C binarisation:
      "Targets are selected based on using the threshold to the binarized vectors. A
       prediction is valid if the proportion for a specific category exceeds 1/C, with C
       representing how many emotion classes."
    Worked example in the paper: 4 classes, five raters give (A, S, S, N, A), so the
    distribution over (N, A, S, H) is (0.2, 0.4, 0.4, 0.0); threshold 1/4 = 0.25; the
    binarised ground truth is (0, 1, 1, 0). `multilabel_targets` reproduces exactly this.

  * Distribution assessment, KLD, and its lineage:
      "Following the approach proposed by Steidl et al. [22], where results are assessed
       using an entropy-based metric, we employ the Kullback-Leibler divergence (KLD) to
       determine the similarity between the model's predicted distribution and the
       subjective annotations."
    Steidl et al. is the soft-label / entropy-based evaluation lineage; Ando et al. is the
    soft-label-modification line ([2] and [63] in that bibliography). Both are cited by
    Chou as the origin of distribution-valued ground truth.

  * KLD direction: the paper's loss (Chapter 5) is written
        sum_j sum_z  P_jz * Y^T_ij * log( Y^T_ij / Y^P_ij )
    i.e. the HUMAN target Y^T is the outer measure and the model prediction Y^P is the
    denominator. That is KL(reference || predicted). `kl_divergence` follows it.

  * KLD's stated limitation, quoted so nobody reports it as a headline number:
      "It lacks a fixed range, which makes it challenging to interpret and compare
       absolute values across different datasets or models."

NOT VERIFIED -- see the module report. The CREMA-D "15.2% of clips have no unique
majority at 5 raters", the 1-rater-to-9-raters 39.3% -> 41.5% actor-intent agreement
figure, and the Interspeech-2025 dominance CCC 0.5735 -> 0.6034 improvement from
retaining a "no agreement" class are motivating claims taken on trust from the task
brief; they are NOT in arXiv:2510.05934 and were not independently checked. The design
choice they motivate (keep "no agreement" as its own outcome instead of dropping the
item) is implemented in `consensus`, which returns an explicit None rather than raising
or silently picking a winner.

Stdlib only. No numpy, no scipy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

__all__ = [
    "consensus",
    "is_contested",
    "kl_divergence",
    "krippendorff_alpha",
    "macro_f1",
    "multilabel_targets",
    "rater_distribution",
]

# Boundary tolerance for the 1/C comparison. Vote proportions are ratios of small
# integers, so a share that is mathematically exactly 1/C can land a few ULPs either
# side of it (e.g. 2/6 vs 1/3). We want that case to be decided by the DEFINITION
# ("exceeds 1/C" -> exactly-at-threshold does not qualify), not by float noise.
_BOUNDARY_TOL = 1e-9


def rater_distribution(answers: list[str], labels: Sequence[str]) -> dict[str, float]:
    """Vote proportions over a fixed label space.

    Returns a dict with one entry per label in `labels`, in that order, whose values sum
    to 1.0. Labels that nobody chose are present with 0.0 -- WHY: every downstream
    function needs C (the size of the label space) to be readable off the distribution
    itself, and dropping zero-vote labels would silently change the 1/C threshold.

    Raises on an answer outside `labels` rather than dropping it: a typo'd or
    out-of-schema rater answer must not quietly shrink the denominator and inflate
    everyone else's share.
    """
    if not labels:
        raise ValueError("labels must be non-empty")
    if len(set(labels)) != len(labels):
        raise ValueError(f"labels contains duplicates: {labels!r}")
    if not answers:
        raise ValueError("answers must be non-empty; a distribution over zero votes is undefined")

    label_set = set(labels)
    counts: dict[str, int] = {label: 0 for label in labels}
    for answer in answers:
        if answer not in label_set:
            raise ValueError(f"answer {answer!r} is not in the label space {tuple(labels)!r}")
        counts[answer] += 1

    total = len(answers)
    return {label: counts[label] / total for label in labels}


def multilabel_targets(dist: Mapping[str, float], threshold: float | None = None) -> set[str]:
    """The 1/C binarisation of a rater distribution (Chou & Lee 2025, Sec. 4.4.5).

    A label is a target when its vote proportion EXCEEDS the threshold, which defaults to
    1/C where C = len(dist). Strictly exceeds, per the paper's wording; a share sitting
    exactly on 1/C is not a target.

    WHY 1/C and not 0.5: 0.5 asks "did a majority say this", which is the very question
    that discards 92% of blended-emotion utterances. 1/C asks "did this label do better
    than chance-uniform", which admits genuine secondary emotions -- a label two of five
    raters chose survives, and co-occurring answers ("sad AND angry") are representable.

    EDGE CASE worth knowing: a perfectly uniform distribution (every label at exactly
    1/C) produces an EMPTY target set, because nothing exceeds the threshold. That is
    correct behaviour, not a bug -- a maximally split item carries no multi-label target.
    Route such items with `consensus`/`is_contested` and score them with `kl_divergence`,
    which is defined for them, rather than with `macro_f1`, which is not informative.

    `threshold` may be overridden (e.g. 0.5 for the multi-label-learning convention, or
    1/(C-1) as used by some facial-expression work Chou cites), but 1/C is the default
    because it is what the all-inclusive rule's reported results use.
    """
    if not dist:
        raise ValueError("dist must be non-empty")
    if threshold is None:
        # C is the size of the label space, which is why rater_distribution keeps zeros.
        threshold = 1.0 / len(dist)

    targets: set[str] = set()
    for label, share in dist.items():
        if share > threshold and not math.isclose(
            share, threshold, rel_tol=_BOUNDARY_TOL, abs_tol=1e-12
        ):
            targets.add(label)
    return targets


def macro_f1(
    predicted: Sequence[set[str]],
    targets: Sequence[set[str]],
    labels: Sequence[str],
    *,
    include_absent_labels: bool = False,
) -> float:
    """Macro-averaged F1 over the label space, for multi-label (1/C-binarised) scoring.

    `predicted[i]` and `targets[i]` are both SETS of labels for item i, as produced by
    `multilabel_targets`. Per label, F1 = 2*TP / (2*TP + FP + FN); the macro average is
    the unweighted mean over labels.

    Two divide-by-zero paths, both handled explicitly:

    * A label with no support AND no predictions (TP = FP = FN = 0) has an undefined F1.
      By default it is EXCLUDED from the average. WHY: including it as 0.0 (sklearn's
      zero_division=0 default) means simply widening the label space -- adding a rare
      emotion nobody ever uses -- mechanically lowers every model's score, which would
      make VoxParity numbers incomparable across item-bank revisions. Pass
      include_absent_labels=True to get the sklearn-compatible number instead.
    * A label predicted but never a target (FP > 0, TP = 0) scores 0.0 and DOES count.
      That is a real error and must be penalised.

    If no label is active at all -- every predicted and every target set is empty -- the
    result is 0.0, not 1.0. WHY: on uniformly-split items `multilabel_targets` yields the
    empty set, and a model that predicts nothing would otherwise score a perfect 1.0 for
    saying nothing. Refusing to answer is not perception.
    """
    if len(predicted) != len(targets):
        raise ValueError(f"predicted has {len(predicted)} items but targets has {len(targets)}")
    if not labels:
        raise ValueError("labels must be non-empty")
    if len(set(labels)) != len(labels):
        raise ValueError(f"labels contains duplicates: {tuple(labels)!r}")

    label_set = set(labels)
    for i, (pred, tgt) in enumerate(zip(predicted, targets, strict=True)):
        unknown = (pred | tgt) - label_set
        if unknown:
            raise ValueError(f"item {i} uses labels outside the label space: {sorted(unknown)!r}")

    scores: list[float] = []
    for label in labels:
        tp = fp = fn = 0
        for pred, tgt in zip(predicted, targets, strict=True):
            in_pred = label in pred
            in_tgt = label in tgt
            if in_pred and in_tgt:
                tp += 1
            elif in_pred:
                fp += 1
            elif in_tgt:
                fn += 1

        denominator = 2 * tp + fp + fn
        if denominator == 0:
            # Label never appears in gold or prediction: undefined, not zero.
            if include_absent_labels:
                scores.append(0.0)
            continue
        scores.append(2 * tp / denominator)

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def kl_divergence(
    predicted: Mapping[str, float],
    reference: Mapping[str, float],
    epsilon: float = 1e-12,
) -> float:
    """KL(reference || predicted) in NATS, with smoothing on the predicted distribution.

        KL(R || P) = sum_c  R_c * log( R_c / P_c )

    DIRECTION, and why this one. The human rater distribution is the reference (the outer
    expectation); the model's distribution is the denominator. This is the direction Chou
    & Lee use as their distribution-label loss (target * log(target / predicted)) and it
    is the one with the right failure mode for a perception benchmark: it is
    zero-avoiding, so a model is heavily penalised for putting near-zero mass on an answer
    that a real minority of humans gave. The reverse direction, KL(predicted ||
    reference), is zero-forcing -- it would let a confident model collapse onto the modal
    human answer and ignore the minority, which is precisely the failure this whole
    module exists to stop measuring away.

    RANGE. There is none. Quoting the paper's own stated limitation: KLD "lacks a fixed
    range, which makes it challenging to interpret and compare absolute values across
    different datasets or models." It is also sensitive to small distribution
    perturbations on sparse data. Report it alongside macro-F1, only ever compare KLD
    values computed on the SAME item set, and never put a bare KLD on a leaderboard.
    Lower is better.

    SMOOTHING. Only `predicted` is smoothed -- P_c := (P_c + eps) / (1 + C*eps) -- because
    only the denominator can produce an infinity. Terms where R_c == 0 contribute exactly
    0.0, the standard 0*log(0) = 0 convention, so a label no human chose neither rewards
    nor punishes the model here. Smoothing the reference too would give every zero-vote
    label a tiny spurious contribution and make the score depend on the size of the label
    space rather than on the ratings.
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive; it is what keeps KLD finite")
    if set(predicted) != set(reference):
        missing = set(reference) - set(predicted)
        extra = set(predicted) - set(reference)
        raise ValueError(
            f"predicted and reference must share a label space "
            f"(missing from predicted: {sorted(missing)!r}, extra: {sorted(extra)!r})"
        )

    ref = _as_distribution(reference, "reference")
    pred = _as_distribution(predicted, "predicted")

    n_labels = len(pred)
    denom = 1.0 + n_labels * epsilon

    total = 0.0
    for label, r in ref.items():
        if r == 0.0:
            continue  # 0 * log 0 == 0
        p = (pred[label] + epsilon) / denom
        total += r * math.log(r / p)

    # KL is non-negative in exact arithmetic; smoothing plus float error can produce a
    # value a hair below zero on near-identical distributions. Clamp rather than emit -0.0
    # or a tiny negative, which would look like a bug in a report.
    return max(total, 0.0)


def consensus(dist: Mapping[str, float]) -> tuple[str | None, float]:
    """The modal label and its share; (None, top_share) when there is no unique mode.

    "No agreement" is a CATEGORY WE KEEP, not an item we throw away -- that is the whole
    point of the Core/Contested track. The caller gets an explicit None plus the share
    the tied top labels each hold, so a tie is recordable and reportable rather than
    silently resolved. (Chou & Lee's own AR training-set construction resolves such ties
    by picking a top-voted class AT RANDOM; that is acceptable for training but would be
    indefensible as a benchmark gold, so we refuse to pick.)

    "Unique majority" here means a UNIQUE MODE (plurality), not a share above 0.5. A
    4-way item where one label holds 0.4 and no other exceeds 0.3 has a perfectly usable
    consensus answer; demanding >0.5 would discard it. The strength of that consensus is
    a separate question, asked by `is_contested`.
    """
    if not dist:
        raise ValueError("dist must be non-empty")

    top_share = max(dist.values())
    winners = [label for label, share in dist.items() if share == top_share]
    if len(winners) != 1:
        return None, top_share
    return winners[0], top_share


def is_contested(dist: Mapping[str, float], core_threshold: float = 0.6) -> bool:
    """Route an item to the Core track (False) or the Contested track (True).

    Core requires BOTH a unique modal label AND that label holding at least
    `core_threshold` of the vote. Contested otherwise.

    Boundary: a share exactly equal to `core_threshold` is Core (>= not >). WHY: with
    five raters the natural core bar is 3/5 = 0.6, and that must count as core.

    Core items keep single-gold scoring against the modal label. Contested items are
    scored against the distribution -- `multilabel_targets` + `macro_f1` for the hard
    decision, `kl_divergence` for the distribution -- so D029's rater disagreements
    become a measurement instead of a discard.
    """
    if not 0.0 < core_threshold <= 1.0:
        raise ValueError(f"core_threshold must be in (0, 1], got {core_threshold!r}")

    label, share = consensus(dist)
    if label is None:
        return True
    return share < core_threshold


def krippendorff_alpha(ratings: Sequence[Sequence[str]]) -> float:
    """Krippendorff's alpha for NOMINAL data, from the coincidence-matrix definition.

    `ratings[u]` is the list of answers given for unit (item) u -- exactly the shape
    `rater_distribution` consumes. Units rated fewer than twice cannot contribute a
    pairable value and are skipped, per the standard definition. Raters need not be
    identified and need not have rated every unit; unlike Cohen's kappa, alpha handles
    ragged / incomplete designs natively.

    WHY CODER IDENTITY IS ABSENT. For the NOMINAL metric the coincidence matrix depends
    only on the per-unit category counts n_uc and the unit's number of ratings m_u, never
    on which coder produced which value, so the reliability matrix collapses to this
    shape without loss. The consequence, which matters when reading the number: nominal
    alpha therefore cannot detect systematic per-rater bias (a rater who consistently
    over-reports "urgent" shows up only as generic disagreement). If VoxParity needs
    per-rater bias, that is a separate analysis, not this statistic.

    Definition implemented (Krippendorff, "Computing Krippendorff's Alpha-Reliability"):

        o_ck = sum_u ( n_uc * n_uk - delta_ck * n_uc ) / (m_u - 1)
        n_c  = sum_k o_ck                    (marginal)
        n    = sum_c n_c                     (total pairable values)
        e_ck = ( n_c * n_k - delta_ck * n_c ) / (n - 1)
        D_o  = sum_{c != k} o_ck             (nominal metric: delta^2 = 1 iff c != k)
        D_e  = sum_{c != k} e_ck
        alpha = 1 - D_o / D_e

    Verified against Krippendorff's own canonical worked example (15 units, 3 observers,
    4 categories, missing values), which the literature reports as alpha = 0.691; see
    test_krippendorff_alpha_canonical_example. Returns 1.0 when there is no expected
    disagreement to normalise by (D_e == 0), i.e. every pairable value is the same
    category -- perfect agreement, though on a degenerate single-category sample.
    """
    # Per-unit category counts; units with < 2 ratings contribute nothing pairable.
    per_unit: list[dict[str, int]] = []
    for unit in ratings:
        values = list(unit)
        if len(values) < 2:
            continue
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        per_unit.append(counts)

    if not per_unit:
        raise ValueError("need at least one unit rated by at least two raters")

    categories = sorted({category for counts in per_unit for category in counts})

    # Coincidence matrix. Each unit contributes its pairs, weighted 1/(m_u - 1) so that a
    # heavily-rated unit does not dominate a lightly-rated one.
    coincidence: dict[tuple[str, str], float] = {}
    for counts in per_unit:
        m_u = sum(counts.values())
        weight = 1.0 / (m_u - 1)
        for c, n_uc in counts.items():
            for k, n_uk in counts.items():
                pairs = n_uc * n_uk - (n_uc if c == k else 0)
                if pairs:
                    coincidence[(c, k)] = coincidence.get((c, k), 0.0) + pairs * weight

    marginals = {c: sum(coincidence.get((c, k), 0.0) for k in categories) for c in categories}
    n_total = sum(marginals.values())
    if n_total <= 1.0:
        raise ValueError("not enough pairable values to compute alpha")

    observed_disagreement = sum(value for (c, k), value in coincidence.items() if c != k)
    expected_disagreement = sum(
        marginals[c] * marginals[k] for c in categories for k in categories if c != k
    ) / (n_total - 1.0)

    if expected_disagreement == 0.0:
        # Only one category was ever used: no disagreement is even possible, so alpha is
        # undefined by the ratio. Report perfect agreement and let the caller notice the
        # sample is degenerate.
        return 1.0

    return 1.0 - observed_disagreement / expected_disagreement


def _as_distribution(dist: Mapping[str, float], name: str) -> dict[str, float]:
    """Validate and normalise a mapping to a proper probability distribution."""
    values = list(dist.values())
    for label, share in dist.items():
        if share < 0.0 or math.isnan(share) or math.isinf(share):
            raise ValueError(f"{name}[{label!r}] = {share!r} is not a valid probability mass")
    total = math.fsum(values)
    if total <= 0.0:
        raise ValueError(f"{name} has zero total mass")
    if math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        return dict(dist)
    # Accept raw counts / unnormalised weights; renormalising is friendlier than raising
    # and cannot change the KLD (which is scale-invariant only after normalisation).
    return {label: share / total for label, share in dist.items()}
