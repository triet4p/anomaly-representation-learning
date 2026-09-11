"""Focused behavioral tests for the Sprint 16 shared diagnostic contract.

Covers plausible metric bugs only: tie handling vs the sklearn convention,
identical-support enforcement, bootstrap determinism, the pooled-scale floor,
score-family separation, frozen-standardization leakage, and the §10 verdict
table. Synthetic inputs only; no data roots, no checkpoints.
"""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

import representation.attribution_metrics as am
from representation.attribution_metrics import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    DiagnosticSet,
    FrozenStandardizer,
    aggregate_patches,
    assert_same_provenance,
    bootstrap_ci,
    component_verdict,
    exceedance_fraction,
    intersect_ids,
    median_lead,
    movement,
    overall_state,
    recall_at_threshold,
    restoration_met,
    severity_spearman,
    stability_ci,
    stability_ratio,
    tie_auc,
)

def test_tie_auc_matches_sklearn_with_ties():
    scores = np.array([0.1, 0.4, 0.4, 0.9, 0.2, 0.7])
    labels = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    assert tie_auc(scores, labels) == roc_auc_score(labels, scores)
    rng = np.random.default_rng(0)
    s = np.round(rng.normal(size=200), 1)
    y = (rng.random(200) < 0.4).astype(float)
    assert tie_auc(s, y) == roc_auc_score(y, s)


def test_tie_auc_edges():
    assert tie_auc([3.0, 2.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]) == 1.0
    assert tie_auc([0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 0.0, 0.0]) == 0.0
    assert tie_auc([0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 1.0, 0.0]) == 0.5
    assert np.isnan(tie_auc([1.0, 2.0], [0.0, 0.0]))
    assert np.isnan(tie_auc([1.0, 2.0], [1.0, 1.0]))
    with pytest.raises(ValueError):
        tie_auc([np.inf, 0.0], [1.0, 0.0])


def test_support_intersection_fails_closed():
    assert intersect_ids(["a", "b"], ["b", "c"]).tolist() == ["b"]
    with pytest.raises(ValueError):
        intersect_ids(["a"], ["b"])
    with pytest.raises(ValueError):
        intersect_ids([], ["a"])


def test_diagnostic_set_rejects_bad_support():
    kw = dict(families={"S_pred": [1.0, 2.0]}, labels=[1.0, 0.0])
    with pytest.raises(ValueError):
        DiagnosticSet(history_id="h", ids=["a", "a"], **kw)
    with pytest.raises(ValueError):
        DiagnosticSet(history_id="h", ids=["a", "b"],
                      families={"S_pred": [1.0]}, labels=[1.0, 0.0])
    with pytest.raises(ValueError):
        DiagnosticSet(history_id="h", ids=["a", "b"],
                      families={"S_pred": [np.nan, 1.0]}, labels=[1.0, 0.0])
    ds = DiagnosticSet(history_id="h", ids=["a", "b"], **kw)
    with pytest.raises(KeyError):
        ds.family("S_pop")
    with pytest.raises(ValueError):
        ds.restrict(["a", "zzz"])
    with pytest.raises(ValueError):
        ds.restrict([])
    sub = ds.restrict(["b", "a"])
    assert sub.ids.tolist() == ["b", "a"]
    assert sub.family("S_pred").tolist() == [2.0, 1.0]


def test_standardizer_rejects_non_2d_input():
    st = FrozenStandardizer.fit(np.zeros((4, 2)), source="h")
    with pytest.raises(ValueError):
        FrozenStandardizer.fit(np.zeros(4), source="h")
    with pytest.raises(ValueError):
        st.apply(np.zeros(4))


def test_stability_ci_resamples_A_with_B_fixed():
    rng = np.random.default_rng(7)
    a = rng.normal(0.5, 1.0, size=40)
    b = rng.normal(0.0, 1.0, size=200)
    out = stability_ci(a, b)
    assert set(out) == {"ratio", "lcb", "ucb", "pooled_scale", "cap",
                        "replicates", "seed"}
    assert out["lcb"] <= out["ratio"] <= out["ucb"]
    assert out["ratio"] == stability_ratio(a, b)["ratio"]
    assert out == stability_ci(a, b)
    inner = np.random.default_rng(BOOTSTRAP_SEED)
    draws = inner.integers(0, a.size, size=(BOOTSTRAP_REPLICATES, a.size))
    expect = sorted(stability_ratio(a[row], b)["ratio"] for row in draws)
    assert out["lcb"] == pytest.approx(np.quantile(expect, 0.025))
    assert out["ucb"] == pytest.approx(np.quantile(expect, 0.975))
    with pytest.raises(ValueError):
        stability_ci([], b)


def test_movement_reports_all_tied_signs():
    m = movement([0.08, -0.08, 0.01])
    assert m["magnitude"] == pytest.approx(0.08)
    assert sorted(m["directions"]) == [-1.0, 1.0]
    assert m["direction"] in (1.0, -1.0)


def test_provenance_guard_fails_closed():
    s1 = FrozenStandardizer.fit(np.zeros((6, 2)), source="H-FIT-28")
    s2 = FrozenStandardizer.fit(np.zeros((6, 2)), source="H-FIT-28")
    s3 = FrozenStandardizer.fit(np.ones((6, 2)), source="H-FIT-28")
    assert s1.token() == s2.token()
    assert s1.token() != s3.token()
    mk = lambda tok: DiagnosticSet(
        history_id="h", ids=["a", "b"],
        families={"S_pred": [1.0, 2.0]}, labels=[1.0, 0.0],
        provenance=tok)
    assert assert_same_provenance(mk(s1.token()), mk(s2.token())) == s1.token()
    with pytest.raises(ValueError):
        assert_same_provenance(mk(s1.token()), mk(s3.token()))
    with pytest.raises(ValueError):
        assert_same_provenance(mk(s1.token()), mk(None))
    with pytest.raises(ValueError):
        assert_same_provenance()
    assert mk(s1.token()).restrict(["a"]).provenance == s1.token()


def test_overall_state_counts_shared_restoration_once():
    both = {"C8": "BOTTLENECK", "C9": "BOTTLENECK"}
    assert overall_state(both) == "MULTIPLE BOTTLENECKS"
    assert overall_state(both, independent_groups={"C8": "tail", "C9": "tail"}) == "IDENTIFIED"
    assert overall_state(
        {"C1": "BOTTLENECK", "C8": "BOTTLENECK", "C9": "BOTTLENECK"},
        independent_groups={"C8": "tail", "C9": "tail"},
    ) == "MULTIPLE BOTTLENECKS"
    with pytest.raises(ValueError):
        overall_state(both, independent_groups={"C99": "tail"})


def test_restrict_set_matches_sorted_list():
    ds = DiagnosticSet(history_id="h", ids=["a", "b", "c", "d"],
                       families={"S_pred": [1.0, 2.0, 3.0, 4.0]},
                       labels=[1.0, 0.0, 1.0, 0.0])
    assert ds.restrict({"a", "b", "c", "d"}).ids.tolist() == ["a", "b", "c", "d"]
    assert ds.restrict(["d", "a"]).ids.tolist() == ["d", "a"]


def test_restoration_arithmetic_and_recall_labels():
    assert restoration_met(0.70, 0.55, 0.20, 0.60) is True
    assert restoration_met(0.60, 0.55, 0.20, 0.60) is False
    assert restoration_met(0.70, 0.55, 0.20, 0.50) is False
    with pytest.raises(ValueError):
        restoration_met(np.nan, 0.5, 0.2, 0.6)
    with pytest.raises(ValueError):
        recall_at_threshold([1.0, 2.0], [0.0, 2.0], 0.5)


def test_bootstrap_deterministic_and_sane():
    rng = np.random.default_rng(3)
    v = rng.normal(size=60)
    first = bootstrap_ci(v)
    second = bootstrap_ci(v)
    assert first == second
    assert first["lcb"] <= first["point"] <= first["ucb"]
    other = bootstrap_ci(v, seed=am.BOOTSTRAP_SEED + 1)
    assert other["lcb"] != first["lcb"] or other["ucb"] != first["ucb"]


def test_pooled_scale_floor_keeps_ratio_finite():
    out = stability_ratio([5.0, 5.0, 5.0, 5.0], [5.0, 5.0, 5.0, 5.0])
    assert out["pooled_scale"] == am.STD_FLOOR
    assert np.isfinite(out["ratio"]) and out["ratio"] == 0.0
    out = stability_ratio([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0])
    assert out["pooled_scale"] > am.STD_FLOOR


def test_score_families_never_fused():
    assert not hasattr(am, "fuse")
    assert not hasattr(am, "fuse_scores")
    ds = DiagnosticSet(
        history_id="h",
        ids=["a", "b", "c", "d"],
        families={"S_pred": [4.0, 3.0, 2.0, 1.0],
                  "S_pop": [1.0, 2.0, 3.0, 4.0]},
        labels=[1.0, 1.0, 0.0, 0.0],
    )
    assert tie_auc(ds.family("S_pred"), ds.labels) == 1.0
    assert tie_auc(ds.family("S_pop"), ds.labels) == 0.0


def test_standardizer_frozen_and_healthy_only():
    healthy = np.array([[0.0, 10.0], [2.0, 12.0]])
    st = FrozenStandardizer.fit(healthy, source="H-FIT-28")
    eval_rows = np.array([[100.0, 100.0]])
    before = st.apply(healthy).copy()
    st.apply(eval_rows)
    assert np.array_equal(st.apply(healthy), before)
    assert st.provenance()["n_fit"] == 2
    assert st.provenance()["source"] == "H-FIT-28"
    with pytest.raises(ValueError):
        st.refit(healthy, source="other")
    with pytest.raises(ValueError):
        FrozenStandardizer.fit(np.zeros((0, 2)), source="x")
    with pytest.raises(ValueError):
        st.apply(np.zeros((1, 5)))


def test_severity_constant_returns_nan_not_crash():
    assert np.isnan(severity_spearman([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]))
    assert np.isnan(severity_spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
    assert severity_spearman([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]) == 1.0


def test_aggregation_requires_explicit_params_and_support():
    s = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    v = np.ones(8, dtype=bool)
    assert aggregate_patches(s, v, "max") == 8.0
    assert aggregate_patches(s, v, "median") == np.percentile(s, 50.0, method="linear")
    assert aggregate_patches(s, v, "topk_mean", k=4) == 6.5
    with pytest.raises(ValueError):
        aggregate_patches(s, v, "topk_mean")
    with pytest.raises(ValueError):
        aggregate_patches(s, v, "percentile")
    with pytest.raises(ValueError):
        aggregate_patches(s, v, "nope")
    short = np.ones(4, dtype=bool)
    with pytest.raises(ValueError):
        aggregate_patches(s[:4], short, "max")


def test_operating_helpers():
    assert recall_at_threshold([1.0, 2.0, 3.0], [0.0, 1.0, 1.0], 1.5) == 1.0
    assert np.isnan(recall_at_threshold([1.0], [0.0], 0.5))
    assert exceedance_fraction([1.0, 2.0, 3.0, 4.0], 2.5) == 0.5
    assert median_lead([3.0, 1.0, 5.0], [True, True, False]) == 2.0
    assert np.isnan(median_lead([1.0], [False]))


def test_movement_reports_magnitude_and_direction():
    m = movement([0.01, -0.08, 0.03])
    assert m["magnitude"] == pytest.approx(0.08)
    assert m["direction"] == -1.0
    assert m["n"] == 3.0


def test_verdict_table_rows():
    base = dict(restoration=True, nuisance_ok=True, replication_ok=True,
                encoders_ok=True)
    assert component_verdict(g1_fail=True, upstream_lcb=0.9, gap=0.2,
                             move_magnitude=0.2, **base) == "UNRESOLVED"
    assert component_verdict(g1_fail=False, upstream_lcb=0.5, gap=0.2,
                             move_magnitude=0.2, **base) == "UNRESOLVED"
    assert component_verdict(g1_fail=False, upstream_lcb=0.9, gap=0.05,
                             move_magnitude=0.02, **base) == "HEALTHY"
    assert component_verdict(g1_fail=False, upstream_lcb=0.9, gap=0.05,
                             move_magnitude=0.08, **base) == "SUSPECT"
    assert component_verdict(g1_fail=False, upstream_lcb=0.9, gap=0.15,
                             move_magnitude=0.15, **base) == "BOTTLENECK"
    fail = dict(base, nuisance_ok=False)
    assert component_verdict(g1_fail=False, upstream_lcb=0.9, gap=0.15,
                             move_magnitude=0.15, **fail) == "SUSPECT"
    with pytest.raises(ValueError):
        component_verdict(g1_fail=False, upstream_lcb=np.nan, gap=0.1,
                          move_magnitude=0.1, **base)


def test_overall_state():
    assert overall_state({"C1": "HEALTHY"}) == "UNRESOLVED"
    assert overall_state({"C1": "BOTTLENECK", "C2": "SUSPECT"}) == "IDENTIFIED"
    assert overall_state({"C1": "BOTTLENECK", "C2": "BOTTLENECK"}) == "MULTIPLE BOTTLENECKS"
    with pytest.raises(ValueError):
        overall_state({})
    with pytest.raises(ValueError):
        overall_state({"C1": "MAYBE"})
