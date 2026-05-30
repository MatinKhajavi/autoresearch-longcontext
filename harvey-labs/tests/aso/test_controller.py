from pytest import approx

from aso.controller import build_jobs, mean_by_variant, successive_halving
from aso.scaffold import Scaffold


def _scaf(tag: str) -> Scaffold:
    return Scaffold(system_prompt=f"PROMPT-{tag}", skills={}, module_config={})


def test_build_jobs_is_variant_cross_task():
    variants = {"a": _scaf("a"), "b": _scaf("b")}
    jobs = build_jobs(variants, ["t1", "t2"], model="m", judge_model="j", max_turns=50)
    assert len(jobs) == 4
    assert {(j["variant_id"], j["task"]) for j in jobs} == {
        ("a", "t1"), ("a", "t2"), ("b", "t1"), ("b", "t2")
    }
    # scaffold serialized for the Modal boundary
    assert jobs[0]["scaffold"]["system_prompt"].startswith("PROMPT-")
    assert jobs[0]["max_turns"] == 50


def test_build_jobs_replicates_by_seeds():
    variants = {"a": _scaf("a")}
    jobs = build_jobs(variants, ["t1", "t2"], model="m", judge_model="j", max_turns=50, seeds=3)
    assert len(jobs) == 6                       # 1 variant × 2 tasks × 3 seeds
    assert sorted(j["seed"] for j in jobs if j["task"] == "t1") == [0, 1, 2]


def test_mean_by_variant_averages_across_tasks():
    results = [
        {"variant_id": "a", "pass_rate": 0.2},
        {"variant_id": "a", "pass_rate": 0.4},
        {"variant_id": "b", "pass_rate": 1.0},
    ]
    assert mean_by_variant(results) == approx({"a": 0.3, "b": 1.0})


def test_successive_halving_prunes_then_promotes():
    variants = {"good": _scaf("good"), "bad1": _scaf("bad1"), "bad2": _scaf("bad2")}
    calls = {"screen_variants": None, "dev_variants": None}

    def fake_eval(jobs):
        # record which variants reached each stage by task-set size
        tasks = {j["task"] for j in jobs}
        stage = "screen" if tasks == {"s1"} else "dev"
        calls[f"{stage}_variants"] = {j["variant_id"] for j in jobs}
        return [
            {"variant_id": j["variant_id"], "task": j["task"], "status": "ok",
             "pass_rate": 0.9 if j["variant_id"] == "good" else 0.1}
            for j in jobs
        ]

    champ, table = successive_halving(
        variants, screen=["s1"], dev=["d1", "d2"], eval_fn=fake_eval,
        keep_m=1, model="m", judge_model="j",
    )
    assert champ == "good"
    assert calls["screen_variants"] == {"good", "bad1", "bad2"}   # all screened
    assert calls["dev_variants"] == {"good"}                       # only survivor promoted
    assert table["survivors"] == ["good"]
    assert table["dev_means"]["good"] == 0.9


def test_successive_halving_keep_m_two():
    variants = {"a": _scaf("a"), "b": _scaf("b"), "c": _scaf("c")}
    scores = {"a": 0.8, "b": 0.5, "c": 0.1}

    def fake_eval(jobs):
        return [{"variant_id": j["variant_id"], "task": j["task"], "status": "ok",
                 "pass_rate": scores[j["variant_id"]]} for j in jobs]

    champ, table = successive_halving(
        variants, screen=["s1"], dev=["d1"], eval_fn=fake_eval,
        keep_m=2, model="m", judge_model="j",
    )
    assert set(table["survivors"]) == {"a", "b"}   # c pruned
    assert champ == "a"
