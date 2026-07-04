#!/usr/bin/env python3
"""Run a targeted strategy-discovery suite for the MoE scheduler.

This is not the broad stratified-v6 evaluation set.  It intentionally focuses
on distributions likely to expose strategies that analytical/fast/C may miss:

  hot-hot + medium-medium + tiny filler

The lower bound saved for every case is the compute-only ideal:

  ceil(sum(ntokens) * 3 * 2048 * 1408 / (2 clusters * 512 MAC/cc))
  = ceil(sum(ntokens) * 8448)

The result file is resumable.  Re-running the same command skips cases whose
beam result is already marked "ok" unless --force is used.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "strategy_discovery_results.json"
DEFAULT_SUMMARY = ROOT / "strategy_discovery_summary.json"

sys.path.insert(0, str(ROOT))

from analytical_scheduler import analytical_schedule
from four_stage_scheduler import FourStageScheduler


IDEAL_CC_PER_TOKEN_EXPERT = 8448
BEAM_MODE = "semantic_pair_split_family_semantic_dedup"

H_VALUES = (16, 24, 32, 48, 64)
BOUNDARY_NTOKS = (
    8,
    9,
    12,
    13,
    15,
    16,
    17,
    23,
    24,
    31,
    32,
    33,
    47,
    48,
    49,
    63,
    64,
    65,
)

CACHE_MODES = (
    "none",
    "hot_pair",
    "medium_pair",
    "hot0",
    "medium0",
)


def compute_only_ideal_cc(total_tokens: int) -> int:
    return math.ceil(total_tokens * IDEAL_CC_PER_TOKEN_EXPERT)


def medium_candidates_for(hot: int) -> List[int]:
    lo = max(2, math.floor(0.35 * hot))
    hi = max(lo, math.ceil(0.70 * hot))
    cands = {max(2, hot // 2 - 1), max(2, hot // 2), max(2, hot // 2 + 1)}
    cands.update(x for x in BOUNDARY_NTOKS if lo <= x <= hi)
    return sorted(cands)


def make_dist(tokens: List[int]) -> Dict[str, int]:
    return {str(eid): int(ntok) for eid, ntok in enumerate(tokens)}


def choose_cache(tokens: List[int], mode: str) -> Tuple[int, int]:
    n = len(tokens)
    if mode == "none":
        return -1, -1
    if mode == "hot_pair" and n >= 2:
        return 0, 1
    if mode == "medium_pair" and n >= 4:
        return 2, 3
    if mode == "hot0":
        return 0, -1
    if mode == "medium0" and n >= 3:
        return 2, -1
    return -1, -1


def case_variants(hot: int, med: int) -> Iterable[Tuple[str, List[int]]]:
    med2 = max(1, med - 1)
    yield "no_tiny", [hot, hot - 1, med, med2]
    yield "tiny1", [hot, hot - 1, med, med2, 1]
    yield "tiny2", [hot, hot - 1, med, med2, 2]
    yield "two_tiny", [hot, hot - 1, med, med2, 1, 1]
    yield "asym_tiny", [hot, max(1, hot - 2), med, max(1, med - 2), 1]


def generate_cases() -> List[dict]:
    cases = []
    seen = set()
    for hot in H_VALUES:
        for med in medium_candidates_for(hot):
            if med >= hot:
                continue
            for variant, tokens in case_variants(hot, med):
                if any(t <= 0 for t in tokens):
                    continue
                for cache_mode in CACHE_MODES:
                    c2, c3 = choose_cache(tokens, cache_mode)
                    key = (tuple(tokens), c2, c3)
                    if key in seen:
                        continue
                    seen.add(key)
                    total = sum(tokens)
                    cases.append(
                        {
                            "case_id": len(cases),
                            "family": "hot_hot_medium_medium_filler",
                            "variant": variant,
                            "hot": hot,
                            "medium": med,
                            "active_n": len(tokens),
                            "assignment_total": total,
                            "compute_only_ideal_cc": compute_only_ideal_cc(total),
                            "cache_mode": cache_mode,
                            "c2": c2,
                            "c3": c3,
                            "tokens_sorted": tokens,
                            "dist": make_dist(tokens),
                        }
                    )
    return cases


def atomic_write(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def new_payload() -> dict:
    cases = generate_cases()
    return {
        "meta": {
            "name": "strategy_discovery_hot_hot_medium_medium_filler",
            "description": (
                "Targeted cases for discovering scheduler strategies. Lower "
                "bound is compute-only ideal; smaller ratios are better."
            ),
            "ideal_formula": "ceil(sum(ntokens) * 8448)",
            "beam_mode": BEAM_MODE,
            "n_cases": len(cases),
            "h_values": list(H_VALUES),
            "cache_modes": list(CACHE_MODES),
            "complete": False,
        },
        "cases": cases,
    }


def load_or_create(path: Path, regenerate: bool) -> dict:
    if regenerate or not path.exists():
        payload = new_payload()
        atomic_write(path, payload)
        return payload
    with path.open() as f:
        return json.load(f)


def action_to_dict(action) -> dict:
    return {
        "tag": action.tag,
        "c2_eid": action.c2_eid,
        "c2_ntok": action.c2_ntok,
        "c2_start": action.c2_start,
        "c2_s1_cached": bool(action.c2_s1_cached),
        "c2_s3_cached": bool(action.c2_s3_cached),
        "c3_eid": action.c3_eid,
        "c3_ntok": action.c3_ntok,
        "c3_start": action.c3_start,
        "c3_s1_cached": bool(action.c3_s1_cached),
        "c3_s3_cached": bool(action.c3_s3_cached),
        "c2_s2pf_start": action.c2_s2pf_start,
        "c3_s2pf_start": action.c3_s2pf_start,
    }


def maybe_run_c_mirror(dist: Dict[int, int], c2: int, c3: int, enabled: bool):
    if not enabled:
        return None
    from eval_c_mirror_v2 import c_mirror_v2_schedule

    return c_mirror_v2_schedule(dist, c2, c3)


def run_one(case: dict, beam_width: int, include_c_mirror: bool) -> None:
    dist = {int(k): int(v) for k, v in case["dist"].items()}
    c2 = int(case["c2"])
    c3 = int(case["c3"])
    ideal = int(case["compute_only_ideal_cc"])

    analytical_t0 = time.time()
    analytical_cc = analytical_schedule(dist, c2, c3)
    analytical_s = time.time() - analytical_t0

    c_mirror_cc = maybe_run_c_mirror(dist, c2, c3, include_c_mirror)

    beam_t0 = time.time()
    beam_cc, history = FourStageScheduler(
        dist,
        beam_width=beam_width,
        initial_cache_c2=c2,
        initial_cache_c3=c3,
    ).run()
    beam_s = time.time() - beam_t0

    result = {
        "status": "ok",
        "beam_width": beam_width,
        "beam_mode": BEAM_MODE,
        "analytical_cc": analytical_cc,
        "analytical_runtime_s": round(analytical_s, 6),
        "beam_cc": beam_cc,
        "beam_runtime_s": round(beam_s, 6),
        "beam_path": [action_to_dict(a) for a in history],
        "beam_path_tags": [a.tag for a in history],
        "analytical_over_ideal": analytical_cc / ideal if ideal else None,
        "beam_over_ideal": beam_cc / ideal if ideal else None,
        "beam_over_analytical": beam_cc / analytical_cc if analytical_cc else None,
        "analytical_minus_beam": analytical_cc - beam_cc,
    }
    if c_mirror_cc is not None:
        result["c_mirror_cc"] = c_mirror_cc
        result["c_mirror_over_ideal"] = c_mirror_cc / ideal if ideal else None
        result["c_mirror_minus_beam"] = c_mirror_cc - beam_cc
    case["result"] = result


def summarize(payload: dict) -> dict:
    cases = payload["cases"]
    ok_cases = [c for c in cases if c.get("result", {}).get("status") == "ok"]
    errors = [c for c in cases if c.get("result", {}).get("status") == "error"]
    wins = [c for c in ok_cases if c["result"]["analytical_minus_beam"] > 0]
    ties = [c for c in ok_cases if c["result"]["analytical_minus_beam"] == 0]
    losses = [c for c in ok_cases if c["result"]["analytical_minus_beam"] < 0]

    def avg(vals):
        return sum(vals) / len(vals) if vals else None

    best_wins = sorted(
        wins,
        key=lambda c: c["result"]["analytical_minus_beam"],
        reverse=True,
    )[:20]

    return {
        "total_cases": len(cases),
        "completed_cases": len(ok_cases),
        "error_cases": len(errors),
        "pending_cases": len(cases) - len(ok_cases) - len(errors),
        "run_complete": len(ok_cases) + len(errors) == len(cases),
        "beam_better_than_analytical": len(wins),
        "beam_equal_analytical": len(ties),
        "beam_worse_than_analytical": len(losses),
        "avg_analytical_over_ideal": avg(
            [c["result"]["analytical_over_ideal"] for c in ok_cases]
        ),
        "avg_beam_over_ideal": avg([c["result"]["beam_over_ideal"] for c in ok_cases]),
        "best_beam_wins": [
            {
                "case_id": c["case_id"],
                "variant": c["variant"],
                "tokens": c["tokens_sorted"],
                "cache_mode": c["cache_mode"],
                "ideal": c["compute_only_ideal_cc"],
                "analytical": c["result"]["analytical_cc"],
                "beam": c["result"]["beam_cc"],
                "analytical_minus_beam": c["result"]["analytical_minus_beam"],
                "beam_over_ideal": c["result"]["beam_over_ideal"],
                "path": c["result"]["beam_path_tags"],
            }
            for c in best_wins
        ],
    }


def write_summary(summary_path: Path, payload: dict) -> dict:
    summary = summarize(payload)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--include-c-mirror", action="store_true")
    parser.add_argument("--save-every", type=int, default=1)
    args = parser.parse_args()

    payload = load_or_create(args.out, args.regenerate)

    if args.generate_only:
        summary = write_summary(args.summary_out, payload)
        print(f"GENERATED {args.out} cases={len(payload['cases'])}")
        print(f"SUMMARY {args.summary_out} run_complete={summary['run_complete']}")
        return

    if args.summary_only:
        summary = write_summary(args.summary_out, payload)
        print(json.dumps(summary, indent=2))
        return

    done_this_run = 0
    error_this_run = 0
    t_all = time.time()
    for idx, case in enumerate(payload["cases"]):
        if args.limit is not None and done_this_run >= args.limit:
            break
        result = case.get("result", {})
        if not args.force and result.get("status") == "ok":
            continue
        try:
            print(
                f"RUN case_id={case['case_id']} variant={case['variant']} "
                f"tokens={case['tokens_sorted']} cache={case['cache_mode']}",
                flush=True,
            )
            run_one(case, args.beam_width, args.include_c_mirror)
            done_this_run += 1
            r = case["result"]
            print(
                f"OK case_id={case['case_id']} analytical={r['analytical_cc']} "
                f"beam={r['beam_cc']} ideal={case['compute_only_ideal_cc']} "
                f"analytical_minus_beam={r['analytical_minus_beam']} "
                f"beam_over_ideal={r['beam_over_ideal']:.6f} "
                f"beam_s={r['beam_runtime_s']:.3f}",
                flush=True,
            )
        except KeyboardInterrupt:
            print("INTERRUPTED: saving progress before exit", flush=True)
            break
        except Exception as exc:
            case["result"] = {"status": "error", "error": repr(exc)}
            error_this_run += 1
            print(f"ERROR case_id={case['case_id']} {exc!r}", flush=True)

        if args.save_every > 0 and (done_this_run + error_this_run) % args.save_every == 0:
            summary = write_summary(args.summary_out, payload)
            payload["meta"]["complete"] = summary["run_complete"]
            payload["meta"]["last_runtime_s"] = round(time.time() - t_all, 6)
            atomic_write(args.out, payload)
            print(
                f"SAVED completed={summary['completed_cases']} "
                f"pending={summary['pending_cases']} out={args.out}",
                flush=True,
            )

    summary = write_summary(args.summary_out, payload)
    payload["meta"]["complete"] = summary["run_complete"]
    payload["meta"]["last_runtime_s"] = round(time.time() - t_all, 6)
    atomic_write(args.out, payload)
    print(
        f"SUMMARY completed={summary['completed_cases']} "
        f"errors={summary['error_cases']} pending={summary['pending_cases']} "
        f"beam_better={summary['beam_better_than_analytical']} "
        f"beam_equal={summary['beam_equal_analytical']} "
        f"beam_worse={summary['beam_worse_than_analytical']}",
        flush=True,
    )
    if summary["run_complete"]:
        print(f"ALL_DONE results={args.out} summary={args.summary_out}", flush=True)
    else:
        print(f"NOT_DONE results={args.out} summary={args.summary_out}", flush=True)


if __name__ == "__main__":
    main()
