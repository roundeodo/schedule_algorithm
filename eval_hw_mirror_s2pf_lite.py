#!/usr/bin/env python3
"""Exact Python mirror of the pruned C/RTL hardware scheduler policy."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import eval_c_mirror_v2 as cm


DEFAULT_INPUTS = (
    ROOT / "scheduler_strategy_coverage_E8.json",
    ROOT / "scheduler_strategy_coverage_E32.json",
    ROOT / "scheduler_strategy_coverage_E64.json",
)


POLICY_TEMPLATES = {
    # Lowest-complexity version:
    #   - pair/split only try no-pf and symmetric both-side S2PF.
    #   - no one-side pair/split S2PF.
    "minimal": {
        "pair": ("both_dma1_end", "none"),
        "split": ("both_dma1_end", "none"),
    },
    # More faithful but still hardware-bounded version:
    #   - pair: none + symmetric both-side placements.
    #   - split: additionally keeps B-only, because full-run statistics showed
    #     split_top0 uses B-only often enough to be worth evaluating.
    "balanced": {
        "pair": ("both_dma1_end", "none"),
        "split": ("both_dma1_end", "b_only_dma1_end", "none"),
    },
}


TOP_POLICIES = ("full", "pruned")
N1_POLICIES = ("full", "pruned")
N1_PRUNED_SOLO_SHAPES = (
    (2, 2),  # C/C
    (0, 0),  # A/A
    (0, 2),  # A/C
    (1, 1),  # B/B
    (0, 1),  # A/B
)


def _split_cuts_for_policy(ntok: int, *, top_policy: str) -> list[int]:
    cuts: list[int] = []

    def add(cut: int) -> None:
        if 1 <= cut <= ntok - 1 and cut not in cuts:
            cuts.append(cut)

    add((ntok + 1) // 2)

    if top_policy == "pruned":
        # Keep the only non-half split cut that still had visible usage in the
        # full C trace. front_m4/front_m8 and tail cuts are dropped for HW.
        add(2)
        return cuts

    if top_policy == "full":
        for md in (8, 4, 2):
            add(md)
            add(ntok - md)
        return cuts

    raise ValueError(f"unknown top_policy {top_policy!r}; choose {TOP_POLICIES}")


def _n1_split_cuts(ntok: int, *, n1_policy: str) -> list[int]:
    h1 = (ntok + 1) // 2
    if n1_policy == "pruned":
        return [h1] if 1 <= h1 <= ntok - 1 else []
    if n1_policy == "full":
        cuts = [h1]
        h2 = ntok // 2
        if h2 != h1 and 1 <= h2 <= ntok - 1:
            cuts.append(h2)
        return cuts
    raise ValueError(f"unknown n1_policy {n1_policy!r}; choose {N1_POLICIES}")


def _start_for(kind: str, sn: cm.CSnap, s3: int) -> int:
    if kind == "dma1_end":
        return sn.dma1_end
    raise ValueError(f"unknown S2PF start kind: {kind}")


def _apply_required_s2pf(sn: cm.CSnap, s3: int, kind: str) -> cm.CSnap | None:
    cand = cm._cc_apply_s2pf(sn, s3, _start_for(kind, sn, s3))
    return cand if cand.s2pf_start >= 0 else None


def _consider_s2pf_template(
    template: str,
    sa: cm.CSnap,
    s3a: int,
    sb: cm.CSnap,
    s3b: int,
) -> tuple[cm.CSnap, cm.CSnap] | None:
    if template == "none":
        return (sa, sb) if cm._cc_bw_ok(sa, sb) else None

    if template.startswith("both_"):
        kind = template.removeprefix("both_")
        ta = _apply_required_s2pf(sa, s3a, kind)
        tb = _apply_required_s2pf(sb, s3b, kind)
        if ta is None or tb is None:
            return None
        return (ta, tb) if cm._cc_bw_ok(ta, tb) else None

    if template.startswith("a_only_"):
        kind = template.removeprefix("a_only_")
        ta = _apply_required_s2pf(sa, s3a, kind)
        if ta is None:
            return None
        return (ta, sb) if cm._cc_bw_ok(ta, sb) else None

    if template.startswith("b_only_"):
        kind = template.removeprefix("b_only_")
        tb = _apply_required_s2pf(sb, s3b, kind)
        if tb is None:
            return None
        return (sa, tb) if cm._cc_bw_ok(sa, tb) else None

    raise ValueError(f"unknown S2PF template: {template}")


def _hw_try_s2pf_pair(
    family: str,
    sa: cm.CSnap,
    s3a: int,
    sb: cm.CSnap,
    s3b: int,
    *,
    policy: str,
) -> tuple[cm.CSnap, cm.CSnap]:
    """Bounded hardware-friendly replacement for C's 25-way try_s2pf_pair()."""
    if policy not in POLICY_TEMPLATES:
        raise ValueError(f"unknown policy {policy!r}; choose {sorted(POLICY_TEMPLATES)}")

    family_key = "split" if family in {"split_top0", "n1_split"} else "pair"
    templates = POLICY_TEMPLATES[policy][family_key]

    for template in templates:
        cand = _consider_s2pf_template(template, sa, s3a, sb, s3b)
        if cand is not None:
            return cand
    return sa, sb


def _hw_sim1(c2: cm.CSnap, c3: cm.CSnap, eid: int, ntok: int, *, policy: str) -> int:
    t = max(c2.task_end, c3.task_end)
    best = cm.C_INF
    for sn_ci in (c2, c3):
        cc = cm._cc_swiglu_hit(eid, sn_ci, t)
        cf = cm._cc_down_hit(eid, sn_ci, t)
        sn = cm._cc_mk_snap(t, cm.C_SHAPE_C, cm.C_SHAPE_C, ntok, eid, cc, cf)
        best = min(best, sn.task_end)
        if not cc:
            sn2 = cm._cc_mk_snap(t, cm.C_SHAPE_B, cm.C_SHAPE_B, ntok, eid, False, False)
            best = min(best, sn2.task_end)

    if ntok >= 2:
        ca = (ntok + 1) // 2
        cb = ntok - ca
        sw_a = cm._cc_swiglu_hit(eid, c2, t)
        dn_a = cm._cc_down_hit(eid, c2, t)
        sw_b = cm._cc_swiglu_hit(eid, c3, t)
        dn_b = cm._cc_down_hit(eid, c3, t)
        s1a, s3a, s1b, s3b = cm._cc_pick_shapes(ca, cb, sw_a, dn_a, sw_b, dn_b, t)
        sna = cm._cc_mk_snap(t, s1a, s3a, ca, eid, sw_a, dn_a)
        snb = cm._cc_mk_snap(t, s1b, s3b, cb, eid, sw_b, dn_b)
        sna, snb = _hw_try_s2pf_pair("n1_split", sna, s3a, snb, s3b, policy=policy)
        if cm._cc_bw_ok(sna, snb):
            best = min(best, max(sna.task_end, snb.task_end))

    return t + cm._cc_best_task(ntok) if best == cm.C_INF else best


def _hw_continuation_cost(c2: cm.CSnap, c3: cm.CSnap, rem: tuple, *, policy: str) -> int:
    nr = len(rem)
    if nr == 0:
        return max(c2.task_end, c3.task_end)
    if nr == 1:
        return _hw_sim1(c2, c3, rem[0][0], rem[0][1], policy=policy)
    if nr == 2 and rem[0][1] + rem[1][1] <= cm.C_EXACT_TAIL_MAX:
        te = min(c2.task_end, c3.task_end)
        tl = max(c2.task_end, c3.task_end)
        ss = te + cm._cc_best_task(rem[0][1]) + cm._cc_best_task(rem[1][1])
        pa = tl + max(cm._cc_best_conc(rem[0][1]), cm._cc_best_conc(rem[1][1]))
        return min(max(tl, ss), pa)
    return cm._cc_greedy_h(c2.task_end, c3.task_end, rem)


def _hw_cand_better(best, cost: int, snap_max: int, rem_len: int) -> bool:
    """Mirror the RTL key: cost, remaining count, then max task end."""
    if best is None:
        return True
    return (cost, rem_len, snap_max) < best


def hw_mirror_schedule(
    token_dist: dict[int, int],
    initial_cache_c2: int = -1,
    initial_cache_c3: int = -1,
    *,
    policy: str = "balanced",
    top_policy: str = "pruned",
    n1_policy: str = "pruned",
) -> int:
    """Run the hardware-oriented scheduler mirror and return makespan."""
    if policy not in POLICY_TEMPLATES:
        raise ValueError(f"unknown policy {policy!r}; choose {sorted(POLICY_TEMPLATES)}")
    if top_policy not in TOP_POLICIES:
        raise ValueError(f"unknown top_policy {top_policy!r}; choose {TOP_POLICIES}")
    if n1_policy not in N1_POLICIES:
        raise ValueError(f"unknown n1_policy {n1_policy!r}; choose {N1_POLICIES}")

    remaining = tuple(sorted(((int(e), int(n)) for e, n in token_dist.items()), key=lambda x: -x[1]))
    c2 = cm._cc_initial(initial_cache_c2)
    c3 = cm._cc_initial(initial_cache_c3)

    while remaining:
        top0_eid, top0_ntok = remaining[0]
        t2, t3 = c2.task_end, c3.task_end
        tnow = max(t2, t3)
        both_idle = t2 == t3

        if cm._cc_s4pf_ok_with_peer(c2, c3):
            c2 = cm._cc_apply_s4pf_ghost(c2)
        if cm._cc_s4pf_ok_with_peer(c3, c2):
            c3 = cm._cc_apply_s4pf_ghost(c3)

        c2c0 = cm._cc_swiglu_hit(top0_eid, c2, tnow)
        c2f0 = cm._cc_down_hit(top0_eid, c2, tnow)
        c3c0 = cm._cc_swiglu_hit(top0_eid, c3, tnow)
        c3f0 = cm._cc_down_hit(top0_eid, c3, tnow)

        if len(remaining) == 1:
            best_cost = cm.C_INF
            best_sn = None
            best_cl = 0
            is_split = False
            split_snb = None

            solo_shapes = N1_PRUNED_SOLO_SHAPES if n1_policy == "pruned" else tuple(
                (s1, s3) for s1 in (0, 1, 2) for s3 in (0, 1, 2)
            )
            for ci in (0, 1):
                snap_ci = c2 if ci == 0 else c3
                peer = c3 if ci == 0 else c2
                tst = snap_ci.task_end
                cc = cm._cc_swiglu_hit(top0_eid, snap_ci, tst)
                cf = cm._cc_down_hit(top0_eid, snap_ci, tst)
                for s1, s3 in solo_shapes:
                    sn = cm._cc_mk_snap(tst, s1, s3, top0_ntok, top0_eid, cc, cf)
                    if not cm._cc_bw_ok(sn, peer):
                        continue
                    ms = max(sn.task_end, peer.task_end)
                    if ms < best_cost:
                        best_cost = ms
                        best_sn = sn
                        best_cl = ci
                        is_split = False

            if top0_ntok >= 2:
                for cut_a in _n1_split_cuts(top0_ntok, n1_policy=n1_policy):
                    cut_b = top0_ntok - cut_a
                    s1a, s3a, s1b, s3b = cm._cc_pick_shapes(
                        cut_a, cut_b, c2c0, c2f0, c3c0, c3f0, tnow
                    )
                    sna = cm._cc_mk_snap(tnow, s1a, s3a, cut_a, top0_eid, c2c0, c2f0)
                    snb = cm._cc_mk_snap(tnow, s1b, s3b, cut_b, top0_eid, c3c0, c3f0)
                    sna, snb = _hw_try_s2pf_pair("n1_split", sna, s3a, snb, s3b, policy=policy)
                    if not cm._cc_bw_ok(sna, snb):
                        continue
                    e = max(sna.task_end, snb.task_end)
                    if e < best_cost:
                        best_cost = e
                        best_sn = sna
                        split_snb = snb
                        is_split = True

            if not both_idle:
                idle_ci = 0 if t2 < t3 else 1
                idle_s = c2 if idle_ci == 0 else c3
                busy_s = c3 if idle_ci == 0 else c2
                idle_t = t2 if idle_ci == 0 else t3
                tpts = cm._cc_busy_time_points(busy_s, idle_t)
                if n1_policy == "pruned":
                    tpts = tpts[1:]
                for tst in tpts:
                    cc = cm._cc_swiglu_hit(top0_eid, idle_s, tst)
                    cf = cm._cc_down_hit(top0_eid, idle_s, tst)
                    sn = cm._cc_mk_snap(tst, cm.C_SHAPE_C, cm.C_SHAPE_C, top0_ntok, top0_eid, cc, cf)
                    ok = cm._cc_bw_ok(sn, busy_s) if idle_ci == 0 else cm._cc_bw_ok(busy_s, sn)
                    if not ok:
                        continue
                    ms = max(sn.task_end, busy_s.task_end)
                    if ms < best_cost:
                        best_cost = ms
                        best_sn = sn
                        best_cl = idle_ci
                        is_split = False

            remaining = ()
            if is_split:
                c2, c3 = best_sn, split_snb
            else:
                if best_cl == 0:
                    c2 = best_sn
                else:
                    c3 = best_sn
            break

        if both_idle:
            best_key = None
            best_snap = None
            best_rem = None

            def eval_pair(family, sa, s1a, s3a, sb, s1b, s3b, rem_after):
                nonlocal best_key, best_snap, best_rem
                ta, tb = _hw_try_s2pf_pair(family, sa, s3a, sb, s3b, policy=policy)
                if not cm._cc_bw_ok(ta, tb):
                    return
                cost = _hw_continuation_cost(ta, tb, rem_after, policy=policy)
                smx = max(ta.task_end, tb.task_end)
                if _hw_cand_better(best_key, cost, smx, len(rem_after)):
                    best_key = (cost, len(rem_after), smx)
                    best_snap = (ta, tb)
                    best_rem = rem_after

            top0_ks = [1] if top_policy == "pruned" else list(range(1, min(3, len(remaining) - 1) + 1))
            for k in top0_ks:
                if k >= len(remaining):
                    continue
                keid, kntok = remaining[k]
                rem_after = cm._cc_remove_eids(remaining, top0_eid, keid)

                sw_a, dn_a = c2c0, c2f0
                sw_b = cm._cc_swiglu_hit(keid, c3, tnow)
                dn_b = cm._cc_down_hit(keid, c3, tnow)
                s1a, s3a, s1b, s3b = cm._cc_pick_shapes(top0_ntok, kntok, sw_a, dn_a, sw_b, dn_b, tnow)
                sa = cm._cc_mk_snap(tnow, s1a, s3a, top0_ntok, top0_eid, sw_a, dn_a)
                sb = cm._cc_mk_snap(tnow, s1b, s3b, kntok, keid, sw_b, dn_b)
                eval_pair("pair_top0_topk", sa, s1a, s3a, sb, s1b, s3b, rem_after)

                if top_policy == "full":
                    sw_a = cm._cc_swiglu_hit(keid, c2, tnow)
                    dn_a = cm._cc_down_hit(keid, c2, tnow)
                    sw_b, dn_b = c3c0, c3f0
                    s1a, s3a, s1b, s3b = cm._cc_pick_shapes(kntok, top0_ntok, sw_a, dn_a, sw_b, dn_b, tnow)
                    sa = cm._cc_mk_snap(tnow, s1a, s3a, kntok, keid, sw_a, dn_a)
                    sb = cm._cc_mk_snap(tnow, s1b, s3b, top0_ntok, top0_eid, sw_b, dn_b)
                    eval_pair("pair_top0_topk", sa, s1a, s3a, sb, s1b, s3b, rem_after)

            if len(remaining) >= 3:
                if top_policy == "pruned":
                    kj_pairs = [(1, 2), (2, 3)]
                else:
                    mkj = min(3, len(remaining) - 1)
                    kj_pairs = [(k, j) for k in range(1, mkj) for j in range(k + 1, mkj + 1)]
                for k, j in kj_pairs:
                    if j >= len(remaining):
                        continue
                    eid_k, nt_k = remaining[k]
                    eid_j, nt_j = remaining[j]
                    rem_after = cm._cc_remove_eids(remaining, eid_k, eid_j)
                    if not rem_after:
                        continue

                    sw_a = cm._cc_swiglu_hit(eid_k, c2, tnow)
                    dn_a = cm._cc_down_hit(eid_k, c2, tnow)
                    sw_b = cm._cc_swiglu_hit(eid_j, c3, tnow)
                    dn_b = cm._cc_down_hit(eid_j, c3, tnow)
                    s1a, s3a, s1b, s3b = cm._cc_pick_shapes(nt_k, nt_j, sw_a, dn_a, sw_b, dn_b, tnow)
                    sa = cm._cc_mk_snap(tnow, s1a, s3a, nt_k, eid_k, sw_a, dn_a)
                    sb = cm._cc_mk_snap(tnow, s1b, s3b, nt_j, eid_j, sw_b, dn_b)
                    eval_pair("pair_kj", sa, s1a, s3a, sb, s1b, s3b, rem_after)

                    if top_policy == "full":
                        sw_a = cm._cc_swiglu_hit(eid_j, c2, tnow)
                        dn_a = cm._cc_down_hit(eid_j, c2, tnow)
                        sw_b = cm._cc_swiglu_hit(eid_k, c3, tnow)
                        dn_b = cm._cc_down_hit(eid_k, c3, tnow)
                        s1a, s3a, s1b, s3b = cm._cc_pick_shapes(nt_j, nt_k, sw_a, dn_a, sw_b, dn_b, tnow)
                        sa = cm._cc_mk_snap(tnow, s1a, s3a, nt_j, eid_j, sw_a, dn_a)
                        sb = cm._cc_mk_snap(tnow, s1b, s3b, nt_k, eid_k, sw_b, dn_b)
                        eval_pair("pair_kj", sa, s1a, s3a, sb, s1b, s3b, rem_after)

            if top0_ntok >= 2:
                rem_after = cm._cc_remove_eids(remaining, top0_eid)
                for cut_a in _split_cuts_for_policy(top0_ntok, top_policy=top_policy):
                    cut_b = top0_ntok - cut_a
                    s1a, s3a, s1b, s3b = cm._cc_pick_shapes(cut_a, cut_b, c2c0, c2f0, c3c0, c3f0, tnow)
                    sa = cm._cc_mk_snap(tnow, s1a, s3a, cut_a, top0_eid, c2c0, c2f0)
                    sb = cm._cc_mk_snap(tnow, s1b, s3b, cut_b, top0_eid, c3c0, c3f0)
                    eval_pair("split_top0", sa, s1a, s3a, sb, s1b, s3b, rem_after)

            if best_snap is not None:
                c2, c3 = best_snap
                remaining = best_rem
            else:
                c2 = cm._cc_mk_snap(tnow, cm.C_SHAPE_C, cm.C_SHAPE_C, top0_ntok, top0_eid, c2c0, c2f0)
                remaining = remaining[1:]
            continue

        idle_ci = 0 if t2 < t3 else 1
        idle_sn = c2 if idle_ci == 0 else c3
        busy_sn = c3 if idle_ci == 0 else c2
        idle_t = t2 if idle_ci == 0 else t3

        best_ms = cm.C_INF
        best_nb = None
        for tst in cm._cc_busy_time_points(busy_sn, idle_t):
            cc = cm._cc_swiglu_hit(top0_eid, idle_sn, tst)
            cf = cm._cc_down_hit(top0_eid, idle_sn, tst)
            sn = cm._cc_mk_snap(tst, cm.C_SHAPE_C, cm.C_SHAPE_C, top0_ntok, top0_eid, cc, cf)
            if sn.bw_s3 > 0 and cm.C_TD3[cm.C_SHAPE_C] <= sn.s2_end - sn.dma1_end:
                cand = cm._cc_apply_s2pf(sn, cm.C_SHAPE_C, sn.dma1_end)
                if cand.s2pf_start >= 0:
                    ok2 = cm._cc_bw_ok(cand, busy_sn) if idle_ci == 0 else cm._cc_bw_ok(busy_sn, cand)
                    if ok2:
                        sn = cand
            ok = cm._cc_bw_ok(sn, busy_sn) if idle_ci == 0 else cm._cc_bw_ok(busy_sn, sn)
            if not ok:
                continue
            ms = max(sn.task_end, busy_sn.task_end)
            if ms < best_ms:
                best_ms = ms
                best_nb = sn

        remaining = remaining[1:]
        if best_nb is not None:
            if idle_ci == 0:
                c2 = best_nb
            else:
                c3 = best_nb
        else:
            cch = c2c0 if idle_ci == 0 else c3c0
            cfh = c2f0 if idle_ci == 0 else c3f0
            sf = cm._cc_mk_snap(idle_t, cm.C_SHAPE_C, cm.C_SHAPE_C, top0_ntok, top0_eid, cch, cfh)
            if idle_ci == 0:
                c2 = sf
            else:
                c3 = sf

    return max(c2.task_end, c3.task_end)


def _stratified_indices(n_cases: int, n_pick: int | None) -> list[int]:
    if n_pick is None or n_pick >= n_cases:
        return list(range(n_cases))
    if n_pick <= 0:
        return []
    if n_pick == 1:
        return [0]
    return sorted(set(round(i * (n_cases - 1) / (n_pick - 1)) for i in range(n_pick)))


def _summarize_ratios(ratios: Iterable[float]) -> dict:
    vals = list(ratios)
    if not vals:
        return {}
    return {
        "n": len(vals),
        "mean": statistics.mean(vals),
        "median": statistics.median(vals),
        "max": max(vals),
        "min": min(vals),
        "exact": sum(1 for r in vals if abs(r - 1.0) <= 1e-12),
        "le_1pct": sum(1 for r in vals if r <= 1.01),
        "le_5pct": sum(1 for r in vals if r <= 1.05),
        "lt_full": sum(1 for r in vals if r < 1.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=sorted(POLICY_TEMPLATES), default="balanced")
    parser.add_argument("--top-policy", choices=TOP_POLICIES, default="pruned")
    parser.add_argument("--n1-policy", choices=N1_POLICIES, default="pruned")
    parser.add_argument("--sample-per-file", type=int, default=100)
    parser.add_argument("--out", type=Path, default=ROOT / "hw_mirror_s2pf_lite_eval.json")
    args = parser.parse_args()

    sample = None if args.sample_per_file < 0 else args.sample_per_file
    rows = []
    t0 = time.perf_counter()
    for path in DEFAULT_INPUTS:
        payload = json.loads(path.read_text())
        cases = payload["cases"]
        idxs = _stratified_indices(len(cases), sample)
        for idx in idxs:
            case = cases[idx]
            dist = {int(k): int(v) for k, v in case["dist"].items()}
            full = cm.c_mirror_v2_schedule(dist, int(case["c2"]), int(case["c3"]))
            hw = hw_mirror_schedule(
                dist,
                int(case["c2"]),
                int(case["c3"]),
                policy=args.policy,
                top_policy=args.top_policy,
                n1_policy=args.n1_policy,
            )
            rows.append(
                {
                    "file": path.name,
                    "case_id": case["case_id"],
                    "active_n": case["active_n"],
                    "m_total": case["m_total"],
                    "construction": case["construction"],
                    "full_c_cc": full,
                    "hw_cc": hw,
                    "ratio": hw / full if full else None,
                }
            )

    ratios = [r["ratio"] for r in rows if r["ratio"] is not None]
    report = {
        "policy": args.policy,
        "top_policy": args.top_policy,
        "n1_policy": args.n1_policy,
        "sample_per_file": args.sample_per_file,
        "runtime_s": time.perf_counter() - t0,
        "summary": _summarize_ratios(ratios),
        "worst_cases": sorted(rows, key=lambda r: r["ratio"], reverse=True)[:20],
        "rows": rows,
    }
    args.out.write_text(json.dumps(report, indent=2))

    s = report["summary"]
    print(
        f"policy={args.policy} top_policy={args.top_policy} "
        f"n1_policy={args.n1_policy} cases={s.get('n', 0)} runtime_s={report['runtime_s']:.3f}"
    )
    print(
        f"ratio hw/full: mean={s.get('mean', 0):.6f} "
        f"median={s.get('median', 0):.6f} max={s.get('max', 0):.6f} min={s.get('min', 0):.6f}"
    )
    print(
        f"exact={s.get('exact', 0)} le_1pct={s.get('le_1pct', 0)} "
        f"le_5pct={s.get('le_5pct', 0)} hw_better_than_full={s.get('lt_full', 0)}"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
