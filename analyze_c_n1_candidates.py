#!/usr/bin/env python3
"""Analyze the final-expert (n=1) candidate selected by the current SW scheduler."""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import eval_c_mirror_v2 as cm


DEFAULT_FILES = (
    ROOT / "scheduler_eval_inputs_E8_stratified_v6.json",
    ROOT / "scheduler_eval_inputs_E32_stratified_v6.json",
    ROOT / "scheduler_eval_inputs_E64_stratified_v6.json",
)

SHAPE_NAME = {cm.C_SHAPE_A: "A", cm.C_SHAPE_B: "B", cm.C_SHAPE_C: "C"}


def stratified_indices(n_cases: int, n_pick: int | None) -> list[int]:
    if n_pick is None or n_pick >= n_cases:
        return list(range(n_cases))
    if n_pick <= 0:
        return []
    if n_pick == 1:
        return [0]
    return sorted(set(round(i * (n_cases - 1) / (n_pick - 1)) for i in range(n_pick)))


def split_cut_kind(ntok: int, cut: int) -> str:
    if cut == (ntok + 1) // 2:
        return "half_ceil"
    if cut == ntok // 2:
        return "half_floor"
    return "other"


def shape_pair(s1: int, s3: int) -> str:
    return f"{SHAPE_NAME.get(s1, s1)}/{SHAPE_NAME.get(s3, s3)}"


def s2pf_can_start(s: cm.CSnap, s3: int, ps: int) -> bool:
    return s.bw_s3 > 0 and ps >= s.task_start and ps + cm.C_TD3[s3] <= s.s2_end


def s2pf_dma1_start_valid(s: cm.CSnap, s3: int) -> bool:
    return (
        s2pf_can_start(s, s3, s.task_start)
        and s.dma1_end >= s.task_start
        and s.dma1_end + cm.C_TD3[s3] <= s.s2_end
    )


def lite_try_s2pf_pair(
    sa: cm.CSnap,
    s3a: int,
    sb: cm.CSnap,
    s3b: int,
    *,
    is_split: bool,
) -> tuple[cm.CSnap, cm.CSnap, str]:
    best_sc = -1
    best_ss = (1 << 64) - 1
    best_a, best_b = sa, sb
    best_tag = "invalid"
    can_a = s2pf_can_start(sa, s3a, sa.task_start)
    can_b = s2pf_can_start(sb, s3b, sb.task_start)
    hi_a = sa.s2_end - cm.C_TD3[s3a] if can_a else 0
    hi_b = sb.s2_end - cm.C_TD3[s3b] if can_b else 0

    def take(tag: str, ta: cm.CSnap, tb: cm.CSnap, score_count: int, score_sum: int) -> None:
        nonlocal best_sc, best_ss, best_a, best_b, best_tag
        if score_count > best_sc or (score_count == best_sc and score_sum < best_ss):
            best_sc = score_count
            best_ss = score_sum
            best_a, best_b = ta, tb
            best_tag = tag

    if cm._cc_bw_ok(sa, sb):
        take("none", sa, sb, 0, 0)

    if can_a and can_b:
        ta = cm._cc_apply_s2pf(sa, s3a, sa.task_start)
        tb = cm._cc_apply_s2pf(sb, s3b, sb.task_start)
        if ta.s2pf_start >= 0 and tb.s2pf_start >= 0 and cm._cc_bw_ok(ta, tb):
            take("both_task_start", ta, tb, 2, sa.task_start + sb.task_start)

    if s2pf_dma1_start_valid(sa, s3a) and s2pf_dma1_start_valid(sb, s3b):
        ta = cm._cc_apply_s2pf(sa, s3a, sa.dma1_end)
        tb = cm._cc_apply_s2pf(sb, s3b, sb.dma1_end)
        if ta.s2pf_start >= 0 and tb.s2pf_start >= 0 and cm._cc_bw_ok(ta, tb):
            take("both_dma1_end", ta, tb, 2, sa.dma1_end + sb.dma1_end)

    if is_split:
        if can_b:
            tb = cm._cc_apply_s2pf(sb, s3b, hi_b)
            if tb.s2pf_start >= 0 and cm._cc_bw_ok(sa, tb):
                take("b_only_latest", sa, tb, 1, hi_b)
    else:
        if can_a and can_b:
            ta = cm._cc_apply_s2pf(sa, s3a, hi_a)
            tb = cm._cc_apply_s2pf(sb, s3b, hi_b)
            if ta.s2pf_start >= 0 and tb.s2pf_start >= 0 and cm._cc_bw_ok(ta, tb):
                take("both_latest", ta, tb, 2, hi_a + hi_b)

    return best_a, best_b, best_tag


def continuation_cost_lite(c2: cm.CSnap, c3: cm.CSnap, rem: tuple) -> int:
    nr = len(rem)
    if nr == 0:
        return max(c2.task_end, c3.task_end)
    if nr == 1:
        return cm._cc_sim1(c2, c3, rem[0][0], rem[0][1])
    if nr == 2 and rem[0][1] + rem[1][1] <= cm.C_EXACT_TAIL_MAX:
        te = min(c2.task_end, c3.task_end)
        tl = max(c2.task_end, c3.task_end)
        ss = te + cm._cc_best_task(rem[0][1]) + cm._cc_best_task(rem[1][1])
        pa = tl + max(cm._cc_best_conc(rem[0][1]), cm._cc_best_conc(rem[1][1]))
        return min(max(tl, ss), pa)
    return cm._cc_greedy_h(c2.task_end, c3.task_end, rem)


def schedule_with_n1_trace(token_dist: dict[int, int], c2_cache: int, c3_cache: int) -> tuple[int, dict[str, Any]]:
    remaining = tuple(sorted(((int(e), int(n)) for e, n in token_dist.items()), key=lambda x: -x[1]))
    c2 = cm._cc_initial(c2_cache)
    c3 = cm._cc_initial(c3_cache)
    n1_event: dict[str, Any] | None = None

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
            best_event: dict[str, Any] | None = None
            is_split = False
            split_snb = None

            for ci in (0, 1):
                snap_ci = c2 if ci == 0 else c3
                peer = c3 if ci == 0 else c2
                tst = snap_ci.task_end
                cc = cm._cc_swiglu_hit(top0_eid, snap_ci, tst)
                cf = cm._cc_down_hit(top0_eid, snap_ci, tst)
                for s1 in (cm.C_SHAPE_A, cm.C_SHAPE_B, cm.C_SHAPE_C):
                    for s3 in (cm.C_SHAPE_A, cm.C_SHAPE_B, cm.C_SHAPE_C):
                        sn = cm._cc_mk_snap(tst, s1, s3, top0_ntok, top0_eid, cc, cf)
                        if not cm._cc_bw_ok(sn, peer):
                            continue
                        ms = max(sn.task_end, peer.task_end)
                        if ms < best_cost:
                            best_cost = ms
                            best_sn = sn
                            best_cl = ci
                            is_split = False
                            best_event = {
                                "method": "solo",
                                "cluster": "C2" if ci == 0 else "C3",
                                "shape": shape_pair(s1, s3),
                                "s1": SHAPE_NAME[s1],
                                "s3": SHAPE_NAME[s3],
                                "start_kind": "cluster_task_end",
                                "has_s2pf": False,
                            }

            if top0_ntok >= 2:
                cuts = []
                h1 = (top0_ntok + 1) // 2
                h2 = top0_ntok // 2
                cuts.append(h1)
                if h2 != h1 and 1 <= h2 <= top0_ntok - 1:
                    cuts.append(h2)
                for cut_a in cuts:
                    cut_b = top0_ntok - cut_a
                    s1a, s3a, s1b, s3b = cm._cc_pick_shapes(cut_a, cut_b, c2c0, c2f0, c3c0, c3f0, tnow)
                    sna = cm._cc_mk_snap(tnow, s1a, s3a, cut_a, top0_eid, c2c0, c2f0)
                    snb = cm._cc_mk_snap(tnow, s1b, s3b, cut_b, top0_eid, c3c0, c3f0)
                    sna, snb, tag = lite_try_s2pf_pair(sna, s3a, snb, s3b, is_split=True)
                    if not cm._cc_bw_ok(sna, snb):
                        continue
                    e = max(sna.task_end, snb.task_end)
                    if e < best_cost:
                        best_cost = e
                        best_sn = sna
                        split_snb = snb
                        is_split = True
                        best_event = {
                            "method": "split",
                            "cut_kind": split_cut_kind(top0_ntok, cut_a),
                            "cut_a": cut_a,
                            "cut_b": cut_b,
                            "shape_a": shape_pair(s1a, s3a),
                            "shape_b": shape_pair(s1b, s3b),
                            "s2pf": tag,
                            "has_s2pf": tag != "none",
                        }

            if not both_idle:
                idle_ci = 0 if t2 < t3 else 1
                idle_s = c2 if idle_ci == 0 else c3
                busy_s = c3 if idle_ci == 0 else c2
                idle_t = t2 if idle_ci == 0 else t3
                for ti, tst in enumerate(cm._cc_busy_time_points(busy_s, idle_t)):
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
                        best_event = {
                            "method": "early_solo",
                            "cluster": "C2" if idle_ci == 0 else "C3",
                            "shape": "C/C",
                            "start_idx": ti,
                            "start_kind": "idle_t" if ti == 0 else f"busy_release_{ti}",
                            "has_s2pf": False,
                        }

            remaining = ()
            n1_event = best_event or {"method": "fallback"}
            n1_event.update({"top0_ntok": top0_ntok, "both_idle": both_idle})
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

            def eval_pair(sa, s3a, sb, s3b, rem_after, *, is_split_pair: bool):
                nonlocal best_key, best_snap, best_rem
                ta, tb, _ = lite_try_s2pf_pair(sa, s3a, sb, s3b, is_split=is_split_pair)
                if not cm._cc_bw_ok(ta, tb):
                    return
                cost = continuation_cost_lite(ta, tb, rem_after)
                smx = max(ta.task_end, tb.task_end)
                smn = min(ta.task_end, tb.task_end)
                if cm._cc_cand_better(best_key, cost, smx, smn, len(rem_after)):
                    best_key = (cost, smx, smn, len(rem_after))
                    best_snap = (ta, tb)
                    best_rem = rem_after

            max_k = min(3, len(remaining) - 1)
            for k in range(1, max_k + 1):
                keid, kntok = remaining[k]
                rem_after = cm._cc_remove_eids(remaining, top0_eid, keid)
                sw_a, dn_a = c2c0, c2f0
                sw_b = cm._cc_swiglu_hit(keid, c3, tnow)
                dn_b = cm._cc_down_hit(keid, c3, tnow)
                s1a, s3a, s1b, s3b = cm._cc_pick_shapes(top0_ntok, kntok, sw_a, dn_a, sw_b, dn_b, tnow)
                sa = cm._cc_mk_snap(tnow, s1a, s3a, top0_ntok, top0_eid, sw_a, dn_a)
                sb = cm._cc_mk_snap(tnow, s1b, s3b, kntok, keid, sw_b, dn_b)
                eval_pair(sa, s3a, sb, s3b, rem_after, is_split_pair=False)

                sw_a = cm._cc_swiglu_hit(keid, c2, tnow)
                dn_a = cm._cc_down_hit(keid, c2, tnow)
                sw_b, dn_b = c3c0, c3f0
                s1a, s3a, s1b, s3b = cm._cc_pick_shapes(kntok, top0_ntok, sw_a, dn_a, sw_b, dn_b, tnow)
                sa = cm._cc_mk_snap(tnow, s1a, s3a, kntok, keid, sw_a, dn_a)
                sb = cm._cc_mk_snap(tnow, s1b, s3b, top0_ntok, top0_eid, sw_b, dn_b)
                eval_pair(sa, s3a, sb, s3b, rem_after, is_split_pair=False)

            if len(remaining) >= 3:
                mkj = min(3, len(remaining) - 1)
                for k in range(1, mkj):
                    for j in range(k + 1, mkj + 1):
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
                        eval_pair(sa, s3a, sb, s3b, rem_after, is_split_pair=False)

                        sw_a = cm._cc_swiglu_hit(eid_j, c2, tnow)
                        dn_a = cm._cc_down_hit(eid_j, c2, tnow)
                        sw_b = cm._cc_swiglu_hit(eid_k, c3, tnow)
                        dn_b = cm._cc_down_hit(eid_k, c3, tnow)
                        s1a, s3a, s1b, s3b = cm._cc_pick_shapes(nt_j, nt_k, sw_a, dn_a, sw_b, dn_b, tnow)
                        sa = cm._cc_mk_snap(tnow, s1a, s3a, nt_j, eid_j, sw_a, dn_a)
                        sb = cm._cc_mk_snap(tnow, s1b, s3b, nt_k, eid_k, sw_b, dn_b)
                        eval_pair(sa, s3a, sb, s3b, rem_after, is_split_pair=False)

            if top0_ntok >= 2:
                cuts = []
                h1 = (top0_ntok + 1) // 2
                h2 = top0_ntok // 2
                cuts.append(h1)
                if h2 != h1 and h2 >= 1:
                    cuts.append(h2)
                for md in (8, 4, 2):
                    if md < top0_ntok and md not in cuts:
                        cuts.append(md)
                    if top0_ntok > md:
                        k2 = top0_ntok - md
                        if k2 >= 1 and k2 not in cuts:
                            cuts.append(k2)
                rem_after = cm._cc_remove_eids(remaining, top0_eid)
                for cut_a in cuts:
                    cut_b = top0_ntok - cut_a
                    if cut_a == 0 or cut_b == 0:
                        continue
                    s1a, s3a, s1b, s3b = cm._cc_pick_shapes(cut_a, cut_b, c2c0, c2f0, c3c0, c3f0, tnow)
                    sa = cm._cc_mk_snap(tnow, s1a, s3a, cut_a, top0_eid, c2c0, c2f0)
                    sb = cm._cc_mk_snap(tnow, s1b, s3b, cut_b, top0_eid, c3c0, c3f0)
                    eval_pair(sa, s3a, sb, s3b, rem_after, is_split_pair=True)

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
            if sn.bw_s3 > 0 and cm.C_TD3[cm.C_SHAPE_C] <= sn.s2_end - sn.task_start:
                hi = sn.s2_end - cm.C_TD3[cm.C_SHAPE_C]
                cand = cm._cc_apply_s2pf(sn, cm.C_SHAPE_C, hi)
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

    return max(c2.task_end, c3.task_end), n1_event or {"method": "none"}


def counter(events: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(collections.Counter(str(e.get(key, "NA")) for e in events))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-per-file", type=int, default=-1)
    parser.add_argument("--out", type=Path, default=ROOT / "c_n1_candidate_summary_full.json")
    args = parser.parse_args()

    t0 = time.perf_counter()
    events: list[dict[str, Any]] = []
    sample = None if args.sample_per_file < 0 else args.sample_per_file
    for path in DEFAULT_FILES:
        payload = json.loads(path.read_text())
        cases = payload["cases"]
        for idx in stratified_indices(len(cases), sample):
            case = cases[idx]
            dist = {int(k): int(v) for k, v in case["dist"].items()}
            makespan, ev = schedule_with_n1_trace(dist, int(case["c2"]), int(case["c3"]))
            ev.update(
                {
                    "file": path.name,
                    "case_id": case["case_id"],
                    "profile": case["profile"],
                    "active_n": case["active_n"],
                    "m_total": case["m_total"],
                    "makespan": makespan,
                }
            )
            events.append(ev)

    by_method = counter(events, "method")
    solo = [e for e in events if e.get("method") == "solo"]
    split = [e for e in events if e.get("method") == "split"]
    early = [e for e in events if e.get("method") == "early_solo"]
    report = {
        "sample_per_file": args.sample_per_file,
        "runtime_s": time.perf_counter() - t0,
        "total": len(events),
        "method": by_method,
        "method_by_file": {
            name: counter([e for e in events if e["file"] == name], "method")
            for name in sorted({e["file"] for e in events})
        },
        "method_by_profile": {
            name: counter([e for e in events if e["profile"] == name], "method")
            for name in sorted({e["profile"] for e in events})
        },
        "solo_shape": counter(solo, "shape"),
        "solo_cluster": counter(solo, "cluster"),
        "split_cut_kind": counter(split, "cut_kind"),
        "split_s2pf": counter(split, "s2pf"),
        "early_start_kind": counter(early, "start_kind"),
        "early_cluster": counter(early, "cluster"),
        "top0_ntok_by_method": {
            method: dict(collections.Counter(str(e.get("top0_ntok", "NA")) for e in events if e.get("method") == method))
            for method in sorted(by_method)
        },
        "events": events,
    }
    args.out.write_text(json.dumps(report, indent=2))

    print(f"cases={len(events)} runtime_s={report['runtime_s']:.3f}")
    print("method", json.dumps(report["method"], sort_keys=True))
    print("solo_shape", json.dumps(report["solo_shape"], sort_keys=True))
    print("split_cut_kind", json.dumps(report["split_cut_kind"], sort_keys=True))
    print("split_s2pf", json.dumps(report["split_s2pf"], sort_keys=True))
    print("early_start_kind", json.dumps(report["early_start_kind"], sort_keys=True))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
