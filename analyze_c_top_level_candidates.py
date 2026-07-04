#!/usr/bin/env python3
"""Analyze committed top-level candidates in the current C-style scheduler."""

from __future__ import annotations

import argparse
import collections
import json
import sys
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
    for md in (8, 4, 2):
        if cut == md:
            return f"front_m{md}"
        if cut == ntok - md:
            return f"tail_m{md}"
    return "other"


def event_counter(events: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(collections.Counter(str(e.get(key, "NA")) for e in events))


def schedule_with_top_trace(token_dist: dict[int, int], c2_cache: int, c3_cache: int):
    remaining = tuple(sorted(((int(e), int(n)) for e, n in token_dist.items()), key=lambda x: -x[1]))
    c2 = cm._cc_initial(c2_cache)
    c3 = cm._cc_initial(c3_cache)
    events: list[dict[str, Any]] = []

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
            best_event = None

            for ci in (0, 1):
                snap_ci = c2 if ci == 0 else c3
                peer = c3 if ci == 0 else c2
                tst = snap_ci.task_end
                cc = cm._cc_swiglu_hit(top0_eid, snap_ci, tst)
                cf = cm._cc_down_hit(top0_eid, snap_ci, tst)
                for s1 in (0, 1, 2):
                    for s3 in (0, 1, 2):
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
                                "family": "n1_solo",
                                "shape_s1": s1,
                                "shape_s3": s3,
                                "cluster": ci,
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
                    s1a, s3a, s1b, s3b = cm._cc_pick_shapes(
                        cut_a, cut_b, c2c0, c2f0, c3c0, c3f0, tnow
                    )
                    sna = cm._cc_mk_snap(tnow, s1a, s3a, cut_a, top0_eid, c2c0, c2f0)
                    snb = cm._cc_mk_snap(tnow, s1b, s3b, cut_b, top0_eid, c3c0, c3f0)
                    sna, snb = cm._cc_try_s2pf_pair(sna, s3a, snb, s3b)
                    if not cm._cc_bw_ok(sna, snb):
                        continue
                    e = max(sna.task_end, snb.task_end)
                    if e < best_cost:
                        best_cost = e
                        best_sn = sna
                        split_snb = snb
                        is_split = True
                        best_event = {
                            "family": "n1_split",
                            "cut_kind": split_cut_kind(top0_ntok, cut_a),
                            "cut": cut_a,
                        }

            if not both_idle:
                idle_ci = 0 if t2 < t3 else 1
                idle_s = c2 if idle_ci == 0 else c3
                busy_s = c3 if idle_ci == 0 else c2
                idle_t = t2 if idle_ci == 0 else t3
                for start_idx, tst in enumerate(cm._cc_busy_time_points(busy_s, idle_t)):
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
                            "family": "n1_early",
                            "start_idx": start_idx,
                            "start_kind": "idle_t" if start_idx == 0 else f"busy_release_{start_idx}",
                            "cluster": idle_ci,
                        }

            remaining = ()
            events.append(best_event)
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
            best_event = None

            def eval_pair(event, sa, s1a, s3a, sb, s1b, s3b, rem_after):
                nonlocal best_key, best_snap, best_rem, best_event
                ta, tb = cm._cc_try_s2pf_pair(sa, s3a, sb, s3b)
                if not cm._cc_bw_ok(ta, tb):
                    return
                cost = cm._cc_continuation_cost(ta, tb, rem_after)
                smx = max(ta.task_end, tb.task_end)
                smn = min(ta.task_end, tb.task_end)
                if cm._cc_cand_better(best_key, cost, smx, smn, len(rem_after)):
                    best_key = (cost, smx, smn, len(rem_after))
                    best_snap = (ta, tb)
                    best_rem = rem_after
                    best_event = event

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
                eval_pair(
                    {"family": "pair_top0_topk", "k": k, "direction": "top0_c2"},
                    sa, s1a, s3a, sb, s1b, s3b, rem_after,
                )

                sw_a = cm._cc_swiglu_hit(keid, c2, tnow)
                dn_a = cm._cc_down_hit(keid, c2, tnow)
                sw_b, dn_b = c3c0, c3f0
                s1a, s3a, s1b, s3b = cm._cc_pick_shapes(kntok, top0_ntok, sw_a, dn_a, sw_b, dn_b, tnow)
                sa = cm._cc_mk_snap(tnow, s1a, s3a, kntok, keid, sw_a, dn_a)
                sb = cm._cc_mk_snap(tnow, s1b, s3b, top0_ntok, top0_eid, sw_b, dn_b)
                eval_pair(
                    {"family": "pair_top0_topk", "k": k, "direction": "top0_c3"},
                    sa, s1a, s3a, sb, s1b, s3b, rem_after,
                )

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
                        eval_pair(
                            {"family": "pair_kj", "k": k, "j": j, "direction": "k_c2"},
                            sa, s1a, s3a, sb, s1b, s3b, rem_after,
                        )

                        sw_a = cm._cc_swiglu_hit(eid_j, c2, tnow)
                        dn_a = cm._cc_down_hit(eid_j, c2, tnow)
                        sw_b = cm._cc_swiglu_hit(eid_k, c3, tnow)
                        dn_b = cm._cc_down_hit(eid_k, c3, tnow)
                        s1a, s3a, s1b, s3b = cm._cc_pick_shapes(nt_j, nt_k, sw_a, dn_a, sw_b, dn_b, tnow)
                        sa = cm._cc_mk_snap(tnow, s1a, s3a, nt_j, eid_j, sw_a, dn_a)
                        sb = cm._cc_mk_snap(tnow, s1b, s3b, nt_k, eid_k, sw_b, dn_b)
                        eval_pair(
                            {"family": "pair_kj", "k": k, "j": j, "direction": "k_c3"},
                            sa, s1a, s3a, sb, s1b, s3b, rem_after,
                        )

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
                    eval_pair(
                        {
                            "family": "split_top0",
                            "cut": cut_a,
                            "cut_kind": split_cut_kind(top0_ntok, cut_a),
                        },
                        sa, s1a, s3a, sb, s1b, s3b, rem_after,
                    )

            if best_snap is not None:
                c2, c3 = best_snap
                remaining = best_rem
                events.append(best_event)
            else:
                c2 = cm._cc_mk_snap(tnow, cm.C_SHAPE_C, cm.C_SHAPE_C, top0_ntok, top0_eid, c2c0, c2f0)
                remaining = remaining[1:]
                events.append({"family": "fallback_top0"})
            continue

        idle_ci = 0 if t2 < t3 else 1
        idle_sn = c2 if idle_ci == 0 else c3
        busy_sn = c3 if idle_ci == 0 else c2
        idle_t = t2 if idle_ci == 0 else t3

        best_ms = cm.C_INF
        best_nb = None
        best_event = None
        for start_idx, tst in enumerate(cm._cc_busy_time_points(busy_sn, idle_t)):
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
                best_event = {
                    "family": "single_idle",
                    "start_idx": start_idx,
                    "start_kind": "idle_t" if start_idx == 0 else f"busy_release_{start_idx}",
                    "cluster": idle_ci,
                }

        remaining = remaining[1:]
        if best_nb is not None:
            if idle_ci == 0:
                c2 = best_nb
            else:
                c3 = best_nb
            events.append(best_event)
        else:
            cch = c2c0 if idle_ci == 0 else c3c0
            cfh = c2f0 if idle_ci == 0 else c3f0
            sf = cm._cc_mk_snap(idle_t, cm.C_SHAPE_C, cm.C_SHAPE_C, top0_ntok, top0_eid, cch, cfh)
            if idle_ci == 0:
                c2 = sf
            else:
                c3 = sf
            events.append({"family": "single_idle_fallback"})

    return max(c2.task_end, c3.task_end), events


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    pair_top0 = [e for e in events if e.get("family") == "pair_top0_topk"]
    pair_kj = [e for e in events if e.get("family") == "pair_kj"]
    split = [e for e in events if e.get("family") in {"split_top0", "n1_split"}]
    single = [e for e in events if e.get("family") in {"single_idle", "n1_early"}]
    return {
        "n_events": len(events),
        "family": event_counter(events, "family"),
        "pair_top0_k": event_counter(pair_top0, "k"),
        "pair_top0_direction": event_counter(pair_top0, "direction"),
        "pair_kj_combo": dict(collections.Counter(f"{e.get('k')},{e.get('j')}" for e in pair_kj)),
        "pair_kj_direction": event_counter(pair_kj, "direction"),
        "split_cut_kind": event_counter(split, "cut_kind"),
        "single_start_kind": event_counter(single, "start_kind"),
    }


def print_summary(summary: dict[str, Any]) -> None:
    n = summary["n_events"]
    print(f"committed_top_level_events={n}")
    for section in (
        "family",
        "pair_top0_k",
        "pair_top0_direction",
        "pair_kj_combo",
        "pair_kj_direction",
        "split_cut_kind",
        "single_start_kind",
    ):
        total = sum(summary[section].values())
        print(f"\n{section}:")
        for k, v in collections.Counter(summary[section]).most_common():
            pct = 100.0 * v / total if total else 0.0
            print(f"  {k:18s} {v:8d} {pct:6.2f}%")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-per-file", type=int, default=100)
    parser.add_argument("--out", type=Path, default=ROOT / "c_top_level_candidate_summary.json")
    args = parser.parse_args()

    sample = None if args.sample_per_file < 0 else args.sample_per_file
    all_events: list[dict[str, Any]] = []
    checked = 0
    mismatches = 0
    by_file: dict[str, Any] = {}
    for path in DEFAULT_FILES:
        payload = json.loads(path.read_text())
        cases = payload["cases"]
        idxs = stratified_indices(len(cases), sample)
        file_events: list[dict[str, Any]] = []
        for idx in idxs:
            case = cases[idx]
            dist = {int(k): int(v) for k, v in case["dist"].items()}
            ms, events = schedule_with_top_trace(dist, int(case["c2"]), int(case["c3"]))
            ref = cm.c_mirror_v2_schedule(dist, int(case["c2"]), int(case["c3"]))
            if ms != ref:
                mismatches += 1
            checked += 1
            file_events.extend(events)
            all_events.extend(events)
        by_file[path.name] = summarize(file_events)

    summary = summarize(all_events)
    report = {
        "sample_per_file": args.sample_per_file,
        "checked_cases": checked,
        "mirror_mismatches": mismatches,
        "overall": summary,
        "by_file": by_file,
    }
    args.out.write_text(json.dumps(report, indent=2))
    print_summary(summary)
    print(f"\nchecked_cases={checked} mirror_mismatches={mismatches}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
