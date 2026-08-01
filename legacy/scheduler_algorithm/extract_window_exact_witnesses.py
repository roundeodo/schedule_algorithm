#!/usr/bin/env python3
"""Extract immutable, replay-audited witnesses from window exact campaigns.

The campaign aggregate is allowed to remain incomplete: a feasible root
fragment is already a constructive sufficiency certificate for that case and
window.  This tool never turns timeout or partial exhaustion into evidence.
It verifies every referenced manifest/hash, checks per-action window
visibility, and replays the complete explicit-DMA history before emitting a
proof payload suitable for ``evaluate_directed_window_grid.py``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import four_stage_scheduler as reference  # noqa: E402
from run_four_stage_reference import deserialize_action  # noqa: E402


TICK_CC = reference.SCHEDULE_TIME_QUANTUM_CC


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ticks(cc: int) -> str:
    value = Fraction(int(cc), TICK_CC)
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_immutable(path: Path, payload: dict) -> None:
    serialized = json.dumps(payload, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise SystemExit(f"refusing to overwrite different witness proof: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, path)


def _audit_fragment(
    *,
    prior_row: dict,
    campaign_path: Path,
    campaign: dict,
    campaign_row: dict,
    expected_prior_sha256: str,
) -> tuple[dict, tuple[int, int]]:
    name = str(campaign_row["name"])
    if not campaign_row.get("feasible"):
        raise RuntimeError(f"{name}: selected campaign row is not feasible")
    fragment_name = campaign_row.get("feasible_fragment")
    fragment_hash = campaign_row.get("feasible_fragment_sha256")
    if not fragment_name or not fragment_hash:
        raise RuntimeError(f"{name}: feasible row lacks immutable fragment evidence")

    work_dir = Path(campaign_row["work_dir"])
    manifest_path = work_dir / "manifest.json"
    fragment_path = work_dir / str(fragment_name)
    if not manifest_path.is_file() or not fragment_path.is_file():
        raise RuntimeError(f"{name}: missing case manifest or feasible fragment")
    if _sha256(fragment_path) != fragment_hash:
        raise RuntimeError(f"{name}: feasible fragment hash mismatch")

    manifest = _load(manifest_path)
    fragment = _load(fragment_path)
    if fragment.get("manifest_id") != manifest.get("manifest_id"):
        raise RuntimeError(f"{name}: fragment/case-manifest ID mismatch")
    if manifest.get("case") != name:
        raise RuntimeError(f"{name}: case manifest names a different case")
    if manifest.get("prior_proof_sha256") != expected_prior_sha256:
        raise RuntimeError(f"{name}: case manifest prior-proof hash mismatch")
    current_reference_hash = _sha256(HERE / "four_stage_scheduler.py")
    if manifest.get("reference_sha256") != current_reference_hash:
        raise RuntimeError(f"{name}: case manifest reference hash is stale")
    window = reference.normalize_candidate_window(
        tuple(int(value) for value in manifest["candidate_window"])
    )
    if window is None:
        raise RuntimeError(f"{name}: witness campaign is not window restricted")
    if list(window) != list(campaign["manifest"]["candidate_window"]):
        raise RuntimeError(f"{name}: case/campaign window mismatch")

    target_ticks = Fraction(str(prior_row["best_reference_ticks"]))
    if (
        not prior_row.get("proven_optimal")
        or Fraction(str(prior_row["certified_lower_bound_ticks"])) != target_ticks
        or Fraction(str(manifest["target_ticks"])) != target_ticks
        or Fraction(str(campaign_row["target_ticks"])) != target_ticks
    ):
        raise RuntimeError(f"{name}: witness target is not the proved LB=UB")
    if not fragment.get("feasible") or fragment.get("exhaustive"):
        raise RuntimeError(f"{name}: malformed feasible fragment verdict")
    if not fragment.get("actions"):
        raise RuntimeError(f"{name}: feasible fragment has no history")

    token_dist = {
        eid: int(ntok)
        for eid, ntok in enumerate(prior_row["counts"])
        if int(ntok) > 0
    }
    actions = tuple(deserialize_action(raw) for raw in fragment["actions"])
    state = reference.FourStageScheduler(token_dist)._initial_state()
    for index, action in enumerate(actions):
        visible = reference.candidate_window_visible_eids(
            state.c2, state.c3, state.remaining, window
        )
        if not reference.action_within_candidate_window(action, visible):
            raise RuntimeError(
                f"{name}: action {index} violates window {window}"
            )
        state = reference.apply_action(state, action)
    if state.remaining:
        raise RuntimeError(f"{name}: feasible history is non-terminal")
    replay_cc = reference.validate_schedule_history(actions, token_dist)
    target_cc = target_ticks * TICK_CC
    if target_cc.denominator != 1 or replay_cc != int(target_cc):
        raise RuntimeError(
            f"{name}: replay {_ticks(replay_cc)} != target {target_ticks}"
        )

    row = dict(prior_row)
    row.update(
        actions=list(fragment["actions"]),
        best_reference_ticks=str(target_ticks),
        certified_lower_bound_ticks=str(target_ticks),
        optimality_gap=0.0,
        proven_optimal=True,
        history_replay_valid=True,
        termination="restricted_window_feasible_optimal_witness",
        expansions=int(fragment.get("expansions", 0)),
        generated_states=int(fragment.get("generated", 0)),
        search_runtime_s=float(fragment.get("runtime_s", 0.0)),
        selected_history_source="window_exact_root_fragment",
        selected_proof_source=str(fragment_path.resolve()),
    )
    row["window_exact_evidence"] = {
        "campaign": str(campaign_path.resolve()),
        "case_manifest": str(manifest_path.resolve()),
        "case_manifest_sha256": _sha256(manifest_path),
        "manifest_id": manifest["manifest_id"],
        "fragment": str(fragment_path.resolve()),
        "fragment_sha256": fragment_hash,
        "candidate_window": list(window),
        "target_ticks": str(target_ticks),
        "semantic_root_groups": int(manifest["semantic_root_groups"]),
        "root_children": int(manifest["root_children"]),
        "feasible_group_index": int(fragment["group_index"]),
        "semantic_group_key": list(fragment["semantic_group_key"]),
        "expansions": int(fragment.get("expansions", 0)),
        "generated": int(fragment.get("generated", 0)),
        "runtime_s": float(fragment.get("runtime_s", 0.0)),
        "history_replay_ticks": _ticks(replay_cc),
        "per_action_window_visible": True,
    }
    return row, window


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-proof", type=Path, required=True)
    parser.add_argument(
        "--campaign", type=Path, action="append", required=True
    )
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prior = _load(args.prior_proof)
    if not prior.get("complete"):
        raise SystemExit(f"prior proof is incomplete: {args.prior_proof}")
    prior_by_name = {str(row["name"]): row for row in prior["cases"]}
    prior_hash = _sha256(args.prior_proof)
    requested = set(args.case)

    candidates: dict[str, list[tuple[tuple[int, int, int], dict]]] = {}
    for campaign_path in args.campaign:
        campaign = _load(campaign_path)
        manifest = campaign.get("manifest", {})
        if campaign.get("schema") != "window_exact_priority_campaign_v1":
            raise SystemExit(f"unexpected campaign schema: {campaign_path}")
        if manifest.get("prior_proof_sha256") != prior_hash:
            raise SystemExit(f"campaign prior-proof hash mismatch: {campaign_path}")
        for campaign_row in campaign.get("cases", []):
            name = str(campaign_row["name"])
            if requested and name not in requested:
                continue
            if not campaign_row.get("feasible"):
                continue
            if name not in prior_by_name:
                raise SystemExit(f"campaign case absent from prior proof: {name}")
            row, window = _audit_fragment(
                prior_row=prior_by_name[name],
                campaign_path=campaign_path,
                campaign=campaign,
                campaign_row=campaign_row,
                expected_prior_sha256=prior_hash,
            )
            priority = (sum(window), window[0], window[1])
            candidates.setdefault(name, []).append((priority, row))

    missing = requested - set(candidates)
    if missing:
        raise SystemExit(f"requested cases lack feasible fragments: {sorted(missing)}")
    rows = [
        min(candidates[name], key=lambda item: item[0])[1]
        for name in sorted(candidates)
    ]
    if not rows:
        raise SystemExit("no feasible window witnesses found")
    window_counts = Counter(
        "+".join(str(value) for value in row["window_exact_evidence"]["candidate_window"])
        for row in rows
    )
    payload = {
        "schema": "window_exact_witness_proof_v1",
        "complete": True,
        "proof_model": "explicit_dma_lane_four_stage_window_exact",
        "prior_proof": str(args.prior_proof.resolve()),
        "prior_proof_sha256": prior_hash,
        "campaigns": [str(path.resolve()) for path in args.campaign],
        "summary": {
            "cases": len(rows),
            "proven_optimal": len(rows),
            "history_replay_valid": len(rows),
            "per_action_window_visible": len(rows),
            "by_candidate_window": dict(sorted(window_counts.items())),
        },
        "cases": rows,
    }
    _atomic_write_immutable(args.output, payload)
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
