#!/usr/bin/env python3
"""Compare bounded TOP+BOTTOM expert windows against validated reference histories.

For a proved-optimal target history, direct visibility is already a constructive
window-sufficiency witness.  If that history is not visible, a restricted legal
beam searches for an alternative optimal history.  Unproved targets are always
searched because merely replaying their current best-known history cannot rule
out improvement.  Every expensive case/window probe runs in a fresh process.
"""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import analyze_directed_case_classification as history_audit
import four_stage_scheduler as reference
from run_four_stage_reference import deserialize_action


HERE = Path(__file__).resolve().parent
PROVER = HERE / "prove_top4_bottom2_directed.py"


def _window_name(window: tuple[int, int]) -> str:
    top, bottom = window
    return f"top{top}" if bottom == 0 else f"top{top}+bottom{bottom}"


def _parse_windows(text: str) -> tuple[tuple[int, int], ...]:
    windows = []
    for item in text.split(";"):
        fields = item.strip().split(",")
        if len(fields) != 2:
            raise SystemExit("--windows must use TOP,BOTTOM;TOP,BOTTOM syntax")
        try:
            window = (int(fields[0]), int(fields[1]))
        except ValueError as exc:
            raise SystemExit("--windows must contain integers") from exc
        if window[0] <= 0 or window[1] < 0:
            raise SystemExit("window TOP must be positive and BOTTOM non-negative")
        windows.append(window)
    if len(set(windows)) != len(windows):
        raise SystemExit("--windows contains duplicates")
    return tuple(windows)


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reuse_complete_direct_audit(
    audit_path: Path,
    *,
    names: list[str],
    windows: tuple[tuple[int, int], ...],
    target_proof_sha256: str,
    case_input_sha256: str | None,
) -> dict[str, dict[str, list[dict]]]:
    """Reuse only hash-verified direct witnesses from a completed audit.

    This avoids repeating the potentially expensive equal-load relabel DFS for
    histories that were already replayed by the same explicit-DMA reference.
    Unresolved rows remain unresolved; heuristic rows and exact-campaign rows
    are deliberately not promoted through this path.
    """
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "directed_window_grid_v1" or not payload.get(
        "complete"
    ):
        raise SystemExit(f"prior window audit is incomplete: {audit_path}")
    config = payload.get("run_config", {})
    if not config.get("direct_only"):
        raise SystemExit(
            f"prior window audit was not direct-only: {audit_path}"
        )
    if config.get("target_proof_sha256") != target_proof_sha256:
        raise SystemExit(f"prior window audit target hash mismatch: {audit_path}")
    if config.get("case_input_sha256") != case_input_sha256:
        raise SystemExit(f"prior window audit case-input hash mismatch: {audit_path}")
    if config.get("source_sha256", {}).get(
        "four_stage_scheduler.py"
    ) != _file_sha256(HERE / "four_stage_scheduler.py"):
        raise SystemExit(f"prior window audit reference hash mismatch: {audit_path}")

    def collect_source_hashes(
        audit_config: dict, seen_audits: set[str]
    ) -> dict[str, str]:
        if audit_config.get("target_proof_sha256") != target_proof_sha256:
            raise SystemExit("target hash mismatch in prior audit chain")
        if audit_config.get("case_input_sha256") != case_input_sha256:
            raise SystemExit("case-input hash mismatch in prior audit chain")
        if audit_config.get("source_sha256", {}).get(
            "four_stage_scheduler.py"
        ) != _file_sha256(HERE / "four_stage_scheduler.py"):
            raise SystemExit("reference hash mismatch in prior audit chain")
        source_hashes = {
            str(Path(audit_config["target_proof"]).resolve()): audit_config[
                "target_proof_sha256"
            ]
        }
        witness_paths = audit_config.get("witness_proofs", [])
        witness_hashes = audit_config.get("witness_proof_sha256", [])
        if len(witness_paths) != len(witness_hashes):
            raise SystemExit(
                f"prior window audit witness manifest is malformed: {audit_path}"
            )
        for path, digest in zip(witness_paths, witness_hashes):
            resolved = str(Path(path).resolve())
            previous = source_hashes.get(resolved)
            if previous is not None and previous != digest:
                raise SystemExit(
                    f"conflicting source hashes in prior audit chain: {resolved}"
                )
            source_hashes[resolved] = digest

        parent = audit_config.get("prior_window_audit")
        if parent is None:
            return source_hashes
        parent_path = Path(parent).resolve()
        parent_key = str(parent_path)
        if parent_key in seen_audits:
            raise SystemExit(f"cycle in prior window audit chain: {parent_path}")
        expected_parent_hash = audit_config.get("prior_window_audit_sha256")
        if (
            not parent_path.is_file()
            or expected_parent_hash is None
            or _file_sha256(parent_path) != expected_parent_hash
        ):
            raise SystemExit(
                f"prior window audit chain hash mismatch: {parent_path}"
            )
        parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
        if (
            parent_payload.get("schema") != "directed_window_grid_v1"
            or not parent_payload.get("complete")
            or not parent_payload.get("run_config", {}).get("direct_only")
        ):
            raise SystemExit(
                f"invalid parent in prior window audit chain: {parent_path}"
            )
        inherited = collect_source_hashes(
            parent_payload["run_config"], seen_audits | {parent_key}
        )
        for source, digest in inherited.items():
            previous = source_hashes.get(source)
            if previous is not None and previous != digest:
                raise SystemExit(
                    f"conflicting source hashes in prior audit chain: {source}"
                )
            source_hashes[source] = digest
        return source_hashes

    source_hashes = collect_source_hashes(
        config, {str(audit_path.resolve())}
    )
    for source, expected in source_hashes.items():
        source_path = Path(source)
        if not source_path.is_file() or _file_sha256(source_path) != expected:
            raise SystemExit(
                f"prior window audit source hash mismatch: {source_path}"
            )

    requested_windows = {_window_name(window) for window in windows}
    rows_by_key = {}
    for row in payload.get("results", []):
        key = (str(row["name"]), str(row["window"]))
        if key in rows_by_key:
            raise SystemExit(f"duplicate row in prior window audit: {key}")
        rows_by_key[key] = row
    expected_keys = {
        (name, window_name)
        for name in names
        for window_name in requested_windows
    }
    missing = expected_keys - set(rows_by_key)
    if missing:
        raise SystemExit(
            f"prior window audit is missing requested pairs: {sorted(missing)}"
        )

    reused = {
        name: {window_name: [] for window_name in requested_windows}
        for name in names
    }
    for key in sorted(expected_keys):
        row = rows_by_key[key]
        if not row.get("target_proven_optimal"):
            raise SystemExit(f"prior audit target is not proved optimal: {key}")
        if not row.get("direct_target_history_visible"):
            continue
        if (
            not row.get("window_reaches_target")
            or row.get("window_status") != "proved_sufficient_direct"
        ):
            raise SystemExit(f"invalid direct verdict in prior audit: {key}")
        sources = row.get("direct_witness_sources", [])
        if not sources:
            raise SystemExit(f"prior direct verdict has no source: {key}")
        for source in sources:
            source_path = str(Path(source["source"]).resolve())
            if source_path not in source_hashes:
                raise SystemExit(
                    f"prior direct source is absent from its manifest: {source_path}"
                )
            reused[key[0]][key[1]].append(
                {
                    "source": source["source"],
                    "equal_load_id_relabel": bool(
                        source.get("equal_load_id_relabel")
                    ),
                    "reused_from_audit": str(audit_path.resolve()),
                    "reused_audit_sha256": _file_sha256(audit_path),
                }
            )
    return reused


def _summary(rows: list[dict], windows: tuple[tuple[int, int], ...]) -> dict:
    by_window = {}
    for window in windows:
        name = _window_name(window)
        members = [row for row in rows if row["window"] == name]
        proved_targets = [row for row in members if row["target_proven_optimal"]]
        unproved_targets = [row for row in members if not row["target_proven_optimal"]]
        exact_insufficient = [
            row for row in proved_targets
            if row["window_status"] == "proved_insufficient_exact"
        ]
        unresolved = [
            row for row in proved_targets
            if row["window_status"] == "unresolved"
        ]
        by_window[name] = {
            "entries": sum(window),
            "top_entries": window[0],
            "bottom_entries": window[1],
            "cases": len(members),
            "proved_target_cases": len(proved_targets),
            "proved_optimal_covered": sum(
                row["window_reaches_target"] for row in proved_targets
            ),
            "proved_optimal_exact_insufficient": [
                row["name"] for row in exact_insufficient
            ],
            "proved_optimal_unresolved": [
                row["name"] for row in unresolved
            ],
            "unproved_target_cases": len(unproved_targets),
            "unproved_best_known_reached": sum(
                row["window_reaches_target"] for row in unproved_targets
            ),
            "unproved_target_improved": sum(
                row.get("window_minus_target_ticks") is not None
                and Fraction(row["window_minus_target_ticks"]) < 0
                for row in unproved_targets
            ),
            "restricted_searches": sum(row["restricted_search_run"] for row in members),
            "restricted_histories_found": sum(
                row["restricted_window_history_found"] for row in members
            ),
        }
    return {
        "case_window_pairs": len(rows),
        "windows": len(windows),
        "by_window": by_window,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-input",
        type=Path,
        help="external generated cases; omit for the frozen built-in suite",
    )
    parser.add_argument("--target-proof", type=Path, required=True)
    parser.add_argument(
        "--witness-proof",
        type=Path,
        action="append",
        default=[],
        help=(
            "optional additional proof payloads; equal-optimal replay-valid "
            "histories are audited as alternative window witnesses"
        ),
    )
    parser.add_argument(
        "--prior-window-audit",
        type=Path,
        help=(
            "completed direct-only audit whose hash-verified direct verdicts "
            "are reused; only newly supplied witness proofs are replayed"
        ),
    )
    parser.add_argument(
        "--exact-campaign",
        type=Path,
        action="append",
        default=[],
        help=(
            "completed restricted-window root campaign; may be repeated to "
            "add exact sufficiency or insufficiency evidence"
        ),
    )
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument(
        "--only-proven-targets",
        action="store_true",
        help="exclude unproved target rows from the window grid",
    )
    parser.add_argument(
        "--require-all-proven-targets",
        action="store_true",
        help="reject the run unless every selected target has a global proof",
    )
    parser.add_argument(
        "--expected-cases",
        type=int,
        default=0,
        help="reject the run unless the selected case count matches",
    )
    parser.add_argument(
        "--reach-target-only",
        action="store_true",
        help=(
            "treat any replay-valid saved best-known history as a constructive "
            "target-reachability witness; search only incompatible histories"
        ),
    )
    parser.add_argument(
        "--direct-only",
        action="store_true",
        help=(
            "audit saved replay-valid histories only; leave incompatible "
            "case/window pairs unresolved without launching heuristic probes"
        ),
    )
    parser.add_argument(
        "--windows",
        default=(
            "4,0;4,2;4,4;4,8;6,0;6,2;6,4;6,8;"
            "8,0;8,2;8,4;8,8;10,0;12,0;14,0;16,0"
        ),
    )
    parser.add_argument("--seed-beam-widths", default="8")
    parser.add_argument("--seed-beam-modes", default="completion,cache,lpt,f_g")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--keep-fragments", action="store_true")
    args = parser.parse_args()

    if args.expected_cases < 0:
        raise SystemExit("--expected-cases must be non-negative")

    windows = _parse_windows(args.windows)
    target_payload = json.loads(args.target_proof.read_text(encoding="utf-8"))
    if not target_payload.get("complete", False):
        raise SystemExit(f"target proof is incomplete: {args.target_proof}")
    target_by_name = {row["name"]: row for row in target_payload["cases"]}
    exact_by_key: dict[tuple[str, str], dict] = {}
    for campaign_path in args.exact_campaign:
        campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
        if campaign.get("schema") != "target_root_branch_proof_v1":
            raise SystemExit(f"unsupported exact campaign: {campaign_path}")
        if not campaign.get("complete"):
            raise SystemExit(f"exact campaign is incomplete: {campaign_path}")
        manifest = campaign["manifest"]
        if manifest.get("reference_sha256") != _file_sha256(
            HERE / "four_stage_scheduler.py"
        ):
            raise SystemExit(f"{campaign_path}: reference source SHA-256 mismatch")
        raw_window = manifest.get("candidate_window")
        if not isinstance(raw_window, list) or len(raw_window) != 2:
            raise SystemExit(
                f"campaign is not a restricted-window proof: {campaign_path}"
            )
        window = (int(raw_window[0]), int(raw_window[1]))
        window_name = _window_name(window)
        case_name = str(manifest["case"])
        if case_name not in target_by_name:
            raise SystemExit(
                f"exact campaign case is absent from target proof: {case_name}"
            )
        target = target_by_name[case_name]
        if [int(value) for value in manifest["counts"]] != [
            int(value) for value in target["counts"]
        ]:
            raise SystemExit(f"{campaign_path}: distribution mismatch")
        quantum = int(manifest["time_quantum_cc"])
        target_cc = Fraction(target["best_reference_ticks"]) * quantum
        if target_cc.denominator != 1 or int(target_cc) != int(manifest["target_cc"]):
            raise SystemExit(f"{campaign_path}: optimum target mismatch")
        if not target.get("proven_optimal") or Fraction(
            target["certified_lower_bound_ticks"]
        ) != Fraction(target["best_reference_ticks"]):
            raise SystemExit(f"{campaign_path}: target is not proved LB=UB")
        summary = campaign["summary"]
        found = bool(summary.get("found_feasible_history"))
        exhausted = bool(summary.get("all_root_groups_exhaustive"))
        if found == exhausted:
            raise SystemExit(f"{campaign_path}: invalid exact campaign verdict")
        if exhausted and (
            int(summary.get("recorded_groups", -1))
            != int(manifest["semantic_root_groups"])
            or any(
                branch["feasible"] or not branch["exhaustive"]
                for branch in campaign["branches"]
            )
        ):
            raise SystemExit(
                f"{campaign_path}: exhaustive verdict omits a root group"
            )
        exact = {
            "status": (
                "proved_sufficient_exact" if found
                else "proved_insufficient_exact"
            ),
            "campaign": str(campaign_path.resolve()),
            "campaign_sha256": _file_sha256(campaign_path),
            "manifest_id": manifest["manifest_id"],
            "root_children": int(manifest["root_children"]),
            "semantic_root_groups": int(manifest["semantic_root_groups"]),
            "total_expansions": int(summary["total_expansions"]),
            "total_generated": int(summary["total_generated"]),
        }
        if found:
            feasible = [row for row in campaign["branches"] if row["feasible"]]
            if not feasible:
                raise SystemExit(f"{campaign_path}: feasible branch is missing")
            branch = min(feasible, key=lambda row: int(row["group_index"]))
            distribution = {
                eid: int(ntok)
                for eid, ntok in enumerate(manifest["counts"])
                if int(ntok) > 0
            }
            history = tuple(
                deserialize_action(action) for action in branch["actions"]
            )
            replay_cc = reference.validate_schedule_history(history, distribution)
            if replay_cc != int(target_cc):
                raise SystemExit(
                    f"{campaign_path}: feasible history replay misses optimum"
                )
            scheduler = reference.FourStageScheduler(distribution)
            state = scheduler._initial_state()
            for action in history:
                visible_eids = reference.candidate_window_visible_eids(
                    state.c2, state.c3, state.remaining, window
                )
                if not reference.action_within_candidate_window(
                    action, visible_eids
                ):
                    raise SystemExit(
                        f"{campaign_path}: feasible history violates its window"
                    )
                state = reference.apply_action(state, action)
            exact["feasible_group_index"] = int(branch["group_index"])
            exact["history_replay_cc"] = replay_cc
        key = (case_name, window_name)
        previous = exact_by_key.get(key)
        if previous is not None and previous["status"] != exact["status"]:
            raise SystemExit(f"conflicting exact campaigns for {key}")
        exact_by_key[key] = exact
    witnesses_by_name: dict[str, list[tuple[str, dict]]] = {}
    for witness_path in args.witness_proof:
        witness_payload = json.loads(witness_path.read_text(encoding="utf-8"))
        for row in witness_payload["cases"]:
            if not row.get("history_replay_valid") or not row.get("proven_optimal"):
                continue
            witnesses_by_name.setdefault(row["name"], []).append(
                (str(witness_path), row)
            )
    if args.case_input is None:
        names = [str(row["name"]) for row in target_payload["cases"]]
    else:
        source = json.loads(args.case_input.read_text(encoding="utf-8"))
        names = [str(row["name"]) for row in source["cases"]]
    missing_targets = set(names) - set(target_by_name)
    if missing_targets:
        raise SystemExit(
            f"target proof is missing cases: {sorted(missing_targets)}"
        )
    if args.case:
        requested = set(args.case)
        missing = requested - set(names)
        if missing:
            raise SystemExit(f"unknown cases: {sorted(missing)}")
        names = [name for name in names if name in requested]
    if args.expected_cases > 0 and len(names) != args.expected_cases:
        raise SystemExit(
            f"selected case count {len(names)} != expected {args.expected_cases}"
        )
    if args.require_all_proven_targets:
        unproven = [name for name in names if not target_by_name[name]["proven_optimal"]]
        if unproven:
            raise SystemExit(
                f"selected targets are not all proved optimal: {unproven}"
            )
    if args.only_proven_targets:
        names = [name for name in names if target_by_name[name]["proven_optimal"]]

    config = {
        "case_input": str(args.case_input.resolve()) if args.case_input else None,
        "case_input_sha256": (
            _file_sha256(args.case_input) if args.case_input else None
        ),
        "target_proof": str(args.target_proof.resolve()),
        "target_proof_sha256": _file_sha256(args.target_proof),
        "witness_proofs": [str(path.resolve()) for path in args.witness_proof],
        "witness_proof_sha256": [
            _file_sha256(path) for path in args.witness_proof
        ],
        "prior_window_audit": (
            str(args.prior_window_audit.resolve())
            if args.prior_window_audit
            else None
        ),
        "prior_window_audit_sha256": (
            _file_sha256(args.prior_window_audit)
            if args.prior_window_audit
            else None
        ),
        "exact_campaigns": [str(path.resolve()) for path in args.exact_campaign],
        "exact_campaign_sha256": [
            _file_sha256(path) for path in args.exact_campaign
        ],
        "source_sha256": {
            "evaluate_directed_window_grid.py": _file_sha256(Path(__file__)),
            "prove_top4_bottom2_directed.py": _file_sha256(PROVER),
            "four_stage_scheduler.py": _file_sha256(
                HERE / "four_stage_scheduler.py"
            ),
            "analyze_directed_case_classification.py": _file_sha256(
                HERE / "analyze_directed_case_classification.py"
            ),
        },
        "cases": names,
        "only_proven_targets": args.only_proven_targets,
        "require_all_proven_targets": args.require_all_proven_targets,
        "expected_cases": args.expected_cases,
        "reach_target_only": args.reach_target_only,
        "direct_only": args.direct_only,
        "windows": windows,
        "seed_beam_widths": args.seed_beam_widths,
        "seed_beam_modes": args.seed_beam_modes,
        "search_bound_policy": "proved_target_only_v1",
        "probe_isolation": "one_width_one_mode_per_process_v1",
    }
    config_id = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    work_dir = args.work_dir or (
        args.output.parent / ".window_fragments" / config_id
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / "manifest.json"
    manifest = {
        "config_id": config_id,
        "run_config": json.loads(json.dumps(config, sort_keys=True)),
    }
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != manifest:
            raise SystemExit(
                f"work-dir manifest mismatch: {work_dir}; use a new work-dir"
            )
    else:
        legacy_fragments = list(work_dir.glob("*.json"))
        if legacy_fragments:
            raise SystemExit(
                f"work-dir contains unmanifested fragments: {work_dir}; "
                "use a new work-dir"
            )
        _atomic_write(manifest_path, manifest)

    history_audit.WINDOWS = windows
    prior_direct = (
        _reuse_complete_direct_audit(
            args.prior_window_audit,
            names=names,
            windows=windows,
            target_proof_sha256=config["target_proof_sha256"],
            case_input_sha256=config["case_input_sha256"],
        )
        if args.prior_window_audit
        else None
    )
    direct_witnesses: dict[str, dict[str, list[dict]]] = {}
    for name in names:
        target = target_by_name[name]
        target_ticks = Fraction(target["best_reference_ticks"])
        candidates = list(witnesses_by_name.get(name, ()))
        if prior_direct is None:
            candidates.insert(0, (str(args.target_proof), target))
        by_window = (
            {
                window_name: list(sources)
                for window_name, sources in prior_direct[name].items()
            }
            if prior_direct is not None
            else {_window_name(window): [] for window in windows}
        )
        for source_path, candidate in candidates:
            if (
                not candidate.get("history_replay_valid")
                or not candidate.get("proven_optimal")
                or Fraction(candidate["best_reference_ticks"]) != target_ticks
            ):
                continue
            unchecked_windows = tuple(
                window
                for window in windows
                if not by_window[_window_name(window)]
            )
            if not unchecked_windows:
                break
            history_audit.WINDOWS = unchecked_windows
            audit = history_audit._audit_history(candidate)
            for window_name, compatible in audit["symmetry_window_compatible"].items():
                if compatible:
                    by_window[window_name].append(
                        {
                            "source": source_path,
                            "equal_load_id_relabel": not audit["window_compatible"][
                                window_name
                            ],
                        }
                    )
        history_audit.WINDOWS = windows
        # An unproved target history remains useful diagnostic visibility even
        # though it cannot be elevated to an optimal witness.
        if not target["proven_optimal"]:
            audit = history_audit._audit_history(target)
            for window_name, compatible in audit["symmetry_window_compatible"].items():
                if compatible:
                    by_window[window_name].append(
                        {
                            "source": str(args.target_proof),
                            "equal_load_id_relabel": not audit["window_compatible"][
                                window_name
                            ],
                        }
                    )
        direct_witnesses[name] = by_window
    try:
        beam_widths = tuple(
            int(field.strip())
            for field in args.seed_beam_widths.split(",")
            if field.strip()
        )
    except ValueError as exc:
        raise SystemExit("--seed-beam-widths must contain integers") from exc
    beam_modes = tuple(
        field.strip() for field in args.seed_beam_modes.split(",") if field.strip()
    )
    if not beam_widths or any(width <= 0 for width in beam_widths):
        raise SystemExit("--seed-beam-widths must be positive")
    if not beam_modes:
        raise SystemExit("--seed-beam-modes must not be empty")
    rows = []
    total = len(names) * len(windows)
    progress = 0
    for name in names:
        target = target_by_name[name]
        target_ticks = Fraction(target["best_reference_ticks"])
        for window in windows:
            progress += 1
            window_name = _window_name(window)
            direct_sources = direct_witnesses[name][window_name]
            direct = bool(direct_sources)
            exact = exact_by_key.get((name, window_name))
            if (
                direct
                and exact is not None
                and exact["status"] == "proved_insufficient_exact"
            ):
                raise SystemExit(
                    f"exact insufficiency contradicts a direct witness: "
                    f"{name} {window_name}"
                )
            # For pure window reachability, a replay-valid visible best-known
            # history is already a constructive witness even when its target
            # has not been proved globally optimal.  The default mode retains
            # the stronger behavior and keeps searching unproved targets.
            run_search = False if args.direct_only or exact is not None else (
                not direct if args.reach_target_only else not (
                    target["proven_optimal"] and direct
                )
            )
            restricted_row = None
            found = False
            probe_summaries = []
            if run_search:
                valid_rows = []
                for beam_width in beam_widths:
                    for beam_mode in beam_modes:
                        fragment = work_dir / (
                            f"{name}__t{window[0]}_b{window[1]}__"
                            f"w{beam_width}_{beam_mode}.json"
                        )
                        probe_row = None
                        resumed = False
                        if fragment.exists():
                            payload = json.loads(fragment.read_text(encoding="utf-8"))
                            if (
                                payload.get("complete")
                                and len(payload.get("cases", ())) == 1
                            ):
                                probe_row = payload["cases"][0]
                                resumed = True
                        if probe_row is None:
                            command = [
                                sys.executable,
                                str(PROVER),
                                "--case",
                                name,
                                "--time-limit-s",
                                "0",
                                "--seed-window",
                                f"{window[0]},{window[1]}",
                                "--seed-beam-widths",
                                str(beam_width),
                                "--seed-beam-modes",
                                beam_mode,
                                "--output",
                                str(fragment.resolve()),
                            ]
                            if args.case_input is not None:
                                command.extend(
                                    ["--case-input", str(args.case_input.resolve())]
                                )
                            # A proved optimum is used only as a numerical bound;
                            # window validity still requires a newly found history.
                            if target["proven_optimal"]:
                                command.extend(
                                    [
                                        "--prior-proof",
                                        str(args.target_proof.resolve()),
                                    ]
                                )
                            print(
                                f"[{progress}/{total}] search {name} {window_name} "
                                f"w={beam_width} mode={beam_mode}",
                                flush=True,
                            )
                            completed = subprocess.run(command, cwd=HERE, check=False)
                            if completed.returncode != 0:
                                raise SystemExit(
                                    f"{name} {window_name} w={beam_width} "
                                    f"mode={beam_mode} failed with exit code "
                                    f"{completed.returncode}; rerun to resume"
                                )
                            probe_row = json.loads(
                                fragment.read_text(encoding="utf-8")
                            )["cases"][0]
                        else:
                            print(
                                f"[{progress}/{total}] resume {name} {window_name} "
                                f"w={beam_width} mode={beam_mode}",
                                flush=True,
                            )
                        probe_found = any(
                            trial.get("window_history_found") is True
                            for trial in probe_row.get("seed_beam_trials", ())
                        )
                        probe_ticks = Fraction(probe_row["best_reference_ticks"])
                        probe_reaches_target = bool(
                            probe_found and probe_ticks <= target_ticks
                        )
                        probe_summaries.append(
                            {
                                "beam_width": beam_width,
                                "rank_mode": beam_mode,
                                "resumed": resumed,
                                "window_history_found": probe_found,
                                "window_reaches_target": probe_reaches_target,
                                "makespan_ticks": probe_row["best_reference_ticks"],
                                "expanded": sum(
                                    int(trial.get("expanded", 0))
                                    for trial in probe_row.get(
                                        "seed_beam_trials", ()
                                    )
                                ),
                                "generated": sum(
                                    int(trial.get("generated", 0))
                                    for trial in probe_row.get(
                                        "seed_beam_trials", ()
                                    )
                                ),
                                "runtime_s": probe_row["total_runtime_s"],
                            }
                        )
                        if probe_found:
                            valid_rows.append(probe_row)
                            if target["proven_optimal"] and probe_reaches_target:
                                break
                    if target["proven_optimal"] and any(
                        Fraction(row["best_reference_ticks"]) <= target_ticks
                        for row in valid_rows
                    ):
                        break
                if valid_rows:
                    restricted_row = min(
                        valid_rows,
                        key=lambda row: Fraction(row["best_reference_ticks"]),
                    )
                    found = True

            restricted_ticks = (
                Fraction(restricted_row["best_reference_ticks"])
                if restricted_row is not None and found
                else None
            )
            reaches = bool(
                direct
                or (
                    exact is not None
                    and exact["status"] == "proved_sufficient_exact"
                )
                or (restricted_ticks is not None and restricted_ticks <= target_ticks)
            )
            window_status = (
                "proved_sufficient_direct"
                if direct
                else exact["status"]
                if exact is not None
                else "heuristic_witness"
                if reaches
                else "unresolved"
            )
            row = {
                "name": name,
                "window": window_name,
                "top_entries": window[0],
                "bottom_entries": window[1],
                "total_entries": sum(window),
                "target_proven_optimal": bool(target["proven_optimal"]),
                "target_ticks": str(target_ticks),
                "direct_target_history_visible": direct,
                "direct_witness_sources": direct_sources,
                "direct_witness_uses_equal_load_id_relabel": any(
                    source["equal_load_id_relabel"] for source in direct_sources
                ),
                "exact_campaign_evidence": exact,
                "restricted_search_run": run_search,
                "restricted_window_history_found": found,
                "restricted_probe_summaries": probe_summaries,
                "restricted_ticks": (
                    str(restricted_ticks) if restricted_ticks is not None else None
                ),
                "window_minus_target_ticks": (
                    str(restricted_ticks - target_ticks)
                    if restricted_ticks is not None
                    else None
                ),
                "window_reaches_target": reaches,
                "window_status": window_status,
                "evidence": (
                    "saved_proven_optimal_history"
                    if target["proven_optimal"] and direct
                    else "restricted_exact_optimal_history"
                    if window_status == "proved_sufficient_exact"
                    else "restricted_exact_infeasible"
                    if window_status == "proved_insufficient_exact"
                    else "restricted_proven_optimal_history"
                    if target["proven_optimal"] and reaches
                    else "saved_unproven_best_known_history"
                    if direct
                    else "restricted_best_known_history"
                    if found
                    else "unresolved"
                ),
            }
            rows.append(row)
            _atomic_write(
                args.output,
                {
                    "schema": "directed_window_grid_v1",
                    "complete": False,
                    "run_config": config,
                    "summary": _summary(rows, windows),
                    "results": rows,
                },
            )

    result = {
        "schema": "directed_window_grid_v1",
        "complete": True,
        "run_config": config,
        "summary": _summary(rows, windows),
        "results": rows,
    }
    _atomic_write(args.output, result)
    if not args.keep_fragments:
        for fragment in work_dir.glob("*.json"):
            fragment.unlink()
        try:
            work_dir.rmdir()
            work_dir.parent.rmdir()
        except OSError:
            pass
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
