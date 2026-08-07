#!/usr/bin/env python3
"""Verify the top5+bottom1 plus 4+4 reserve refill protocol.

The scheduler stores two ordered banks:

* a hot bank with T0..T4 followed by four hotter-side reserve entries;
* a cold bank with B0 followed by four colder-side reserve entries.

The cold bank is stored cold-to-hot.  Software owns the unloaded middle and
returns top descriptors first, then bottom descriptors, in one or two 64-bit
refill beats.  This script replays the frozen proof traces and checks the
bounded-bank invariants independently of the RTL implementation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import verify_scheduler_rtl_unified_policy as datasets
from scheduler_rtl_distilled_profiles import COMPILED_PROFILES


HERE = Path(__file__).resolve().parent
PROOF_INPUT = HERE / "results/policy_search/olmoe_top2_projection_65_optimal_v1.json"
PROOF_RESULT = (
    HERE
    / "results/policy_search/"
    "bounded_top5_bottom1_fixed_lane_targeted_s4pf_certificate_validation.json"
)

TOP_VISIBLE = 5
TOP_RESERVE = 4
BOTTOM_VISIBLE = 1
BOTTOM_RESERVE = 4
TOP_CAPACITY = TOP_VISIBLE + TOP_RESERVE
BOTTOM_CAPACITY = BOTTOM_VISIBLE + BOTTOM_RESERVE
REFILL_LOW_WATER = 1
REFILL_QUAD_ENTRIES = 4
REFILL_MAX_ENTRIES = 6
TASK_COUNT_BITS = 8
DEPLOYMENT_TOKEN_LIMIT = 256

Descriptor = tuple[int, int]


@dataclass
class WindowState:
    remaining: list[Descriptor]
    top: list[Descriptor]
    bottom: list[Descriptor]  # cold-to-hot: B0 is bottom[0]

    @classmethod
    def initialize(cls, remaining: list[Descriptor]) -> "WindowState":
        top_count = min(TOP_CAPACITY, len(remaining))
        bottom_count = min(BOTTOM_CAPACITY, len(remaining) - top_count)
        top = list(remaining[:top_count])
        bottom = list(reversed(remaining[len(remaining) - bottom_count :]))
        state = cls(remaining=list(remaining), top=top, bottom=bottom)
        state.check()
        return state

    def loaded_eids(self) -> set[int]:
        return {eid for eid, _ntok in self.top + self.bottom}

    def hidden(self) -> list[Descriptor]:
        loaded = self.loaded_eids()
        return [entry for entry in self.remaining if entry[0] not in loaded]

    def visible(self) -> tuple[list[Descriptor], Descriptor | None]:
        visible_top = self.top[:TOP_VISIBLE]
        if self.bottom:
            bottom = self.bottom[0]
        elif self.top:
            bottom = self.top[-1]
        else:
            bottom = None
        return visible_top, bottom

    def check(self) -> None:
        expected = sorted(self.remaining, key=lambda entry: (-entry[1], entry[0]))
        if expected != self.remaining:
            raise AssertionError("remaining descriptors lost deterministic order")
        if len(self.top) > TOP_CAPACITY or len(self.bottom) > BOTTOM_CAPACITY:
            raise AssertionError("window bank capacity exceeded")
        loaded = self.top + self.bottom
        if len({eid for eid, _ntok in loaded}) != len(loaded):
            raise AssertionError("descriptor duplicated across window banks")
        if any(entry not in self.remaining for entry in loaded):
            raise AssertionError("loaded descriptor is no longer active")
        if self.top != self.remaining[: len(self.top)]:
            raise AssertionError("hot bank is not a remaining-list prefix")
        expected_bottom = list(reversed(self.remaining[-len(self.bottom) :]))
        if self.bottom and self.bottom != expected_bottom:
            raise AssertionError("cold bank is not a remaining-list suffix")
        visible_top, bottom = self.visible()
        if visible_top != self.remaining[: min(TOP_VISIBLE, len(self.remaining))]:
            raise AssertionError("T0..T4 window mismatch")
        expected_b0 = self.remaining[-1] if self.remaining else None
        if bottom != expected_b0:
            raise AssertionError("B0 window mismatch")

    def consume(self, consumed_eids: set[int]) -> None:
        if not consumed_eids:
            raise AssertionError("a scheduling round must consume work")
        visible_top, bottom = self.visible()
        visible_eids = {eid for eid, _ntok in visible_top}
        if bottom is not None:
            visible_eids.add(bottom[0])
        if not consumed_eids <= visible_eids:
            raise AssertionError(
                f"action selected hidden descriptors {sorted(consumed_eids-visible_eids)}"
            )

        self.remaining = [
            entry for entry in self.remaining if entry[0] not in consumed_eids
        ]
        self.top = [entry for entry in self.top if entry[0] not in consumed_eids]
        self.bottom = [
            entry for entry in self.bottom if entry[0] not in consumed_eids
        ]

        # Once the unloaded middle is empty, ownership moves only at the bank
        # boundary.  Pop the hottest cold entry into the hot-bank tail.
        if not self.hidden():
            desired_top = min(TOP_CAPACITY, len(self.remaining))
            while len(self.top) < desired_top and self.bottom:
                self.top.append(self.bottom.pop())
        self.check()

    def refill_request(self) -> tuple[int, int]:
        hidden = self.hidden()
        if not hidden:
            return 0, 0
        top_reserve = max(0, len(self.top) - TOP_VISIBLE)
        bottom_reserve = max(0, len(self.bottom) - BOTTOM_VISIBLE)
        top_low = top_reserve <= REFILL_LOW_WATER
        bottom_low = bottom_reserve <= REFILL_LOW_WATER
        if not (top_low or bottom_low):
            return 0, 0

        desired_top = min(TOP_CAPACITY, len(self.remaining))
        desired_bottom = min(
            BOTTOM_CAPACITY, len(self.remaining) - desired_top
        )
        top_deficit = max(0, desired_top - len(self.top))
        bottom_deficit = max(0, desired_bottom - len(self.bottom))
        # Near pointer convergence the same final middle descriptor would fill
        # either side's nominal deficit.  Give it to the hot side and let the
        # no-hidden boundary migration establish the canonical partition.
        top_count = min(top_deficit, len(hidden))
        bottom_count = min(bottom_deficit, len(hidden) - top_count)
        if top_count > REFILL_QUAD_ENTRIES or bottom_count > REFILL_QUAD_ENTRIES:
            raise AssertionError("one side of a refill exceeds one quad")
        if top_count + bottom_count > REFILL_MAX_ENTRIES:
            raise AssertionError("low-water refill exceeds two-beat bound")
        return top_count, bottom_count

    def apply_refill(self, top_count: int, bottom_count: int) -> None:
        if top_count > REFILL_QUAD_ENTRIES or bottom_count > REFILL_QUAD_ENTRIES:
            raise AssertionError("one side of a refill exceeds one quad")
        if top_count + bottom_count > REFILL_MAX_ENTRIES:
            raise AssertionError("refill response exceeds two-beat bound")
        hidden = self.hidden()
        if top_count + bottom_count > len(hidden):
            raise AssertionError("software refill pointers crossed")
        top_entries = hidden[:top_count]
        bottom_entries = list(reversed(hidden[len(hidden) - bottom_count :]))
        if set(top_entries) & set(bottom_entries):
            raise AssertionError("top and bottom refill descriptors alias")
        self.top.extend(top_entries)
        self.bottom.extend(bottom_entries)
        if not self.hidden():
            desired_top = min(TOP_CAPACITY, len(self.remaining))
            while len(self.top) < desired_top and self.bottom:
                self.top.append(self.bottom.pop())
        self.check()


def _sorted_distribution(distribution: dict[int, int]) -> list[Descriptor]:
    return sorted(
        (
            (int(eid), int(ntok))
            for eid, ntok in distribution.items()
            if int(ntok) > 0
        ),
        key=lambda entry: (-entry[1], entry[0]),
    )


def _verify_task_word_binding_contract() -> None:
    """Prove that two task-word flags encode every retained DMA binding.

    The owning cluster fixes the identity of a single lane.  An active stage
    therefore needs only one extra bit to distinguish that lane from BOTH;
    skip/cache fields already represent NONE.  S3 and S2PF are mutually
    exclusive and can share the late-stage flag.
    """
    for slot, token in enumerate(COMPILED_PROFILES):
        profile = token.physical
        for cluster, local in (("c2", "IDMA"), ("c3", "XDMA")):
            s1 = getattr(profile, f"{cluster}_dma_s1")
            s3 = getattr(profile, f"{cluster}_dma_s3")
            s2pf = getattr(profile, f"{cluster}_s2pf")
            for field, binding in (("S1", s1), ("S3", s3), ("S2PF", s2pf)):
                if binding not in {"NONE", local, "BOTH"}:
                    raise AssertionError(
                        f"profile {slot} {cluster} {field} has unencodable "
                        f"binding {binding}"
                    )
            if s3 != "NONE" and s2pf != "NONE":
                raise AssertionError(
                    f"profile {slot} {cluster} drives S3 and S2PF together"
                )

    # m_s2/m_s4 are ceil(tail/2).  At the deployment limit their maximum is
    # 128, so each count fits in the low eight bits and the old ninth bit is
    # available for the explicit-BOTH flag without changing the 64-bit word.
    maximum_count = (DEPLOYMENT_TOKEN_LIMIT + 1) // 2
    if maximum_count >= (1 << TASK_COUNT_BITS):
        raise AssertionError("deployment token limit does not fit task count bits")


def main() -> int:
    _verify_task_word_binding_contract()
    result = json.loads(PROOF_RESULT.read_text(encoding="utf-8"))
    jobs = {
        job["key"]: job
        for job in datasets._proof_jobs(PROOF_INPUT.resolve())
    }
    request_patterns: Counter[tuple[int, int]] = Counter()
    rounds = 0

    for key, row in result["rows"].items():
        window = WindowState.initialize(
            _sorted_distribution(dict(jobs[key]["distribution"]))
        )
        for step in row["trace"]:
            action = step["action"]
            consumed = {
                int(action[field])
                for field in ("c2_eid", "c3_eid")
                if int(action[field]) >= 0
            }
            window.consume(consumed)
            request = window.refill_request()
            if request != (0, 0):
                request_patterns[request] += 1
                window.apply_refill(*request)
            rounds += 1
        if window.remaining:
            raise AssertionError(f"{key}: protocol replay did not terminate")

    maximum = max((sum(request) for request in request_patterns), default=0)
    maximum_beats = (maximum + REFILL_QUAD_ENTRIES - 1) // REFILL_QUAD_ENTRIES
    print(
        "PASS distilled window protocol: "
        f"cases={len(result['rows'])} rounds={rounds} "
        f"max_refill={maximum} max_beats={maximum_beats} "
        f"profiles={len(COMPILED_PROFILES)} "
        f"task_count_bits={TASK_COUNT_BITS} "
        f"patterns={dict(sorted(request_patterns.items()))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
