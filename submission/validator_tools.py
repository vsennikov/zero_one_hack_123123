#!/usr/bin/env python3
"""
validator_tools.py — the validator as an evaluation tool.

Thin wrapper around generate_sequences.validate_sequence. It does NOT modify
the original validator. It adds the three things we actually need:

  1. is_valid(steps) -> bool -> (full-sequence check)
  2. validate_continuation(prefix,continuation) -> Report  -> (correct way to score a
    model's Task-2 output: glue prefix+tail, then report only tail-zone violations)
  3. make_labeled_negatives(...) -> list[Case] -> (break one rule on a valid
    sequence -> Task-3 data)

Key empirical facts:
  * On a VALID sequence, no rule false-fires at any truncation point. Global
    ordering rules are triggered by the "late" step, which in a valid sequence
    always sits after its anchor, so a prefix simply never contains the trigger
    before its anchor.
  * The real trap is validating a generated CONTINUATION on its own: look-back
    rules (e.g. RULE_DEP_NO_CLEAN) miss the clean/develop steps that live in the
    prefix and raise phantom violations. Always validate prefix+continuation
    together. validate_continuation() does this for you.

All rule IDs (for reference / Task-3 PREDICTED_RULE attribution):
  RULE_DEP_NO_CLEAN, RULE_METAL_ETCH_NO_LITHO, RULE_ETCH_NO_MASK,
  RULE_LITHO_LEVEL_SKIP, RULE_IMPLANT_NO_MASK, RULE_CMP_NO_DEP,
  RULE_PAD_OPEN_BEFORE_DEP, RULE_TEST_BEFORE_PASSIVATION,
  RULE_SHIP_BEFORE_TEST, RULE_BACKSIDE_BEFORE_PASSIVATION
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from generate_sequences import (
    validate_sequence,
    Violation,
    DEPOSITION_STEPS,
    CLEAN_STEPS,
    ETCH_STEPS,
    METAL_ETCH_STEPS,
    IMPLANT_STEPS,
    CMP_STEPS,
    ELECTRICAL_TEST_STEPS,
)

ALL_RULES = (
    "RULE_DEP_NO_CLEAN",
    "RULE_METAL_ETCH_NO_LITHO",
    "RULE_ETCH_NO_MASK",
    "RULE_LITHO_LEVEL_SKIP",
    "RULE_IMPLANT_NO_MASK",
    "RULE_CMP_NO_DEP",
    "RULE_PAD_OPEN_BEFORE_DEP",
    "RULE_TEST_BEFORE_PASSIVATION",
    "RULE_SHIP_BEFORE_TEST",
    "RULE_BACKSIDE_BEFORE_PASSIVATION",
)


# --------------------------------------------------------------------------- #
# 1. Basic full-sequence checks
# --------------------------------------------------------------------------- #

def is_valid(steps: list[str]) -> bool:
    """True iff the FULL sequence breaks none of the 10 rules."""
    return len(validate_sequence(steps)) == 0


def first_violation(steps: list[str]) -> Optional[Violation]:
    """The earliest violation (by step index), or None if valid.

    Useful for the cut-and-regenerate loop in part (B): it tells you the
    exact index from which to truncate and re-prompt the model.
    """
    v = validate_sequence(steps)
    if not v:
        return None
    return min(v, key=lambda x: x.step_index)


# --------------------------------------------------------------------------- #
# 2. Scoring a model's continuation correctly
# --------------------------------------------------------------------------- #

@dataclass
class ContinuationReport:
    valid: bool
    # violations whose offending step lies in the continuation (tail) zone:
    tail_violations: list[Violation] = field(default_factory=list)
    # violations located inside the given prefix (should normally be empty,
    # because eval prefixes are valid by construction; surfaced for debugging):
    prefix_violations: list[Violation] = field(default_factory=list)
    cut_index: int = 0

    def __str__(self) -> str:
        head = "VALID" if self.valid else f"INVALID ({len(self.tail_violations)} tail violation(s))"
        lines = [head]
        for v in self.tail_violations:
            lines.append(f"    {v}")
        return "\n".join(lines)


def validate_continuation(prefix: list[str], continuation: list[str]) -> ContinuationReport:
    """
    Correct way to validate a model's Task-2 output.

    The model is asked to predict only the steps AFTER the cut point. Validating
    that tail on its own gives phantom look-back violations. So we glue the
    given (valid) prefix back on, validate the whole thing, then keep only the
    violations whose step_index falls in the continuation zone.

    A continuation is "valid" iff it introduces no new violation. Violations
    that happen to sit inside the prefix are reported separately and do NOT
    count against the model (the eval prefix is valid by construction).
    """
    cut = len(prefix)
    full = list(prefix) + list(continuation)
    all_v = validate_sequence(full)
    tail = [v for v in all_v if v.step_index >= cut]
    head = [v for v in all_v if v.step_index < cut]
    return ContinuationReport(
        valid=(len(tail) == 0),
        tail_violations=tail,
        prefix_violations=head,
        cut_index=cut,
    )


def split_at_fraction(steps: list[str], fraction: float) -> tuple[list[str], list[str]]:
    """Split a full sequence into (prefix, continuation) at the given fraction,
    mirroring how eval_input_valid.csv is built (0.6 / 0.8)."""
    cut = max(1, int(round(len(steps) * fraction)))
    return steps[:cut], steps[cut:]


# --------------------------------------------------------------------------- #
# 3. Labeled negatives for Task 3 (anomaly detection)
# --------------------------------------------------------------------------- #

@dataclass
class NegativeCase:
    rule: str               # which rule was deliberately broken
    steps: list[str]        # the corrupted (invalid) sequence
    note: str = ""          # what edit was applied

    def confirm(self) -> bool:
        """True iff the corruption actually triggers (only) its target rule."""
        rules = {v.rule for v in validate_sequence(self.steps)}
        return self.rule in rules


# Each corruptor takes a VALID sequence and returns (new_steps, note) or None
# if the corruption is not applicable to this particular sequence. Corruptors
# aim to break exactly the one named rule by a minimal edit.

def _idx_of(steps, pred):
    return [i for i, s in enumerate(steps) if pred(s)]


def _break_dep_no_clean(steps, rng):
    # remove the clean step(s) in the 12-window before a deposition
    deps = _idx_of(steps, lambda s: s in DEPOSITION_STEPS)
    rng.shuffle(deps)
    for di in deps:
        clean_in_win = [j for j in range(max(0, di - 12), di) if steps[j] in CLEAN_STEPS]
        if clean_in_win:
            new = [s for k, s in enumerate(steps) if k not in set(clean_in_win)]
            return new, f"removed {len(clean_in_win)} clean step(s) before deposition idx {di}"
    return None


def _break_etch_no_mask(steps, rng):
    # remove the DEVELOP before a patterned etch
    etches = _idx_of(steps, lambda s: s in ETCH_STEPS)
    rng.shuffle(etches)
    for ei in etches:
        dev = [j for j in range(max(0, ei - 12), ei)
               if steps[j] in ("DEVELOP PHOTORESIST", "DEVELOP PAD WINDOW")]
        if dev:
            new = [s for k, s in enumerate(steps) if k not in set(dev)]
            return new, f"removed DEVELOP before etch idx {ei}"
    return None


def _break_metal_etch_no_litho(steps, rng):
    # remove EXPOSE LITHO before a metal etch
    me = _idx_of(steps, lambda s: s in METAL_ETCH_STEPS)
    rng.shuffle(me)
    for mi in me:
        exp = [j for j in range(max(0, mi - 15), mi) if steps[j].startswith("EXPOSE LITHO LEVEL")]
        if exp:
            new = [s for k, s in enumerate(steps) if k not in set(exp)]
            return new, f"removed EXPOSE LITHO before metal etch idx {mi}"
    return None


def _break_implant_no_mask(steps, rng):
    # remove the opener (oxide etch / develop) before an implant
    from generate_sequences import IMPLANT_OPENER_STEPS
    imps = _idx_of(steps, lambda s: s in IMPLANT_STEPS)
    rng.shuffle(imps)
    for ii in imps:
        op = [j for j in range(max(0, ii - 15), ii) if steps[j] in IMPLANT_OPENER_STEPS]
        if op:
            new = [s for k, s in enumerate(steps) if k not in set(op)]
            return new, f"removed implant opener before implant idx {ii}"
    return None


def _break_cmp_no_dep(steps, rng):
    from generate_sequences import FILL_STEPS
    cmps = _idx_of(steps, lambda s: s in CMP_STEPS)
    rng.shuffle(cmps)
    for ci in cmps:
        dep = [j for j in range(max(0, ci - 6), ci) if steps[j] in FILL_STEPS]
        if dep:
            new = [s for k, s in enumerate(steps) if k not in set(dep)]
            return new, f"removed fill/deposition before CMP idx {ci}"
    return None


def _break_litho_level_skip(steps, rng):
    # find an ALIGN MASK LEVEL N and bump it to N+2 to skip a level
    aligns = _idx_of(steps, lambda s: s.startswith("ALIGN MASK LEVEL ")
                     and s.split("ALIGN MASK LEVEL ")[1].isdigit())
    if len(aligns) < 2:
        return None
    # take the 2nd align, set its level to (prev_level + 2)
    i_prev, i_curr = aligns[0], aligns[1]
    prev_lvl = int(steps[i_prev].split("ALIGN MASK LEVEL ")[1])
    new = list(steps)
    new[i_curr] = f"ALIGN MASK LEVEL {prev_lvl + 2}"
    return new, f"jumped litho level to {prev_lvl + 2} at idx {i_curr}"


def _break_test_before_passivation(steps, rng):
    # move first electrical test to just before CURE PASSIVATION
    cure = next((i for i, s in enumerate(steps) if s == "CURE PASSIVATION"), None)
    test = next((i for i, s in enumerate(steps) if s in ELECTRICAL_TEST_STEPS), None)
    if cure is None or test is None or test < cure:
        return None
    new = list(steps)
    t = new.pop(test)
    new.insert(cure, t)  # now before cure
    return new, "moved electrical test before CURE PASSIVATION"


def _break_ship_before_test(steps, rng):
    ship = next((i for i, s in enumerate(steps) if s == "SHIP LOT"), None)
    sort = next((i for i, s in enumerate(steps) if s == "WAFER SORT TEST"), None)
    if ship is None or sort is None or ship < sort:
        return None
    new = list(steps)
    s = new.pop(ship)
    new.insert(sort, s)  # SHIP LOT now before WAFER SORT TEST
    return new, "moved SHIP LOT before WAFER SORT TEST"


def _break_backside_before_passivation(steps, rng):
    cure = next((i for i, s in enumerate(steps) if s == "CURE PASSIVATION"), None)
    bsm = next((i for i, s in enumerate(steps) if s == "DEPOSIT BACKSIDE METAL"), None)
    if cure is None or bsm is None or bsm < cure:
        return None
    new = list(steps)
    b = new.pop(bsm)
    new.insert(cure, b)
    return new, "moved DEPOSIT BACKSIDE METAL before CURE PASSIVATION"


def _break_pad_open_before_dep(steps, rng):
    from generate_sequences import PAD_WINDOW_STEPS
    dep = next((i for i, s in enumerate(steps)
                if s in ("DEPOSIT PASSIVATION", "DEPOSIT PASSIVATION LAYER")), None)
    pad = next((i for i, s in enumerate(steps) if s in PAD_WINDOW_STEPS), None)
    if dep is None or pad is None or pad < dep:
        return None
    new = list(steps)
    p = new.pop(pad)
    new.insert(dep, p)  # pad window now before passivation deposition
    return new, "moved pad-window step before DEPOSIT PASSIVATION"


_CORRUPTORS = {
    "RULE_DEP_NO_CLEAN": _break_dep_no_clean,
    "RULE_ETCH_NO_MASK": _break_etch_no_mask,
    "RULE_METAL_ETCH_NO_LITHO": _break_metal_etch_no_litho,
    "RULE_IMPLANT_NO_MASK": _break_implant_no_mask,
    "RULE_CMP_NO_DEP": _break_cmp_no_dep,
    "RULE_LITHO_LEVEL_SKIP": _break_litho_level_skip,
    "RULE_TEST_BEFORE_PASSIVATION": _break_test_before_passivation,
    "RULE_SHIP_BEFORE_TEST": _break_ship_before_test,
    "RULE_BACKSIDE_BEFORE_PASSIVATION": _break_backside_before_passivation,
    "RULE_PAD_OPEN_BEFORE_DEP": _break_pad_open_before_dep,
}


def make_negative(valid_steps: list[str], rule: str,
                  rng: Optional[random.Random] = None) -> Optional[NegativeCase]:
    """Break exactly one named rule on a valid sequence. Returns a confirmed
    NegativeCase, or None if the rule cannot be applied to this sequence."""
    if rule not in _CORRUPTORS:
        raise ValueError(f"Unknown rule {rule!r}. Known: {sorted(_CORRUPTORS)}")
    rng = rng or random.Random()
    res = _CORRUPTORS[rule](list(valid_steps), rng)
    if res is None:
        return None
    new_steps, note = res
    case = NegativeCase(rule=rule, steps=new_steps, note=note)
    return case if case.confirm() else None


def make_labeled_negatives(valid_sequences: list[list[str]],
                           rng: Optional[random.Random] = None,
                           per_rule: Optional[int] = None) -> list[NegativeCase]:
    """Given a pool of valid sequences, produce confirmed negatives spread over
    all 10 rules. If per_rule is None, makes one negative per (sequence, rule)
    where applicable."""
    rng = rng or random.Random()
    out: list[NegativeCase] = []
    counts = {r: 0 for r in _CORRUPTORS}
    pool = list(valid_sequences)
    rng.shuffle(pool)
    for steps in pool:
        for rule in _CORRUPTORS:
            if per_rule is not None and counts[rule] >= per_rule:
                continue
            case = make_negative(steps, rule, rng)
            if case is not None:
                out.append(case)
                counts[rule] += 1
    return out
