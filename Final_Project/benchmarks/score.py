"""Unified scoring utilities for all AgentFlow benchmarks."""

from __future__ import annotations

import math
import re
import string
from collections import Counter


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _extract_answer_tag(s: str) -> str:
    """Pull text from <answer>...</answer> tags if present."""
    m = re.search(r"<answer>(.*?)</answer>", s, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _normalize_text(s: str) -> str:
    """Lower-case, strip articles / punctuation / whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def _extract_number(s: str) -> float | None:
    """Try to pull a single number from a string."""
    m = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    if m:
        return float(m.group())
    return None


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def exact_match(pred: str, gold: str | list[str]) -> float:
    """Case-insensitive exact match.  Gold may be a list of acceptable answers."""
    golds = gold if isinstance(gold, list) else [gold]
    pred_n = _normalize_text(pred)
    return float(any(_normalize_text(g) == pred_n for g in golds))


def f1_score(pred: str, gold: str | list[str]) -> float:
    """Token-level F1 (max over gold list)."""
    golds = gold if isinstance(gold, list) else [gold]
    pred_toks = _normalize_text(pred).split()
    best = 0.0
    for g in golds:
        gold_toks = _normalize_text(g).split()
        common = Counter(pred_toks) & Counter(gold_toks)
        n_common = sum(common.values())
        if n_common == 0:
            continue
        prec = n_common / len(pred_toks) if pred_toks else 0
        rec = n_common / len(gold_toks) if gold_toks else 0
        best = max(best, 2 * prec * rec / (prec + rec) if (prec + rec) else 0)
    return best


def numeric_match(pred: str, gold: str | int | float, tol: float = 1e-3) -> float:
    """Check numeric equality within tolerance."""
    p = _extract_number(pred)
    g = float(gold) if not isinstance(gold, float) else gold
    if p is None:
        return 0.0
    return float(math.isclose(p, g, abs_tol=tol))


def mc_accuracy(pred: str, gold: str) -> float:
    """Multiple-choice accuracy -- extract first letter A-E from pred."""
    pred_letter = ""
    m = re.search(r"\b([A-E])\b", pred.upper())
    if m:
        pred_letter = m.group(1)
    gold_letter = ""
    m2 = re.search(r"\b([A-E])\b", gold.upper())
    if m2:
        gold_letter = m2.group(1)
    if not pred_letter or not gold_letter:
        return exact_match(pred, gold)
    return float(pred_letter == gold_letter)


# ---------------------------------------------------------------------------
# Per-benchmark scoring dispatch
# ---------------------------------------------------------------------------

SEARCH_BENCHMARKS = {"bamboogle", "2wiki", "hotpotqa", "musique"}
NUMERIC_MATH = {"aime24", "amc23"}
SCIENCE_MC = {"gpqa", "medqa"}
AGENTIC = {"gaia"}


def score_sample(benchmark: str, pred: str, sample: dict) -> dict[str, float]:
    """Return ``{metric_name: value}`` for a single prediction."""
    # Try to extract from <answer> tags first
    answer_tag = _extract_answer_tag(pred)
    if answer_tag:
        pred = answer_tag
    gold = sample.get("answer", "")

    if benchmark in SEARCH_BENCHMARKS or benchmark in AGENTIC:
        return {
            "em": exact_match(pred, gold),
            "f1": f1_score(pred, gold),
        }
    elif benchmark in NUMERIC_MATH:
        return {"accuracy": numeric_match(pred, gold)}
    elif benchmark == "gameof24":
        # Gold is a list of equivalent expression strings, not a single number.
        return {"accuracy": exact_match(pred, gold)}
    elif benchmark in SCIENCE_MC:
        return {"accuracy": mc_accuracy(pred, gold)}
    else:
        return {"em": exact_match(pred, gold), "f1": f1_score(pred, gold)}
