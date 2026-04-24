"""Flow-GRPO training on HumanEval (code generation) -- Phase 6.

Trains Qwen3.5-0.8B with LoRA to complete Python functions from docstrings,
evaluated by execution-based pass@1 — the same metric as run_humaneval.py.

Reward design:
  - execution_reward  (weight 3): 1.0 if generated function passes all HumanEval
                                  test cases (check(candidate) runs cleanly), else 0
  - format_reward     (weight 1): 1.0 for ```python block, 0.5 for any code block

Training task mirrors HumanEval exactly: given a function signature + docstring,
generate the function body. The GRPO loop uses the hidden test suite as reward signal.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

# ---------------------------------------------------------------------------
# HumanEval problem list (shared with run_humaneval.py)
# ---------------------------------------------------------------------------

# Try to import from benchmarks/ (works locally); if that fails in Modal,
# the full problem list is defined inline below as _ALL_PROBLEMS.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
try:
    from benchmarks.run_humaneval import HUMANEVAL_PROBLEMS as _imported_problems, _extract_python_code
    _IMPORT_OK = len(_imported_problems) > 0
except ImportError:
    _imported_problems = []
    _extract_python_code = None
    _IMPORT_OK = False

# Inline fallback problems used when benchmarks/ isn't mounted (smoke test).
_SMOKE_PROBLEMS = [
    {
        "task_id": "HumanEval/27",
        "prompt": (
            "def flip_case(string: str) -> str:\n"
            '    """ For a given string, flip lowercase characters to uppercase and uppercase to lowercase.\n'
            "    >>> flip_case('Hello')\n"
            "    'hELLO'\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate('') == ''\n"
            "    assert candidate('Hello!') == 'hELLO!'\n"
            "    assert candidate('These violent delights have violent ends') == "
            "'tHESE VIOLENT DELIGHTS HAVE VIOLENT ENDS'\n"
        ),
        "entry_point": "flip_case",
    },
    {
        "task_id": "HumanEval/23",
        "prompt": (
            "def strlen(string: str) -> int:\n"
            '    """ Return length of given string\n'
            "    >>> strlen('')\n"
            "    0\n"
            "    >>> strlen('abc')\n"
            "    3\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate('') == 0\n"
            "    assert candidate('x') == 1\n"
            "    assert candidate('asdfgh') == 6\n"
        ),
        "entry_point": "strlen",
    },
    {
        "task_id": "HumanEval/2",
        "prompt": (
            "def truncate_number(number: float) -> float:\n"
            '    """ Given a positive floating point number, return the decimal part.\n'
            "    >>> truncate_number(3.5)\n"
            "    0.5\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate(3.5) == 0.5\n"
            "    assert abs(candidate(1.33) - 0.33) < 1e-6\n"
            "    assert abs(candidate(123.456) - 0.456) < 1e-6\n"
        ),
        "entry_point": "truncate_number",
    },
    {
        "task_id": "HumanEval/13",
        "prompt": (
            "def greatest_common_divisor(a: int, b: int) -> int:\n"
            '    """ Return a greatest common divisor of two integers a and b\n'
            "    >>> greatest_common_divisor(3, 5)\n"
            "    1\n"
            "    >>> greatest_common_divisor(25, 15)\n"
            "    5\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate(3, 7) == 1\n"
            "    assert candidate(10, 15) == 5\n"
            "    assert candidate(49, 14) == 7\n"
            "    assert candidate(144, 60) == 12\n"
        ),
        "entry_point": "greatest_common_divisor",
    },
    {
        "task_id": "HumanEval/28",
        "prompt": (
            "from typing import List\n\n"
            "def concatenate(strings: List[str]) -> str:\n"
            '    """ Concatenate list of strings into a single string\n'
            "    >>> concatenate([])\n"
            "    ''\n"
            "    >>> concatenate(['a', 'b', 'c'])\n"
            "    'abc'\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate([]) == ''\n"
            "    assert candidate(['x', 'y', 'z']) == 'xyz'\n"
            "    assert candidate(['x', 'y', 'z', 'w', 'k']) == 'xyzwk'\n"
        ),
        "entry_point": "concatenate",
    },
    {
        "task_id": "HumanEval/16",
        "prompt": (
            "def count_distinct_characters(string: str) -> int:\n"
            '    """ Given a string, find out how many distinct characters (regardless of case) does it consist of\n'
            "    >>> count_distinct_characters('xyzXYZ')\n"
            "    3\n"
            "    >>> count_distinct_characters('Jerry')\n"
            "    4\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate('') == 0\n"
            "    assert candidate('abcde') == 5\n"
            "    assert candidate('abcde' + 'cade' + 'CADE') == 5\n"
            "    assert candidate('aaaaAAAAaaaa') == 1\n"
        ),
        "entry_point": "count_distinct_characters",
    },
    {
        "task_id": "HumanEval/15",
        "prompt": (
            "def string_sequence(n: int) -> str:\n"
            '    """ Return a string containing space-delimited numbers starting from 0 upto n inclusive.\n'
            "    >>> string_sequence(0)\n"
            "    '0'\n"
            "    >>> string_sequence(5)\n"
            "    '0 1 2 3 4 5'\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate(0) == '0'\n"
            "    assert candidate(3) == '0 1 2 3'\n"
            "    assert candidate(10) == '0 1 2 3 4 5 6 7 8 9 10'\n"
        ),
        "entry_point": "string_sequence",
    },
    {
        "task_id": "HumanEval/24",
        "prompt": (
            "def largest_divisor(n: int) -> int:\n"
            '    """ For a given number n, find the largest number that divides it evenly and is smaller than n\n'
            "    >>> largest_divisor(15)\n"
            "    5\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate(3) == 1\n"
            "    assert candidate(7) == 1\n"
            "    assert candidate(10) == 5\n"
            "    assert candidate(100) == 50\n"
            "    assert candidate(49) == 7\n"
        ),
        "entry_point": "largest_divisor",
    },
]

# Full HumanEval problem set (30 problems) used when benchmarks/ import fails in Modal.
_ALL_PROBLEMS = _SMOKE_PROBLEMS + [
    {
        "task_id": "HumanEval/0",
        "prompt": (
            "from typing import List\n\n"
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            '    """ Check if in given list of numbers, are any two numbers closer to each other than\n'
            "    given threshold.\n"
            "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n"
            "    False\n"
            "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n"
            "    True\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n"
            "    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n"
            "    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n"
            "    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n"
            "    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n"
        ),
        "entry_point": "has_close_elements",
    },
    {
        "task_id": "HumanEval/1",
        "prompt": (
            "from typing import List\n\n"
            "def separate_paren_groups(paren_string: str) -> List[str]:\n"
            '    """ Input to this function is a string containing multiple groups of nested parentheses.\n'
            "    Your goal is to separate those group into separate strings and return the list of those.\n"
            "    Separate groups are balanced and not nested within each other. Ignore spaces.\n"
            "    >>> separate_paren_groups('( ) (( )) (( )( ))')\n"
            "    ['()', '(())', '(()())']\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate('(()()) ((())) () ((())()())') == ['(()())', '((()))', '()', '((())()())']\n"
            "    assert candidate('() (()) ((())) (((())))') == ['()', '(())', '((()))', '(((())))']\n"
            "    assert candidate('(()(())((())))') == ['(()(())((())))']\n"
        ),
        "entry_point": "separate_paren_groups",
    },
    {
        "task_id": "HumanEval/3",
        "prompt": (
            "from typing import List\n\n"
            "def below_zero(operations: List[int]) -> bool:\n"
            '    """ You\'re given a list of deposit and withdrawal operations on a bank account\n'
            "    that starts with zero balance. Detect if at any point the balance falls below zero.\n"
            "    >>> below_zero([1, 2, 3])\n"
            "    False\n"
            "    >>> below_zero([1, 2, -4, 5])\n"
            "    True\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate([]) == False\n"
            "    assert candidate([1, 2, -3, 1, 2, -3]) == False\n"
            "    assert candidate([1, 2, -4, 5, 6]) == True\n"
            "    assert candidate([1, -1, 2, -2, 5, -5, 4, -4]) == False\n"
            "    assert candidate([1, -1, 2, -2, 5, -5, 4, -5]) == True\n"
        ),
        "entry_point": "below_zero",
    },
    {
        "task_id": "HumanEval/4",
        "prompt": (
            "from typing import List\n\n"
            "def mean_absolute_deviation(numbers: List[float]) -> float:\n"
            '    """ For a given list of input numbers, calculate Mean Absolute Deviation\n'
            "    around the mean of this dataset.\n"
            "    MAD = average | x - x_mean |\n"
            "    >>> mean_absolute_deviation([1.0, 2.0, 3.0, 4.0])\n"
            "    1.0\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert abs(candidate([1.0, 2.0, 3.0]) - 2.0/3.0) < 1e-6\n"
            "    assert abs(candidate([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6\n"
            "    assert abs(candidate([1.0, 2.0, 3.0, 4.0, 5.0]) - 6.0/5.0) < 1e-6\n"
        ),
        "entry_point": "mean_absolute_deviation",
    },
    {
        "task_id": "HumanEval/5",
        "prompt": (
            "from typing import List\n\n"
            "def intersperse(numbers: List[int], delimeter: int) -> List[int]:\n"
            '    """ Insert a number \'delimeter\' between every two consecutive elements of input list.\n'
            "    >>> intersperse([], 4)\n"
            "    []\n"
            "    >>> intersperse([1, 2, 3], 4)\n"
            "    [1, 4, 2, 4, 3]\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate([], 7) == []\n"
            "    assert candidate([5, 6, 3, 2], 8) == [5, 8, 6, 8, 3, 8, 2]\n"
            "    assert candidate([2, 2, 2], 2) == [2, 2, 2, 2, 2]\n"
        ),
        "entry_point": "intersperse",
    },
    {
        "task_id": "HumanEval/7",
        "prompt": (
            "from typing import List\n\n"
            "def filter_by_substring(strings: List[str], substring: str) -> List[str]:\n"
            '    """ Filter an input list of strings only for ones that contain given substring.\n'
            "    >>> filter_by_substring([], 'a')\n"
            "    []\n"
            "    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')\n"
            "    ['abc', 'bacd', 'array']\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate([], 'john') == []\n"
            "    assert candidate(['xxx', 'asd', 'xxy', 'john doe', 'xxxAAA', 'xxx'], 'xxx') == ['xxx', 'xxxAAA', 'xxx']\n"
            "    assert candidate(['xxx', 'asd', 'xxy', 'john doe', 'xxxAAA', 'xxx'], 'john') == ['john doe']\n"
        ),
        "entry_point": "filter_by_substring",
    },
    {
        "task_id": "HumanEval/8",
        "prompt": (
            "from typing import List, Tuple\n\n"
            "def sum_product(numbers: List[int]) -> Tuple[int, int]:\n"
            '    """ Return a tuple of sum and product of all integers in a list.\n'
            "    Empty sum should be 0 and empty product should be 1.\n"
            "    >>> sum_product([])\n"
            "    (0, 1)\n"
            "    >>> sum_product([1, 2, 3, 4])\n"
            "    (10, 24)\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate([]) == (0, 1)\n"
            "    assert candidate([1, 1, 1]) == (3, 1)\n"
            "    assert candidate([100, 0]) == (100, 0)\n"
            "    assert candidate([3, 5, 7]) == (15, 105)\n"
            "    assert candidate([10]) == (10, 10)\n"
        ),
        "entry_point": "sum_product",
    },
    {
        "task_id": "HumanEval/9",
        "prompt": (
            "from typing import List\n\n"
            "def rolling_max(numbers: List[int]) -> List[int]:\n"
            '    """ Generate a list of rolling maximum element found until given moment in the sequence.\n'
            "    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])\n"
            "    [1, 2, 3, 3, 3, 4, 4]\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate([]) == []\n"
            "    assert candidate([1, 2, 3, 4]) == [1, 2, 3, 4]\n"
            "    assert candidate([4, 3, 2, 1]) == [4, 4, 4, 4]\n"
            "    assert candidate([3, 2, 3, 100, 3]) == [3, 3, 3, 100, 100]\n"
        ),
        "entry_point": "rolling_max",
    },
    {
        "task_id": "HumanEval/11",
        "prompt": (
            "def string_xor(a: str, b: str) -> str:\n"
            '    """ Input are two strings of 1s and 0s. Perform binary XOR and return result as string.\n'
            "    >>> string_xor('010', '110')\n"
            "    '100'\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate('111000', '101010') == '010010'\n"
            "    assert candidate('1', '1') == '0'\n"
            "    assert candidate('0101', '0000') == '0101'\n"
        ),
        "entry_point": "string_xor",
    },
    {
        "task_id": "HumanEval/12",
        "prompt": (
            "from typing import List, Optional\n\n"
            "def longest(strings: List[str]) -> Optional[str]:\n"
            '    """ Out of list of strings, return the longest one. Return the first one in case of multiple\n'
            "    strings of same length. Return None if input list is empty.\n"
            "    >>> longest([])\n"
            "    >>> longest(['a', 'b', 'c'])\n"
            "    'a'\n"
            "    >>> longest(['a', 'bb', 'ccc'])\n"
            "    'ccc'\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate([]) is None\n"
            "    assert candidate(['x', 'y', 'z']) == 'x'\n"
            "    assert candidate(['x', 'yyy', 'zzzz', 'www', 'kkkk', 'abc']) == 'zzzz'\n"
        ),
        "entry_point": "longest",
    },
    {
        "task_id": "HumanEval/14",
        "prompt": (
            "from typing import List\n\n"
            "def all_prefixes(string: str) -> List[str]:\n"
            '    """ Return list of all prefixes from shortest to longest of the input string.\n'
            "    >>> all_prefixes('abc')\n"
            "    ['a', 'ab', 'abc']\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate('') == []\n"
            "    assert candidate('asdfgh') == ['a', 'as', 'asd', 'asdf', 'asdfg', 'asdfgh']\n"
            "    assert candidate('www') == ['w', 'ww', 'www']\n"
        ),
        "entry_point": "all_prefixes",
    },
    {
        "task_id": "HumanEval/18",
        "prompt": (
            "def how_many_times(string: str, substring: str) -> int:\n"
            '    """ Find how many times a given substring can be found in the original string. Count overlapping cases.\n'
            "    >>> how_many_times('', 'a')\n"
            "    0\n"
            "    >>> how_many_times('aaa', 'a')\n"
            "    3\n"
            "    >>> how_many_times('aaaa', 'aa')\n"
            "    3\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate('', 'x') == 0\n"
            "    assert candidate('xyxyxyx', 'x') == 4\n"
            "    assert candidate('cacacacac', 'cac') == 4\n"
            "    assert candidate('john doe', 'john') == 1\n"
        ),
        "entry_point": "how_many_times",
    },
    {
        "task_id": "HumanEval/20",
        "prompt": (
            "from typing import List, Tuple\n\n"
            "def find_closest_elements(numbers: List[float]) -> Tuple[float, float]:\n"
            '    """ From a supplied list of numbers select and return two that are the closest to each other\n'
            "    in order (smaller number, larger number).\n"
            "    >>> find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0])\n"
            "    (1.0, 2.0)\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2]) == (3.9, 4.0)\n"
            "    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0]) == (5.0, 5.9)\n"
            "    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.2]) == (2.0, 2.2)\n"
            "    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0]) == (2.0, 2.0)\n"
        ),
        "entry_point": "find_closest_elements",
    },
    {
        "task_id": "HumanEval/21",
        "prompt": (
            "from typing import List\n\n"
            "def rescale_to_unit(numbers: List[float]) -> List[float]:\n"
            '    """ Apply a linear transform so the smallest number becomes 0 and the largest becomes 1.\n'
            "    >>> rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0])\n"
            "    [0.0, 0.25, 0.5, 0.75, 1.0]\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate([2.0, 49.9]) == [0.0, 1.0]\n"
            "    assert candidate([100.0, 49.9]) == [1.0, 0.0]\n"
            "    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0]) == [0.0, 0.25, 0.5, 0.75, 1.0]\n"
        ),
        "entry_point": "rescale_to_unit",
    },
    {
        "task_id": "HumanEval/26",
        "prompt": (
            "from typing import List\n\n"
            "def remove_duplicates(numbers: List[int]) -> List[int]:\n"
            '    """ From a list of integers, remove all elements that occur more than once.\n'
            "    Keep order of elements left the same as in the input.\n"
            "    >>> remove_duplicates([1, 2, 3, 2, 4])\n"
            "    [1, 3, 4]\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate([]) == []\n"
            "    assert candidate([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]\n"
            "    assert candidate([1, 2, 3, 2, 4, 3, 5]) == [1, 4, 5]\n"
        ),
        "entry_point": "remove_duplicates",
    },
    {
        "task_id": "HumanEval/30",
        "prompt": (
            "def get_positive(l: list) -> list:\n"
            '    """ Return only positive numbers in the list.\n'
            "    >>> get_positive([-1, 2, -4, 3, 5])\n"
            "    [2, 3, 5]\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate([-1, -2, 4, 5, 6]) == [4, 5, 6]\n"
            "    assert candidate([5, 3, -5, 2, 3, 3, 9, 0, 123, 1, -10]) == [5, 3, 2, 3, 3, 9, 123, 1]\n"
            "    assert candidate([-1, -2]) == []\n"
            "    assert candidate([]) == []\n"
        ),
        "entry_point": "get_positive",
    },
]

# What the training will actually use:
#   - full import succeeded  → _imported_problems (30 problems from benchmarks/)
#   - import failed (Modal)  → _ALL_PROBLEMS (30 problems defined inline above)
#   - smoke_test flag        → _SMOKE_PROBLEMS (8 problems, fast iteration)
HUMANEVAL_PROBLEMS = _imported_problems if _IMPORT_OK else _ALL_PROBLEMS

# ---------------------------------------------------------------------------
# Prompt / constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Think through the problem inside <think> </think> tags, "
    "then provide the complete Python function implementation "
    "inside a ```python ... ``` code block."
)

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
PREAMBLE = "import random, math\nfrom typing import List, Tuple, Optional, Dict\n\n"

# In-process exec() on model-generated code can hang forever (infinite loops).
# Run tests in a subprocess with a hard timeout — same idea as benchmarks/run_humaneval.py.
_EXEC_TIMEOUT_SEC = 10


def _make_prompt(problem: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            "Complete the following Python function. "
            "Provide the FULL function definition (including the signature) "
            "inside a ```python ... ``` block.\n\n"
            f"{problem['prompt']}"
        )},
    ]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_humaneval_data(smoke_test: bool = False) -> tuple[Dataset, Dataset]:
    problems = _SMOKE_PROBLEMS if smoke_test else HUMANEVAL_PROBLEMS

    records = [
        {
            "prompt": _make_prompt(p),
            "test_code": p["test"],
            "entry_point": p["entry_point"],
            "he_prompt": p["prompt"],    # original HumanEval prompt for body-only fallback
        }
        for p in problems
    ]

    import random as _random
    _random.seed(42)
    _random.shuffle(records)

    split = max(1, int(len(records) * 0.85))
    train_ds = Dataset.from_list(records[:split])
    eval_ds = Dataset.from_list(records[split:])
    print(f"HumanEval GRPO: {len(train_ds)} train, {len(eval_ds)} eval")
    return train_ds, eval_ds


# ---------------------------------------------------------------------------
# Code extraction (local fallback if import failed)
# ---------------------------------------------------------------------------

def _local_extract(text: str, entry_point: str) -> str:
    """Extract function definition from model output."""
    # 1. ```python block containing the function
    blocks = CODE_BLOCK_RE.findall(text)
    for block in reversed(blocks):
        if f"def {entry_point}" in block:
            return block.strip()
    if blocks:
        return blocks[-1].strip()

    # 2. Direct def line
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(rf"\s*def\s+{re.escape(entry_point)}\s*\(", line):
            return textwrap.dedent("\n".join(lines[i:])).strip()

    # 3. After </think>
    parts = re.split(r"</think>", text, flags=re.IGNORECASE)
    candidate = parts[-1].strip() if len(parts) > 1 else text.strip()
    candidate = re.sub(r"^```[a-z]*\n?", "", candidate).rstrip("`").strip()
    return candidate


def _extract(text: str, entry_point: str) -> str:
    if _extract_python_code is not None:
        return _extract_python_code(text, entry_point)
    return _local_extract(text, entry_point)


# ---------------------------------------------------------------------------
# Execution-based evaluation
# ---------------------------------------------------------------------------

def _run_humaneval_test(completion: str, he_prompt: str, test_code: str, entry_point: str) -> bool:
    """Execute completion against HumanEval test suite. Returns True if all pass."""
    code = _extract(completion, entry_point)
    if not code:
        return False

    # Determine if we have a full function definition or just the body
    if f"def {entry_point}" in code:
        function_section = code
    else:
        function_section = he_prompt + textwrap.indent(code, "    ")

    full_src = PREAMBLE + function_section + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"

    try:
        ast.parse(full_src)
    except SyntaxError:
        return False

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(full_src)
            tmp_path = f.name
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=_EXEC_TIMEOUT_SEC,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def _completion_text(c) -> str:
    if isinstance(c, list):
        return "".join(msg.get("content", "") for msg in c)
    if isinstance(c, dict):
        return c.get("content", "")
    return str(c)


def execution_reward(completions, test_code, entry_point, he_prompt, **kwargs):
    """1.0 if generated function passes all HumanEval tests, else 0.0."""
    rewards = []
    for c, tests, ep, hep in zip(completions, test_code, entry_point, he_prompt):
        text = _completion_text(c)
        passed = _run_humaneval_test(text, hep, tests, ep)
        rewards.append(1.0 if passed else 0.0)
    return rewards


def format_reward(completions, **kwargs):
    """1.0 for ```python block, 0.5 for any ``` block, 0.25 for <think> only."""
    rewards = []
    for c in completions:
        text = _completion_text(c)
        if re.search(r"```python", text, re.IGNORECASE):
            rewards.append(1.0)
        elif "```" in text:
            rewards.append(0.5)
        elif THINK_RE.search(text):
            rewards.append(0.25)
        else:
            rewards.append(0.0)
    return rewards


# ---------------------------------------------------------------------------
# Checkpoint evaluation
# ---------------------------------------------------------------------------

def evaluate(model, tokenizer, dataset, max_new_tokens: int = 512, num_samples: int = 20):
    model.eval()
    subset = dataset.select(range(min(num_samples, len(dataset))))
    passed = 0
    fmt_ok = 0
    total = 0

    for ex in subset:
        prompt_text = tokenizer.apply_chat_template(
            ex["prompt"], tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.2,
                top_p=0.95,
            )
        completion = tokenizer.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True,
        )

        if _run_humaneval_test(completion, ex["he_prompt"], ex["test_code"], ex["entry_point"]):
            passed += 1
        if re.search(r"```(?:python)?", completion, re.IGNORECASE):
            fmt_ok += 1
        total += 1

    n = max(total, 1)
    results = {
        "pass_at_1": passed / n,
        "format_rate": fmt_ok / n,
        "n_samples": total,
    }
    print(f"  Eval ({total}): pass@1={results['pass_at_1']:.3f}  format={results['format_rate']:.1%}")
    return results


# ---------------------------------------------------------------------------
# Logging callback
# ---------------------------------------------------------------------------

class LogCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        r = logs.get("reward")
        exec_r = logs.get("rewards/execution_reward/mean")
        fmt_r = logs.get("rewards/format_reward/mean")
        if r is not None:
            print(
                f"[step {state.global_step:>4}] "
                f"reward={r:.4f}  exec={exec_r or 0:.4f}  fmt={fmt_r or 0:.4f}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    model_name = "Qwen/Qwen3.5-0.8B"
    smoke_test = os.environ.get("SMOKE_TEST", "0") == "1"
    output_dir = "/runs/flow_grpo_humaneval_smoke" if smoke_test else "/runs/flow_grpo_humaneval"

    print(f"Model:      {model_name}")
    print(f"Smoke test: {smoke_test}")
    print(f"Output dir: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_data, eval_data = load_humaneval_data(smoke_test=smoke_test)

    # Phase 1: Baseline
    print("\n=== Phase 1: Baseline Evaluation ===")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    baseline = evaluate(base_model, tokenizer, eval_data)
    del base_model
    torch.cuda.empty_cache()

    # Phase 2: Flow-GRPO with LoRA
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    print("\n=== Phase 2: Flow-GRPO Training with LoRA ===")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    training_args = GRPOConfig(
        output_dir=output_dir,
        max_steps=8 if smoke_test else 100,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        num_generations=8,
        max_completion_length=512,
        beta=0.001,
        logging_steps=1,
        bf16=True,
        gradient_checkpointing=False,   # kernel 4.4 hangs with checkpointing enabled
        dataloader_num_workers=0,       # avoid fork + CUDA deadlock on Modal
        dataloader_pin_memory=False,
        save_strategy="steps",
        save_steps=10 if smoke_test else 25,
        save_total_limit=6,
        report_to="none",
        reward_weights=[3.0, 1.0],
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=[execution_reward, format_reward],
        args=training_args,
        train_dataset=train_data,
        peft_config=peft_config,
    )
    trainer.add_callback(LogCallback())
    trainer.train()
    del trainer
    torch.cuda.empty_cache()

    # Phase 3: Evaluate checkpoints
    print("\n=== Phase 3: Evaluate Checkpoints ===")
    ckpt_dirs = sorted(
        [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
        key=lambda d: int(d.split("-")[1]),
    )

    results = {}
    for cn in ckpt_dirs:
        cp = os.path.join(output_dir, cn)
        step = int(cn.split("-")[1])
        print(f"\nEvaluating {cn}...")
        bm = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto",
        )
        model = PeftModel.from_pretrained(bm, cp).merge_and_unload()
        results[step] = evaluate(model, tokenizer, eval_data)
        del model, bm
        torch.cuda.empty_cache()

    best_step = max(results, key=lambda s: results[s]["pass_at_1"]) if results else None
    if best_step is not None:
        best = results[best_step]
        print(f"\nBest checkpoint: step {best_step}")
        print(f"  pass@1:      {best['pass_at_1']:.3f} (baseline {baseline['pass_at_1']:.3f})")
        print(f"  format_rate: {best['format_rate']:.1%}")

    final = {
        "model": model_name,
        "baseline": baseline,
        "checkpoints": {str(k): v for k, v in results.items()},
        "best_step": best_step,
    }
    results_path = os.path.join(output_dir, "humaneval_grpo_results.json")
    with open(results_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
