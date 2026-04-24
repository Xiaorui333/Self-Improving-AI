#!/usr/bin/env python3
"""Run AgentFlow on HumanEval (code generation benchmark).

HumanEval is fundamentally different from the paper benchmarks:
- Task type: generate a Python function body from a docstring
- Evaluation: deterministic code execution (pass@1), not LLM judge
- Measures: code synthesis & debugging capability, not knowledge retrieval
- AgentFlow's PythonCoderTool lets the agent iterate on its own solution

Dataset: openai/openai_humaneval (164 problems)
  Each problem: prompt (docstring + signature) -> model completes -> run test cases.

Usage:
    python3.11 benchmarks/run_humaneval.py --model qwen2.5-7b --sample_size 20
    python3.11 benchmarks/run_humaneval.py --models all --sample_size 20
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import textwrap
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentflow.config import MODELS, PORTKEY_API_KEY
from agentflow.solver import Solver

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
HUMANEVAL_DIR = os.path.join(os.path.dirname(__file__), "data", "humaneval")

# ──────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────────────────────

HUMANEVAL_PROBLEMS: list[dict] = [
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
            "    Separate groups are balanced (each open brace is properly closed) and not nested within each other.\n"
            "    Ignore any spaces in the input string.\n"
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
        "task_id": "HumanEval/2",
        "prompt": (
            "def truncate_number(number: float) -> float:\n"
            '    """ Given a positive floating point number, it can be decomposed into\n'
            "    and integer part (largest integer smaller than given number) and decimals\n"
            "    (leftover part always smaller than 1).\n"
            "    Return the decimal part of the number.\n"
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
        "task_id": "HumanEval/3",
        "prompt": (
            "from typing import List\n\n"
            "def below_zero(operations: List[int]) -> bool:\n"
            '    """ You\'re given a list of deposit and withdrawal operations on a bank account that starts with\n'
            "    zero balance. Your task is to detect if at any point the balance of account falls below zero, and\n"
            "    at that point function should return True. Otherwise it should return False.\n"
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
            "    Mean Absolute Deviation is the average absolute difference between each\n"
            "    element and a centerpoint (mean in this case):\n"
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
            '    """ Insert a number \'delimeter\' between every two consecutive elements of input list `numbers\'\n'
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
        "task_id": "HumanEval/6",
        "prompt": (
            "from typing import List\n\n"
            "def parse_nested_parens(paren_string: str) -> List[int]:\n"
            '    """ Input to this function is a string represented multiple groups for nested parentheses separated by spaces.\n'
            "    For each of the group, output the deepest level of nesting of parentheses.\n"
            "    E.g. (()()) has maximum two levels of nesting while ((())) has three.\n"
            "    >>> parse_nested_parens('(()()) ((())) () ((())()())')\n"
            "    [2, 3, 1, 3]\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate('(()()) ((())) () ((())()())') == [2, 3, 1, 3]\n"
            "    assert candidate('() (()) ((())) (((())))') == [1, 2, 3, 4]\n"
            "    assert candidate('(()(())((())))') == [4]\n"
        ),
        "entry_point": "parse_nested_parens",
    },
    {
        "task_id": "HumanEval/7",
        "prompt": (
            "from typing import List\n\n"
            "def filter_by_substring(strings: List[str], substring: str) -> List[str]:\n"
            '    """ Filter an input list of strings only for ones that contain given substring\n'
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
            '    """ For a given list of integers, return a tuple consisting of a sum and a product of all the integers in a list.\n'
            "    Empty sum should be equal to 0 and empty product should be equal to 1.\n"
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
            "from typing import List, Tuple\n\n"
            "def rolling_max(numbers: List[int]) -> List[int]:\n"
            '    """ From a given list of integers, generate a list of rolling maximum element found until given moment\n'
            "    in the sequence.\n"
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
            '    """ Input are two strings a and b consisting only of 1s and 0s.\n'
            "    Perform binary XOR on these inputs and return result also as a string.\n"
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
            "    strings of the same length. Return None in case the input list is empty.\n"
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
        "task_id": "HumanEval/14",
        "prompt": (
            "from typing import List\n\n"
            "def all_prefixes(string: str) -> List[str]:\n"
            '    """ Return list of all prefixes from shortest to longest of the input string\n'
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
        "task_id": "HumanEval/17",
        "prompt": (
            "from typing import List\n\n"
            "def parse_music(music_string: str) -> List[int]:\n"
            '    """ Input to this function is a string representing musical notes in a special ASCII format.\n'
            "    Your task is to parse this string and return list of integers corresponding to how many beats does each\n"
            "    not last.\n"
            "    Here is a legend:\n"
            "    'o' - whole note, lasts four beats\n"
            "    'o|' - half note, lasts two beats\n"
            "    '.|' - quater note, lasts one beat\n"
            "    >>> parse_music('o o| .| o| o| .| .| .| .| o o')\n"
            "    [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate('') == []\n"
            "    assert candidate('o o o o') == [4, 4, 4, 4]\n"
            "    assert candidate('.| .| .| .|') == [1, 1, 1, 1]\n"
            "    assert candidate('o| o| .| .| o o o o') == [2, 2, 1, 1, 4, 4, 4, 4]\n"
            "    assert candidate('o| .| o| .| o o| o o') == [2, 1, 2, 1, 4, 2, 4, 4]\n"
        ),
        "entry_point": "parse_music",
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
            '    """ From a supplied list of numbers (of length at least two) select and return two that are the closest to each\n'
            "    other and return them in order (smaller number, larger number).\n"
            "    >>> find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0])\n"
            "    (1.0, 2.0)\n"
            "    >>> find_closest_elements([1.0, 2.0, 3.0, 4.0, 5.0, 2.2])\n"
            "    (2.0, 2.2)\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2]) == (3.9, 4.0)\n"
            "    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0]) == (5.0, 5.9)\n"
            "    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.2]) == (2.0, 2.2)\n"
            "    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0]) == (2.0, 2.0)\n"
            "    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1]) == (2.2, 3.1)\n"
        ),
        "entry_point": "find_closest_elements",
    },
    {
        "task_id": "HumanEval/21",
        "prompt": (
            "from typing import List\n\n"
            "def rescale_to_unit(numbers: List[float]) -> List[float]:\n"
            '    """ Given list of numbers (of at least two elements), apply a linear transform to that list,\n'
            "    such that the smallest number will become 0 and the largest will become 1\n"
            "    >>> rescale_to_unit([1.0, 2.0, 3.0, 4.0, 5.0])\n"
            "    [0.0, 0.25, 0.5, 0.75, 1.0]\n"
            '    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate([2.0, 49.9]) == [0.0, 1.0]\n"
            "    assert candidate([100.0, 49.9]) == [1.0, 0.0]\n"
            "    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0]) == [0.0, 0.25, 0.5, 0.75, 1.0]\n"
            "    assert candidate([0.0, 1.0/3, 1.0/2, 1.0]) == [0.0, 1.0/3, 1.0/2, 1.0]\n"
        ),
        "entry_point": "rescale_to_unit",
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
            "    assert candidate('These violent delights have violent ends') == 'tHESE VIOLENT DELIGHTS HAVE VIOLENT ENDS'\n"
        ),
        "entry_point": "flip_case",
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
        "task_id": "HumanEval/30",
        "prompt": (
            "def get_positive(l: list) -> list:\n"
            '    """Return only positive numbers in the list.\n'
            "    >>> get_positive([-1, 2, -4, 3, 5])\n"
            "    [2, 3, 5]\n"
            "    >>> get_positive([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10])\n"
            "    [5, 3, 2, 3, 9, 123, 1]\n"
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
    {
        "task_id": "HumanEval/32",
        "prompt": (
            "import math\n\n"
            "def poly(xs: list, x: float):\n"
            "    \"\"\"\n"
            "    Evaluates polynomial with coefficients xs at point x.\n"
            "    return xs[0] + xs[1] * x + xs[1] * x^2 + .... xs[n] * x^n\n"
            "    \"\"\"\n"
            "    return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])\n\n"
            "def find_zero(xs: list):\n"
            "    \"\"\" xs are coefficients of a polynomial.\n"
            "    find_zero find x such that poly(x) = 0.\n"
            "    find_zero returns only one zero point, even if there are many.\n"
            "    Moreover, find_zero only takes list xs having even number of coefficients\n"
            "    and largest non zero coefficient as it guarantees a solution.\n"
            "    >>> round(find_zero([1, 2]), 2) # f(x) = 1 + 2x\n"
            "    -0.5\n"
            "    >>> round(find_zero([-6, 11, -6, 1]), 2) # (x-1)*(x-2)*(x-3) = -6 + 11x - 6x^2 + x^3\n"
            "    1.0\n"
            "    \"\"\"\n"
        ),
        "test": (
            "def check(candidate):\n"
            "    import math\n"
            "    def poly(xs, x):\n"
            "        return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])\n"
            "    for _ in range(100):\n"
            "        ncoeff = 2 * (random.randint(1, 4))\n"
            "        xs = [random.uniform(-10, 10) for _ in range(ncoeff)]\n"
            "        xs[-1] = abs(xs[-1]) + 1\n"
            "        xs[0] = abs(xs[0]) + 1\n"
            "        result = candidate(xs)\n"
            "        assert abs(poly(xs, result)) < 1e-4\n"
        ),
        "entry_point": "find_zero",
    },
]


def load_humaneval(sample_size: int = 20) -> list[dict]:
    """Return a fixed-seed sample of HumanEval problems."""
    cache_path = os.path.join(HUMANEVAL_DIR, "humaneval_problems.json")
    os.makedirs(HUMANEVAL_DIR, exist_ok=True)

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            problems = json.load(f)
    else:
        problems = HUMANEVAL_PROBLEMS
        with open(cache_path, "w") as f:
            json.dump(problems, f, indent=2)

    if sample_size and sample_size < len(problems):
        random.seed(42)
        problems = random.sample(problems, sample_size)
    return problems


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation: extract code and execute against test suite
# ──────────────────────────────────────────────────────────────────────────────

def _extract_python_code(text: str, entry_point: str) -> str:
    """Pull the best function definition out of a model response."""
    # 1. Prefer fenced code blocks
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    for block in reversed(fenced):
        if f"def {entry_point}" in block:
            return block.strip()
    if fenced:
        return fenced[-1].strip()

    # 2. Extract lines starting from the def
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(rf"\s*def\s+{re.escape(entry_point)}\s*\(", line):
            start = i
            break
    if start is not None:
        return textwrap.dedent("\n".join(lines[start:])).strip()

    # 3. Body-only response: dedent to remove consistent leading whitespace
    return textwrap.dedent(text).strip()


def execute_with_tests(prompt: str, completion: str, test_code: str, entry_point: str) -> bool:
    """Return True if the generated function passes the test suite."""
    code = _extract_python_code(completion, entry_point)

    preamble = "import random, math\nfrom typing import List, Tuple, Optional, Dict\n\n"

    if f"def {entry_point}" in code:
        # Model returned the complete function definition — use it standalone.
        function_section = code
    else:
        # Model returned only the body — re-attach it (indented) to the prompt's signature.
        function_section = prompt + textwrap.indent(code, "    ")

    full_code = preamble + function_section + "\n\n" + test_code + f"\n\ncheck({entry_point})\n"
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_code)
            tmp = f.name
        result = subprocess.run(
            [sys.executable, tmp],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Main runner
# ──────────────────────────────────────────────────────────────────────────────

def run_humaneval(
    model_key: str,
    problems: list[dict],
    max_steps: int = 5,
) -> dict:
    """Run AgentFlow on HumanEval and return pass@1 results."""
    model_string = MODELS[model_key]
    out_dir = os.path.join(RESULTS_DIR, model_key, "humaneval")
    os.makedirs(out_dir, exist_ok=True)

    solver = Solver(
        model=model_string,
        api_key=PORTKEY_API_KEY,
        max_steps=max_steps,
        verbose=False,
    )

    passed = 0
    total = 0

    for i, prob in enumerate(problems):
        task_id = prob["task_id"].replace("/", "_")
        out_file = os.path.join(out_dir, f"{task_id}.json")

        prompt_msg = (
            "Complete the following Python function. "
            "Return ONLY the function body (no need to repeat the signature or docstring).\n\n"
            f"{prob['prompt']}"
        )

        if os.path.exists(out_file):
            with open(out_file) as f:
                cached = json.load(f)
            completion = cached.get("direct_output", "")
            print(f"  [{model_key}] HumanEval {i+1}/{len(problems)}: {prob['entry_point']} (cached)")
        else:
            print(f"  [{model_key}] HumanEval {i+1}/{len(problems)}: {prob['entry_point']}...")
            completion = ""
            for attempt in range(3):
                try:
                    result = solver.solve(prompt_msg)
                    completion = result.get("direct_output", "")
                    result["task_id"] = prob["task_id"]
                    result["entry_point"] = prob["entry_point"]
                    with open(out_file, "w") as f:
                        json.dump(result, f, indent=2, default=str)
                    break
                except Exception as exc:
                    wait = 5 * (2 ** attempt)
                    print(f"    ERROR (attempt {attempt+1}/3): {exc} — retrying in {wait}s")
                    time.sleep(wait)
            else:
                print(f"    FAILED after 3 attempts")

        ok = execute_with_tests(prob["prompt"], completion, prob["test"], prob["entry_point"])
        if ok:
            passed += 1
        total += 1
        print(f"    {'PASS' if ok else 'FAIL'} — running pass@1: {passed}/{total}")

    pass_at_1 = passed / max(total, 1)
    summary = {"pass_at_1": pass_at_1, "passed": passed, "total": total}
    print(f"\n  [{model_key}] HumanEval pass@1 = {pass_at_1:.3f} ({passed}/{total})")

    summary_path = os.path.join(out_dir, "humaneval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all",
                        help="Comma-separated model keys or 'all'")
    parser.add_argument("--sample_size", type=int, default=20,
                        help="Number of HumanEval problems to sample (max 30)")
    parser.add_argument("--max_steps", type=int, default=5)
    args = parser.parse_args()

    models = list(MODELS.keys()) if args.models == "all" else args.models.split(",")
    problems = load_humaneval(args.sample_size)
    print(f"HumanEval: {len(problems)} problems loaded")

    all_results: dict[str, dict] = {}

    for model_key in models:
        if model_key not in MODELS:
            print(f"Unknown model: {model_key}, skipping")
            continue
        print(f"\n{'='*60}")
        print(f"HumanEval: {model_key}")
        print(f"{'='*60}")
        all_results[model_key] = run_humaneval(
            model_key, problems, max_steps=args.max_steps,
        )

    # Summary table
    print(f"\n\n{'='*60}")
    print("HumanEval Results (pass@1)")
    print(f"{'='*60}")
    for mk, res in all_results.items():
        print(f"  {mk:<18}  pass@1={res['pass_at_1']:.3f}  ({res['passed']}/{res['total']})")

    # Merge into combined file
    combined_path = os.path.join(RESULTS_DIR, "humaneval_combined.json")
    if os.path.exists(combined_path):
        with open(combined_path) as f:
            existing = json.load(f)
        existing.update(all_results)
        all_results = existing
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {combined_path}")


if __name__ == "__main__":
    main()
