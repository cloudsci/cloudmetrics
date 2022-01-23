import numpy as np


def _parse_example_mask(s):
    return np.array([[float(c) for c in line] for line in s.strip().splitlines()])


EXAMPLES = {}

EXAMPLES["empty"] = [
    _parse_example_mask(
        """
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
"""
    ),
    float("nan"),
]

# Examples are from White et al. (2018)

EXAMPLES["4a"] = [
    _parse_example_mask(
        """
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000011100000
00000000001001100000
00000000000000000000
00000011111100000000
00000011111100000000
00000000000010000000
00000000001000000000
00000000000000000000
00000000000000000000
"""
    ),
    5.42236,
]

EXAMPLES["4b"] = [
    _parse_example_mask(
        """
00000011100000000000
00000011111000000000
00000000110000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000111110000
00000000000110000000
00000000000000000001
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00100000000000000000
00000000000000000000
00000000000000000000
00000000010000000000
00000000010000000000
"""
    ),
    16.4233,
]

EXAMPLES["4c"] = [
    _parse_example_mask(
        """
00000000000000000000
00000000000000000000
00000000000000000000
01100000000000000000
01110000000000000000
00111000000000000000
00010000000000000000
00000000000000000000
00000001111110000000
00000000111111000000
00000000001111111000
00000000000001101000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000110
00000000000000000000
00000000000000000000
00000000000000000000
"""
    ),
    10.0241,
]

EXAMPLES["5a"] = [
    _parse_example_mask(
        """
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000100000000100000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000100000000100000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
"""
    ),
    11.1124,
]

EXAMPLES["5b"] = [
    _parse_example_mask(
        """
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00001110000001110000
00001110000001110000
00001110000001110000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00001110000001110000
00001110000001110000
00001110000001110000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
"""
    ),
    11.1124,
]

EXAMPLES["5c"] = [
    _parse_example_mask(
        """
00000000000000000000
00000000000000000000
00000000000000000000
00011111000011111000
00011111000011111000
00011111000011111000
00011111000011111000
00011111000011111000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00011111000011111000
00011111000011111000
00011111000011111000
00011111000011111000
00011111000011111000
00000000000000000000
00000000000000000000
00000000000000000000
"""
    ),
    11.1124,
]

EXAMPLES["5d"] = [
    _parse_example_mask(
        """
00000000000000000000
00000000000000000000
00111111100111111100
00111111100111111100
00111111100111111100
00111111100111111100
00111111100111111100
00111111100111111100
00111111100111111100
00000000000000000000
00000000000000000000
00111111100111111100
00111111100111111100
00111111100111111100
00111111100111111100
00111111100111111100
00111111100111111100
00111111100111111100
00000000000000000000
00000000000000000000
"""
    ),
    11.1124,
]

EXAMPLES["5e"] = [
    _parse_example_mask(
        """
00000000000000000000
00000000000000000000
00000000000000000000
00011111000011111000
00011111000011111000
00011111000011111000
00011111000011111000
00011111000011111000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000011111000
00001110000011111000
00001110000011111000
00001110000011111000
00000000000011111000
00000000000000000000
00000000000000000000
00000000000000000000
"""
    ),
    11.1124,
]

EXAMPLES["5f"] = [
    _parse_example_mask(
        """
00000000000000000000
00000000000000000000
00000000000000000000
00011111000011111000
00011111000011111000
00011111000011111000
00011111000011111000
00011111000011111000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000000011111000
00000000000011111000
00000100000011111000
00000000000011111000
00000000000011111000
00000000000000000000
00000000000000000000
00000000000000000000
"""
    ),
    11.1124,
]

EXAMPLES["5g"] = [
    _parse_example_mask(
        """
00000000000000000000
00000000000000000000
00000000000111111100
00000000000111111100
00000000000111111100
00000100000111111100
00000000000111111100
00000000000111111100
00000000000111111100
00000000000000000000
00000000000000000000
00000000000111111100
00000000000111111100
00001110000111111100
00001110000111111100
00001110000111111100
00000000000111111100
00000000000111111100
00000000000000000000
00000000000000000000
"""
    ),
    11.1124,
]

EXAMPLES["5h"] = [
    _parse_example_mask(
        """
00000000000000000000
00000000000000000000
00000000000111111100
00000000000111111100
00000000000111111100
00000100000111111100
00000000000111111100
00000000000111111100
00000000000111111100
00000000000000000000
00000000000000000000
00000000000111111100
00000000000111111100
00000000000111111100
00000100000111111100
00000000000111111100
00000000000111111100
00000000000111111100
00000000000000000000
00000000000000000000
"""
    ),
    11.1124,
]
