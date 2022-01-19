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
    float("nan"),
]

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
00000000000111110000
00000000010011110000
00000000000000000000
00000111111000000000
00000111111000000000
00000000000100000000
00000000010000000000
00000000000000000000
00000000000000000000
"""
    ),
    5.42236,
    0.517087,
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
00000000000111110000
00000000000110000000
00000000000000000001
00000000000000000000
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
    0.187769,
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
    0.113135,
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
    0.339405,
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
    0.565675,
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
    0.791944,  # value quoted in White et al is 0.791994, but I think that is incorrect
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
    0.509107,
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
    0.452540,
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
    0.509107,
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
    0.452540,
]
