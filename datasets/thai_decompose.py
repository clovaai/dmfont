"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
from .data_utils import rev_dict

CONSONANTS = [3585, 3586, 3587, 3588, 3589, 3590, 3591, 3592, 3593, 3594, 3595, 3596, 3597, 3598, 3599, 3600, 3601, 3602, 3603, 3604, 3605, 3606, 3607, 3608, 3609, 3610, 3611, 3612, 3613, 3614, 3615, 3616, 3617, 3618, 3619, 3621, 3623, 3624, 3625, 3626, 3627, 3628, 3629, 3630]
UPPERS = [3633, 3635, 3636, 3637, 3638, 3639, 0]
HIGHESTS = [3655, 3656, 3657, 3658, 3659, 3660, 3661, 3662, 0]
LOWERS = [3640, 3641, 3642, 0]

N_CONSONANTS = len(CONSONANTS)
N_UPPERS = len(UPPERS)
N_HIGHESTS = len(HIGHESTS)
N_LOWERS = len(LOWERS)
N_COMPONENTS = N_CONSONANTS + N_UPPERS + N_HIGHESTS + N_LOWERS

def _ord2idx(ords):
    dic = {
        ord_num: i for i, ord_num in enumerate(ords)
    }
    return dic

CONSONANTS_ORD2IDX = _ord2idx(CONSONANTS)
UPPERS_ORD2IDX = _ord2idx(UPPERS)
HIGHESTS_ORD2IDX = _ord2idx(HIGHESTS)
LOWERS_ORD2IDX = _ord2idx(LOWERS)


def compose(consonant, upper, highest, lower):
    """ Compose ords to char """
    s = chr(consonant) \
        + (chr(upper) if upper else '') \
        + (chr(highest) if highest else '') \
        + (chr(lower) if lower else '')

    return s


def compose_ids(consonant, upper, highest, lower):
    """ Compose ids to char """
    consonant = CONSONANTS[consonant]
    upper = UPPERS[upper]
    highest = HIGHESTS[highest]
    lower = LOWERS[lower]

    return compose(consonant, upper, highest, lower)


def complete_chars():
    for consonant in CONSONANTS:
        for upper in UPPERS:
            for highest in HIGHESTS:
                for lower in LOWERS:
                    char = compose(consonant, upper, highest, lower)
                    yield char


ALLCHARS = list(complete_chars())
CHAR2IDX = rev_dict(ALLCHARS)


def ord2idx_2d(ords):
    """
    Args:
        ords: 2d [[consonant, upper, highest, lower], ...] array
    Return: 2d [[consonant_idx, upper_idx, highest_idx, lower_idx], ...] array
    """

    ret = []
    for consonant, upper, highest, lower in ords:
        consonant_idx = CONSONANTS_ORD2IDX[consonant]
        upper_idx = UPPERS_ORD2IDX[upper]
        highest_idx = HIGHESTS_ORD2IDX[highest]
        lower_idx = LOWERS_ORD2IDX[lower]

        ret.append((consonant_idx, upper_idx, highest_idx, lower_idx))

    return ret


def decompose_ords(char):
    """ Decompose char into ords """
    if not char:
        return 0, 0, 0, 0

    consonant = ord(char[0])
    upper = highest = lower = 0
    for ch in char[1:]:
        ch = ord(ch)
        if ch in UPPERS:
            upper = ch
        elif ch in HIGHESTS:
            highest = ch
        elif ch in LOWERS:
            lower = ch
        else:
            raise ValueError(char)

    return consonant, upper, highest, lower


def decompose_ids(char):
    """ Decompose char into ids """
    index = CHAR2IDX[char]
    return decompose_index(index)


def decompose_index(index):
    """ Decompose char index to component indices
    Note: char index come from `complete_chars()`.
    """
    lower_idx = index % N_LOWERS
    index //= N_LOWERS

    highest_idx = index % N_HIGHESTS
    index //= N_HIGHESTS

    upper_idx = index % N_UPPERS
    index //= N_UPPERS

    consonant_idx = index

    return consonant_idx, upper_idx, highest_idx, lower_idx
