"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
COMPONENT_RANGE = (int('3131', 16), int('3163', 16))  # kr 자음/모음
COMPLETE_RANGE = (int('ac00', 16), int('d7a3', 16))   # kr all complete chars
COMPLETE_SET = frozenset(chr(code) for code in range(COMPLETE_RANGE[0], COMPLETE_RANGE[1]+1))
COMPLETE_LIST = sorted(COMPLETE_SET)

CHO_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ',
            'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
JUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ',
             'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']
JONG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ',
             'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ',
             'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

N_CHO, N_JUNG, N_JONG = len(CHO_LIST), len(JUNG_LIST), len(JONG_LIST)
N_COMPONENTS = N_CHO + N_JUNG + N_JONG


def compose(cho, jung, jong):
    """Compose ids to char"""
    char_id = cho * N_JONG * N_JUNG + jung * N_JONG + jong + COMPLETE_RANGE[0]
    return chr(char_id)


def decompose(char):
    """Decompose char to ids"""
    char_code = ord(char)
    if COMPLETE_RANGE[0] <= char_code <= COMPLETE_RANGE[1]:
        char_code -= COMPLETE_RANGE[0]
        jong = char_code % N_JONG
        jung = (char_code // N_JONG) % N_JUNG
        cho = char_code // (N_JONG * N_JUNG)
        char_id = (cho, jung, jong)
    elif COMPONENT_RANGE[0] <= char_code <= COMPONENT_RANGE[1]:
        char_code -= COMPONENT_RANGE[0]
        raise ValueError('Component only ({})'.format(char))
    else:
        raise ValueError('{} is Non kor'.format(char))

    return char_id
