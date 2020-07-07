"""
DMFont
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import numpy as np


def cyclize(loader):
    """ Cyclize loader """
    while True:
        for x in loader:
            yield x


def rev_dict(l):
    """ Reverse dict or list """
    return {k: i for i, k in enumerate(l)}


def uniform_indices(end, n_sample, st=None):
    """ Sample from [0, end) with (almost) equidistant interval """
    if end <= 0:
        return np.empty(0, dtype=np.int)

    # NOTE with endpoint=False, np.linspace does not sample the `end` value
    indices = np.linspace(0, end, num=n_sample, dtype=np.int, endpoint=False)
    if st is None and end:
        st = (end-1 - indices[-1]) // 2
    return indices + st


def sample(population, n_sample, exception=None):
    """ sampling without replacement N elements from set with exception

    Params:
        population: [1d] list or set or np.ndarray
    Return: np.ndarray
    """
    if exception is not None:
        population = set(population) - set(exception)
    if not isinstance(population, np.ndarray):
        population = np.asarray(list(population))
    ids = np.random.choice(len(population), size=n_sample, replace=False)
    return population[ids]


def uniform_sample(population, n_sample, st=None):
    assert not isinstance(population, set), "population should have order"

    N = len(population)
    indices = uniform_indices(N, n_sample, st)

    if isinstance(population, np.ndarray):
        return population[indices]
    elif isinstance(population, list):
        return [population[idx] for idx in indices]
    elif isinstance(population, str):
        return ''.join([population[idx] for idx in indices])
    else:
        raise TypeError(type(population))


def get_fonts(avails):
    return list(avails.keys())


def get_union_chars(avails):
    return sorted(set.union(*map(set, avails.values())))


def get_fonts_unionchars(avails):
    return get_fonts(avails), get_union_chars(avails)


def get_intersection_chars(avails):
    return sorted(set.intersection(*map(set, avails.values())))
