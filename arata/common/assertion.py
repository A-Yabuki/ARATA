# coding: utf-8

from collections.abc import Iterable
from typing import Any, Sequence

def isiterable(obj: Any) -> bool:
    try:
        iterator = iter(obj)
    except TypeError:
        return False
    else:
        return True


def hasinstances(seq: Sequence, t: type) -> bool:
    return False if not isiterable(seq) else all([isinstance(elem, t) for elem in seq])


def isempty(seq: Sequence[Any]) -> bool:
    return True if (seq == None or len(seq) == 0) else False