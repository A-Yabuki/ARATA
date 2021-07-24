# coding: utf-8

import re
from typing import List


def numerical_sort(value: str) -> List[str]:
    """ key of sorting string number """
    
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts