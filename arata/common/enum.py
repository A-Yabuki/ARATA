# coding: utf-8

from enum import IntEnum

class ModeEnum(IntEnum):
    r"""
    Launched application mode.
    """

    NONE = 0
    ANALYZER = 1
    TRAINER = 2


class UIEnum(IntEnum):
    r"""
    User interface
    """

    NONE = 0
    GUI_MODE = 1
    CUI_MODE = 2


class ErrorType(IntEnum):
    r"""
    Error category
    """

    NONE = 0
    IGNORABLE_ERROR = 1
    CRITICAL_ERROR = 2
    INTERRUPTED_ERROR = 3
    INVALID_INPUT_VALUE_ERROR = 11