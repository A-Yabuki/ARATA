# coding: utf-8

from arata.common import platform_utils
from .enum import PlatformEnum, UIEnum
from .singleton import Singleton
from .wrapper_collection import methoddispatch


class CurrentUI(Singleton):

    r"""
    Current UI 
    """
    
    @methoddispatch
    def __init__(self, current_ui: object) -> None:
        pass


    @__init__.register(UIEnum)
    def _(self, current_ui: UIEnum):
        if ("_current_ui" in self.__dict__):
            return None
        
        self._current_ui = current_ui
        return None


    @property
    def current_ui(self) -> UIEnum:
        return  self._current_ui if ("_current_ui" in self.__dict__) else UIEnum.NONE


class EnvironmentInfo(Singleton):
    r"""
    Environment information    
    """

    def __init__(self) -> None:
        self._system = platform_utils.get_platform()
        self._is_supported = platform_utils.is_supported_platform()

    @property
    def platform(self) -> PlatformEnum:
        return self._system

    @property
    def is_supported(self) -> bool:
        return self._is_supported