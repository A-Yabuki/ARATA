# coding: utf-8

from .enum import UIEnum
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
