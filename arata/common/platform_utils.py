# coding:utf-8

import os
import platform

from .enum import ErrorType, PlatformEnum

def get_platform() -> PlatformEnum:

    sys_name = platform.system()

    if sys_name == "Wndows":
        return PlatformEnum.WINDOWS

    elif sys_name == "Linux":
        return PlatformEnum.LINUX

    elif sys_name == "Darwin":
        return PlatformEnum.DARWIN

    elif sys_name == "Java":
        return PlatformEnum.JAVA


def is_supported_platform() -> bool:

    system = get_platform()

    supported = False
    if system in (PlatformEnum.WINDOWS, PlatformEnum.LINUX):
        supported = True

    return supported


def mkdir_recursive(path: str) -> bool:

    try:
        if (os.path.exists(path)):
            return False

        os.makedirs(path)

        return True

    except: 
        raise