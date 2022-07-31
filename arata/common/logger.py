# coding: utf-8

import logging
import os
import traceback

from .constants import ResourcePathConst
from .enum import PlatformEnum
from .globals import EnvironmentInfo
from .platform_utils import mkdir_recursive


class Logger():
    
    r"""
    Logger
    """

    INFO = "info"
    ERROR = "error"

    mkdir_recursive(ResourcePathConst.LOG_OUTPUT_PATH)
    
    # level_name.. ログレベル
    # asctime.. 現時刻
    # message.. メッセージ文字列
    # pathname.. ログが呼び出された物理パス
    # funcName.. ログが呼び出された関数名 (なぜかN大文字)
    formatter = '[%(levelname)s] %(asctime)s : %(message)s'

    file_name = { 
        logging.DEBUG: os.path.join(ResourcePathConst.LOG_OUTPUT_PATH, ResourcePathConst.APP_LOG),
        logging.INFO: os.path.join(ResourcePathConst.LOG_OUTPUT_PATH, ResourcePathConst.APP_LOG),
        logging.WARN: os.path.join(ResourcePathConst.LOG_OUTPUT_PATH, ResourcePathConst.ERROR_LOG),
        logging.CRITICAL: os.path.join(ResourcePathConst.LOG_OUTPUT_PATH, ResourcePathConst.ERROR_LOG),
    }

    logging.basicConfig(format=formatter)
 
    info_log_handler = logging.FileHandler(file_name[logging.DEBUG])

    error_log_handler = logging.FileHandler(file_name[logging.CRITICAL])

    info_logger = logging.getLogger(INFO)
    info_logger.addHandler(info_log_handler)

    error_logger = logging.getLogger(ERROR)
    error_logger.addHandler(error_log_handler)
   

    def __init__(self):
        pass


    @classmethod
    def write_debug(cls, msg: str) -> None:
        
        logger = logging.getLogger(cls.INFO)
        logger.setLevel(logging.DEBUG)
        logger.debug('%s', msg)

    @classmethod
    def write_info(cls, msg: str) -> None:
        
        logger = logging.getLogger(cls.INFO)
        logger.setLevel(logging.INFO)
        logger.info('%s', msg)
        

    @classmethod
    def write_warning(cls, msg: str) -> None:
        
        logger = logging.getLogger(cls.ERROR)
        logger.setLevel(logging.WARN)
        logger.warn('%s', msg)
        

    @classmethod
    def write_critical(cls, msg: str, file_name: str = "", method_name: str = "") -> None:
        
        logger = logging.getLogger(cls.ERROR)
        logger.setLevel(logging.CRITICAL)

        error_msg = msg

        error_msg += "\n%s"%(traceback.format_exc())

        logger.critical('%s', "Critical error occured at Class: %s Method:%s.  \n%s"%(file_name, method_name, error_msg))