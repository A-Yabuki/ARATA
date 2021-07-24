# coding: utf-8

import multiprocessing
import re
import sys
import time

from typing import Callable

from arata.analyzer.analyzer import analyze
from arata.common.constants import MemoryMappedFileName
from arata.common.enum import ModeEnum, UIEnum, ErrorType
from arata.common.error_handler import ChainedErrorHandler, ErrorType
from arata.common.globals import CurrentUI
from arata.common.interprocess_communicator import Communicator, Message
from arata.trainer.trainer import train


def main(mode: ModeEnum, ui: UIEnum) -> bool:

    r"""
    Main function

    Args:
        mode (ModeEnum)... launching mode
        ui (UIEnum)...  user interface

    Returns:
        (bool)... exit normally (True) or abnormally (False)
    """

    func = analyze if mode == ModeEnum.ANALYZER else train

    if ui == UIEnum.GUI_MODE:
        process = multiprocessing.Process(target=func)
        process.daemon = True
        process.start()

        communicator = Communicator()
        communicator.open_mm(MemoryMappedFileName.INTERRUPTER)
        
        while True:

            data = communicator.read_mm()
            msg = Message()
            msg.decode_order(data)

            if msg.msg_category == int(Message.SUPERVISOR_ORDER):
            
                if result.free_int_header == int(1):
                    process.kill()
                    communicator.close_mm()
                    return False

            elif not process.is_alive():
                communicator.close_mm()
                return True

            time.sleep(1)


    elif ui == UIEnum.CUI_MODE:
        func()

        return True


    else:
        return False


if __name__ == "__main__":
    
    r"""
    Entry point
    """

    try:
        mode = int(sys.argv[1])
        ui = int(sys.argv[2])

        if not (mode in ModeEnum.__members__.values()) \
           or not (ui in UIEnum.__members__.values()):
            ChainedErrorHandler(
                "Invalid argument is given.\n" \
                "Required two arguments are 'mode' (0: analyzer, 1: trainer) and 'ui' (0: GUI, 1: CUI).", 
                ErrorType.INVALID_INPUT_VALUE_ERROR)


    except IndexError:
        ChainedErrorHandler(
            "The number of arguments didn't matched the required number (2).\n" \
            "Required two arguments are 'mode' (0: analyzer, 1: trainer) and 'ui' (0: GUI, 1: CUI).", 
            ErrorType.INVALID_INPUT_VALUE_ERROR)

    except:
        ChainedErrorHandler(
            str.format("Wrong args were assigned. mode: {0}, UI: {1}", sys.argv[1], sys.argv[2]), 
            ErrorType.INVALID_INPUT_VALUE_ERROR)
        
    else:
        CurrentUI(UIEnum(ui))

        fin_flag = True
        try:
            fin_flag = main(mode, ui)

        except:
            fin_flag = False

        if fin_flag == False:
            ChainedErrorHandler(
                "All processes were terminated.", 
                ErrorType.INTERRUPTED_ERROR)
