# coding: utf-8

import inspect
import sys

from .enum import ErrorType
from .interprocess_communicator import report_error
from .logger import Logger
from .constants import DetailedErrorMessages, DisplayErrorMessages


class ErrorHandler():

    """
    Error handler base class 
    An error is turned around this subclasses based on Chain of the responsibility design pattern.

    """

    def __init__(self) -> None:
        
        self.next = None;


    def inform(self, msg: str, error_no: ErrorType, file_name: str, func_name: str) -> None:

        process_name = str.format("File: {0}, Method: {1}", file_name, func_name)
        report_error(process_name, error_no, msg)


    def set_next(self, next_handler: 'ErrorHandler') -> 'ErrorHandler':
        
        """ Set a next handler """
        
        self.next = next_handler

        return self


    def catch(self, msg: str, error_no: ErrorType, file_name: str, func_name: str, *args, **kwargs) -> None:

        """ Catch Error or pass through """

        if (not self.handle_error(msg, error_no, file_name, func_name, *args, **kwargs)):
            self.next.catch(msg, error_no, file_name, func_name, *args, **kwargs)

        return


    def handle_error(self, msg: str, 
                     error_no: ErrorType, file_name: str, 
                     func_name: str, *args, **kwargs) -> bool:

        self.inform(msg, error_no, file_name, func_name)
        Logger.write_warning(DisplayErrorMessages.UNEXPECTED_ERROR)
        return True


class InterruptedErrorHandler(ErrorHandler):

    def __init__(self) -> None:
        super().__init__()


    def handle_error(self, msg: str, 
                     error_no: ErrorType, file_name: str, 
                     func_name: str, *args, **kwargs) -> bool:

        if error_no == ErrorType.INTERRUPTED_ERROR:
            self.inform(msg, error_no, file_name, func_name)
            Logger.write_info(msg)
            return True

        else:
            return False


class CriticalErrorHandler(ErrorHandler):

    def __init__(self) -> None:
        super().__init__()


    def handle_error(self, msg: str, 
                     error_no: int, file_name: str, 
                     func_name: str, *args, **kwargs) -> bool:

        if error_no == ErrorType.CRITICAL_ERROR:               
            self.inform(msg, error_no, file_name, func_name)
            Logger.write_critical(msg, file_name, func_name)  
            sys.exit()

        else:
            return False


class IgnorableErrorHandler(ErrorHandler):

    def __init__(self) -> None:
        super().__init__()


    def handle_error(self, msg: str, 
                     error_no: int, file_name: str, 
                     func_name: str, *args, **kwargs) -> bool:

        if error_no == ErrorType.IGNORABLE_ERROR:               
            Logger.write_warning(msg)
            return True
        
        else:
            return False


class InvalidInputValueErrorHandler(ErrorHandler):

    def __init__(self) -> None:
        super().__init__()


    def handle_error(self, msg: str, error_no: int, 
                     file_name: str, func_name: str, 
                     *args, **kwargs) -> bool:

        if error_no == ErrorType.INVALID_INPUT_VALUE_ERROR:
            self.inform(msg, error_no, file_name, func_name)
            Logger.write_critical(msg, file_name, func_name)
            sys.exit()

        else:

            return False


class ChainedErrorHandler():

    handler_chain = InterruptedErrorHandler().set_next(IgnorableErrorHandler().set_next(CriticalErrorHandler().set_next(InvalidInputValueErrorHandler().set_next(ErrorHandler()))))

    def __init__(self, msg: str, error_no: int, *args, **kwargs):
        
        file_name, func_name = inspect.stack()[1].filename, inspect.stack()[1].function
        self.handler_chain.catch(msg, error_no, file_name, func_name, *args, **kwargs)