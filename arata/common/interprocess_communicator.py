# coding: utf-8

import mmap
from typing import Tuple

from .globals import CurrentUI
from .constants import DetailedErrorMessages, MemoryMappedFileName
from .enum import ErrorType, UIEnum
from .singleton import Singleton
from .wrapper_collection import watch_error

class Message():

    r"""
    The message object used to communicate information to GUI with memory mapped files

    --------------
    Data structure
    --------------

    Head
    ----
    category: 4 bytes
    free_num: 4 bytes
    message_length: 4 bytes
    message: "message_length" bytes
    ----
    Tail


    """

    # enum of "category"

    INVALID = 0
    SUPERVISOR_ORDER = 1 # Order from GUI
    PROGRESS = 2 # Progress information
    ERROR = 99 # Error information


    def __init__(self) -> None:

        r"""
        msg_category: bytes (int)
        free_int_header: bytes (int)
        msg_length: bytes (int)
        msg: bytes (str)
        """
        self.msg_category = self.INVALID
        self.free_int_header = 0
        self.msg_length = 0
        self.msg = ""


    def encode_progress(self, progress: int, msg: str) -> bytes:
        
        r"""
        Encodes a progress message

        Args:
            progress (int): progress (%) of current works.
            msg (str): message

        Returns:
            msg_bytes (bytes): bytes in a format that GUI can receive.
        """

        msg_bytes = self._encode(self.PROGRESS, progress, msg)

        self.msg_category = self.PROGRESS
        self.free_int_header = progress
        self.msg_length = len(msg_bytes)
        self.msg = msg

        return msg_bytes


    def encode_error(self, error_category: ErrorType, msg: str) -> bytes:

        r"""
        Encodes an error message

        Args:
            error_category (ErrorType/int): Error Category
            msg (str): message

        Returns:
            msg_bytes (bytes): bytes in a format that GUI can receive.
        """
        msg_bytes = self._encode(self.ERROR, int(error_category), msg)

        self.msg_category = self.ERROR
        self.free_int_header = error_category
        self.msg_length = len(msg_bytes)
        self.msg = msg

        return msg_bytes


    @watch_error(
        DetailedErrorMessages.DECODE_ERROR, 
        ErrorType.CRITICAL_ERROR)
    def decode_order(self, order_bytes: bytes) -> None:
        
        r"""
        Decodes bytes.
        
        Args:
            order_bytes (bytes): encoded this object.
        """

        if len(order_bytes) < 12:
            return

        self.msg_category = self._intpbytes(order_bytes[0:4], int)

        self.free_int_header = self._intpbytes(order_bytes[4:8], int)
        self.msg_length = self._intpbytes(order_bytes[8:12], int)
        self.msg = self._intpbytes(order_bytes[12:], str)


    @watch_error(
        DetailedErrorMessages.ENCODE_ERROR, 
        ErrorType.CRITICAL_ERROR
        )
    def _encode(self, cat: int, free: int, msg: str) -> bytes:
        
        r"""
        Encodes

        Args:
            cat (int): the value inserted into the category field
            free (int): the value inserted into the free field
            msg (str): the value inserted into the message field

        Returns:
            telegram (bytes): bytes in a format that GUI can receive.
        """

        msg_len, msg_bytes = self._cvtbytes(msg)
        _, length_bytes = self._cvtbytes(msg_len)
        _, category_bytes = self._cvtbytes(cat)
        _, free_bytes = self._cvtbytes(free)

        telegram = self._concat(category_bytes, free_bytes, length_bytes, msg_bytes)

        return telegram


    def _concat(self, *args: bytes) -> bytes:

        r"""
        Concatenate fields

        Args:
            args (bytes): bytes

        Returns:
            ret (bytes): the result of concatenating all args.
        """
        ret = b''
        
        for arg in args:
            ret += arg

        return ret


    def _cvtbytes(self, val: object) -> Tuple[int, bytes]:

        if isinstance(val, str):
            return self._str2bytes(val)

        elif isinstance(val, int):
            return self._int2bytes(val)


    def _intpbytes(self, val: bytes, data_type: type):

        if data_type is str:
            return self._bytes2str(val)

        elif data_type is int:
            return self._bytes2int(val)


    def _str2bytes(self, val: str) -> Tuple[int, bytes]:
        
        byte_msg = val.encode('utf-8')
        byte_length = len(byte_msg)
    
        return byte_length, byte_msg


    def _int2bytes(self, val: int) -> Tuple[int, bytes]:
        
        if (val < (2**8)**4):    
            return 4, int(val).to_bytes(4, 'little')

        else:
            raise ValueError("The given value is beyond 4bytes.")


    def _bytes2str(self, val: bytes) -> str:
        return val.decode('utf-8')


    def _bytes2int(self, val: bytes) -> int:
        return int.from_bytes(val, 'little')


class Communicator(Singleton):

    r"""
    Offers basic memory mapped file operations.
    """

    def __init__(self) -> None:
        self.mms = {}
        self.msize = 10000


    def __del__(self) -> None:
        _ = [self.close_mm(name) for name in self.mms] 


    def open_mm(self, name: str) -> None:
        
        r"""
        Open a memory mapped file.

        Args:
            name (str): memory mapped file name to be opened.
        """

        if (name in self.mms):
            return

        self.mms[name] = mmap.mmap(-1, length=self.msize, tagname=name)


    def close_mm(self, name: str) -> None:

        r"""
        Close a memory mapped file.

        Args:
            name (str): memory mapped file name to be closed.
        """
        
        if (name in self.mms) and not self.mms[name].closed:
            self.mms[name].close()
            del self.mms[name]


    @watch_error(
        DetailedErrorMessages.MEMORY_WRITE_ERROR,
        ErrorType.IGNORABLE_ERROR
        )
    def write_mm(self, name: str, data: bytes):

        if (name in self.mms) and not self.mms[name].closed:
            self.mms[name].seek(0)
            self.mms[name].write(data)


    @watch_error(
        DetailedErrorMessages.MEMORY_READ_ERROR, 
        ErrorType.IGNORABLE_ERROR
        )
    def read_mm(self, name: str):

        if self.mms is None:
            return

        self.mms[name].seek(0)
        read_data = self.mms[name].read()
        
        return read_data


def report_progress(process_name: str, progress: int, info: str = "") -> None:
    
    cur_ui = CurrentUI()
    ui_mode = cur_ui.current_ui

    if (ui_mode == UIEnum.GUI_MODE):
        msg = Message()
        info_msg = str.format("{0} [Process: {1}] ", info, process_name)
        data = msg.encode_progress(int(progress), info_msg)
        communicator = Communicator()
        communicator.open_mm(MemoryMappedFileName.REPORTER)
        communicator.write_mm(data)
        communicator.close_mm(MemoryMappedFileName.REPORTER)

    elif (ui_mode == UIEnum.CUI_MODE):
        print(str.format("{0} [{1}  {2} %]", info, process_name, progress), flush=True)


def report_error(process_name: str, error_category: ErrorType, info: str = "") -> None:

    ui_mode = CurrentUI().current_ui

    if (ui_mode == UIEnum.GUI_MODE):
        msg = Message()
        info_msg = str.format("{0} [Process: {1}] ", info, process_name)
        data = msg.encode_error(error_category, info_msg)
        communicator = Communicator()
        communicator.open_mm(MemoryMappedFileName.REPORTER)
        communicator.write_mm(data)
        communicator.close_mm(MemoryMappedFileName.REPORTER)

    elif (ui_mode == UIEnum.CUI_MODE):
        print(str.format("{0} [{1}]", info, process_name), flush=True)
