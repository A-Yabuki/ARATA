# coding: utf-8

import pandas as pd
from typing import Any, Dict, List


class DataTable():

    r"""
    Creates data frame, stores data and outputs a csv file.
    """

    def __init__(self):

        self._df = None


    def create_table(self, data: Dict[str, List[Any]]) -> None:

        r"""
        Create new table to store data.

        Args:
            data (Dict[str, List[Any]]): initial data
        """

        self._df = pd.DataFrame(data)


    def add_numeric_data_column(self, header: str, values: List[float]) -> None:

        r"""
        Adds a column having numeric data to the table

        Args:
            header (str): column header
            values (List[float]): column data
        """

        if (len(values) != len(self._df)):
            return

        self._df[header] = [format(i, '.4f') for i in values]


    def read_csv(self, file_path: str) -> None:
        
        r"""
        Creates data frame from read csv.

        Args:
            file_path (str): csv file path
        """
        self._df = pd.read_csv(file_path)


    def write_csv(self, file_path: str) -> None:

        r""" 
        Writes current data frame into csv file (csv file is overwritten/if not existing, new csv file is created).

        Args:
            file_path (str): csv file path 
        """

        self._df.to_csv(file_path, sep=',', index=False)
