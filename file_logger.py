"""
FileLogger Class

Author: Sheli Kohan
Date: 22.5.23

The FileLogger class provides logging functionality to a file.
"""
import logging


class FileLogger:
    """
    FileLogger is a class that provides logging functionality to a file.
    """

    def __init__(self, logger_name, file_path):
        """
               Initializes the FileLogger instance.

               Args:
                   logger_name (str): The name of the logger.
                   file_path (str): The path to the log file.
               """
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)

        # Create a file handler and set its level to DEBUG
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.DEBUG)

        # Create a formatter and set it for the handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        self.logger.addHandler(file_handler)

    def debug(self, s: str):
        self.logger.debug(s)

    def info(self, s: str):
        self.logger.info(s)

    def warning(self, s: str):
        self.logger.warning(s)

    def error(self, s: str):
        self.logger.error(s)
