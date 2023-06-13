"""
Image Captions Job
Author: Sheli Kohan
Date: 12th June 2023
"""
import os
import time
from typing import List
from file_utils import JsonFileHandler, get_gpu_memory_usage, get_cpu_memory_usage, get_ram_usage
from file_logger import FileLogger
from batch_image_captions import BatchImageCaptions
from abc import ABC, abstractmethod
from image_with_caption import ImageWithCaption


class ImageCaptionsJobManager(ABC):
    """
    Class responsible for managing the image captioning job.
    This job includes logging, tracking, and maintaining the code to ensure
    easy debugging and recovery in case of failures.

    Logging: The job logs important events and error messages to facilitate
    troubleshooting.

    Tracking: The job keeps track of processed batches and their status to avoid
    reprocessing them. The processed batches and their status are stored as pickle
    files for easy retrieval and recovery.

    Maintaining: The job maintains the code by organizing and documenting it
    properly, making it easier to understand and maintain over time.
    """

    logger_name: str = 'image_captioning'

    def __init__(self, logger_path: str, file_processing_status_path: str, **kwargs):
        """
        Initialize the ImageCaptionsJobManager.

        Args:
            logger_path (str): Path to the file logger.
            file_processing_status_path (str): Path to the processed status pickle file.
            **kwargs: Keyword arguments to be passed to BatchImageCaptions.
        """
        self.logger = FileLogger(self.logger_name, logger_path)
        self.bic = BatchImageCaptions(**kwargs)
        self.status = JsonFileHandler(file_processing_status_path, {})

    def get_processing_id(self, file_path: str, out_path: str) -> str:
        """
        Get the unique identifier for a specific processing job.

        Args:
            file_path (str): Input file path.
            out_path (str): Output file path.

        Returns:
            str: Processing job identifier.
        """
        return f'{str(self.bic)}|{file_path}|{out_path}'

    def __process_file(self, file_path: str, out_path: str):
        """
        Process a single file for image captioning.

        Args:
            file_path (str): Path to the input file.
            out_path (str): Path for the output file.
        """
        pid = self.get_processing_id(file_path, out_path)
        f_status = self.status.data.get(pid, self.get_new_file_info(file_path, out_path))
        if self.validate_processing(f_status):
            with open(out_path, 'a', newline='', encoding='utf-8') as out_file:
                self.status.data[pid] = f_status
                writer = self.get_writer(out_file)
                for bid, batch in enumerate(self._iter_batches(file_path), start=1):
                    if bid > f_status["n_batches"]:
                        self.process_batch(bid, batch, f_status, writer)
            f_status["is_done"] = True
            self.logger.info(f"Finished gracefully {file_path} with {bid} batches.")
            self.status.save_data()

    @abstractmethod
    def get_writer(self, out_file):
        """
        Get a writer object for writing to the output file.

        Args:
            out_file: Output file object.

        Returns:
            Writer object for the output file.
        """
        pass

    def validate_processing(self, f_status: dict) -> bool:
        """
        Validate if the processing can be performed based on the file status.

        Args:
            f_status (dict): File status information.

        Returns:
            bool: True if the processing can be performed, False otherwise.
        """
        if os.path.exists(f_status["out_path"]) and f_status["n_batches"] == 0:
            self.logger.warning(f"The file '{f_status['out_path']}' already exists, the code crashed.")
            raise FileExistsError(f"The file '{f_status['out_path']}' already exists.")
        if f_status["is_done"]:
            self.logger.info(f"Skip '{f_status['out_path']}' already processed.")
            return False
        return True

    def process_batch(self, bid: int, batch, f_status: dict, writer):
        """
        Process a single batch for image captioning.

        Args:
            bid (int): Batch ID.
            batch : can be any type of object, depending on the child implementation
            f_status (dict): File status information.
        """
        start_time = time.time()
        processed_images = self.bic.get_image_captions_from_urls(self.get_batch_picture_urls(batch))
        batch = self.get_batch_with_image_captions(batch, processed_images)
        self.save_batch(bid, writer, batch)
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.logger.info(f"Done batch {bid}, Runtime: {elapsed_time:.2f} seconds.")
        self.notify_batch_is_done(f_status, elapsed_time)

    def get_new_file_info(self, file_path: str, out_path: str) -> dict:
        """
        Create a new file info dictionary.

        Args:
            file_path (str): Input file path.
            out_path (str): Output file path.

        Returns:
            dict: New file info dictionary.
        """
        return {"file_path": file_path, "out_path": out_path, "n_batches": 0, "is_done": False, "cpu_usage": [],
                "gpu_usage": [], "iter_time": [], "ram_usage": []}

    def notify_batch_is_done(self, f_status: dict, elapsed_time: float):
        """
        Notify that a batch is done and update the processing status.

        Args:
            f_status (dict): File processing status.
            elapsed_time (float): Elapsed time for the batch.
        """
        f_status["n_batches"] += 1
        f_status["gpu_usage"].append(get_gpu_memory_usage())
        f_status["cpu_usage"].append(get_cpu_memory_usage())
        f_status["ram_usage"].append(get_ram_usage())
        f_status["iter_time"].append(elapsed_time)
        self.status.save_data()

    def process_file(self, file_path: str, out_path: str):
        """
        Process a single file for image captioning and close the internal processing pool at the end.

        Args:
            file_path (str): Path to the input file.
            out_path (str): Path for the output file.
        """
        self.__process_file(file_path, out_path)
        self.bic.close_pool()

    @abstractmethod
    def get_batch_picture_urls(self, batch) -> List[str]:
        """
        Abstract method to extract picture URLs from a batch.

        Args:
            batch: The batch to extract picture URLs from.

        Returns:
            List[str]: List of picture URLs.
        """
        pass

    @abstractmethod
    def get_batch_with_image_captions(self, batch, processed_images: List[ImageWithCaption]) :
        """
        Abstract method to update the batch with image captions.

        Args:
            batch: The batch to update.
            processed_images: The processed images containing captions.

        Returns:
            Updated batch with image captions.
        """
        pass

    @abstractmethod
    def _iter_batches(self, file_path):
        """
        Abstract method to iterate through the file in batches.

        Args:
            file_path (str): Path to the input file.

        Yields:
            A batch of data from the input file in any desired format

        """
        pass

    def process_files(self, files: List[str], out_paths: List[str]):
        """
        Process multiple files for image captioning.

        Args:
            files (list): List of input file paths.
            out_paths (list): List of output file paths.
        """
        assert len(files) == len(out_paths)
        for file_path, out_path in zip(files, out_paths):
            self.__process_file(file_path, out_path)
        self.bic.close_pool()

    @abstractmethod
    def save_batch(self, bid, writer, batch):
        """
        Save the processed batch to the output file.

        Args:
            writer: Writer object for writing to the output file.
            bid (int): Batch ID.
            batch: Batch of data.
        """
        pass

