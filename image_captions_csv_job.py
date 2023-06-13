from typing import List

from image_captions_job import ImageCaptionsJobManager
import csv
import pandas as pd
import argparse


class CSVImageCaptionsJobManager(ImageCaptionsJobManager):
    """
    CSVImageCaptionsJobManager is a concrete implementation of ImageCaptionsJobManager
    specifically for processing CSV files with image captions.
    """

    def __init__(
            self,
            runtime_logger_path: str,
            file_processing_status_path: str,
            picture_url_column: str,
            picture_description_column: str,
            **kwargs
    ):
        """
        Initialize the CSVImageCaptionsJobManager.

        Args:
            runtime_logger_path (str): Path to the file logger.
            file_processing_status_path (str): Path to the processed status pickle file.
            picture_url_column (str): Column name for picture URLs in the CSV file.
            picture_description_column (str): Column name for picture descriptions in the CSV file.
            **kwargs: Keyword arguments to be passed to BatchImageCaptions.
        """
        super().__init__(runtime_logger_path, file_processing_status_path, **kwargs)
        self.picture_url_column = picture_url_column
        self.picture_description_column = picture_description_column

    def save_batch(self, bid, writer, batch):
        if bid == 1:
            writer.writerow(batch.columns)
        writer.writerows(batch.values)

    def get_writer(self, out_file):
        return csv.writer(out_file)

    def get_batch_picture_urls(self, batch: pd.DataFrame) -> List[str]:
        """
        Extract picture URLs from a CSV batch.

        Args:
            batch (pd.DataFrame): The CSV batch.

        Returns:
            List[str]: List of picture URLs.
        """
        return batch[self.picture_url_column].tolist()

    def get_batch_with_image_captions(self, batch: pd.DataFrame, processed_images) -> pd.DataFrame:
        """
        Update the CSV batch with image captions.

        Args:
            batch (pd.DataFrame): The CSV batch to update.
            processed_images: The processed images containing captions.

        Returns:
            pd.DataFrame: Updated CSV batch with image captions.
        """
        batch[self.picture_description_column] = [image.caption for image in processed_images]
        return batch

    def _iter_batches(self, file_path):
        """
        Iterate through the CSV file in batches.

        Args:
            file_path (str): Path to the input CSV file.

        Yields:
            pd.DataFrame: A batch of data from the CSV file.

        """
        for batch in pd.read_csv(file_path, chunksize=self.bic.batch_size):
            yield batch


class ABNBImageCaptionsJobManager(CSVImageCaptionsJobManager):
    """
    ABNBImageCaptionsJobManager is a concrete implementation of CSVImageCaptionsJobManager
    specifically for processing ABNB (Airbnb) CSV files with image captions.
    """

    def __init__(self, logger_path: str, file_processing_status_path: str, **kwargs):
        super().__init__(logger_path, file_processing_status_path, "picture_url", "picture_description", **kwargs)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Image Captions Job Manager")
    parser.add_argument("--files", nargs="+", help="Input file paths", required=True)
    parser.add_argument("--out_paths", nargs="+", help="Output file paths", required=True)
    parser.add_argument("--files_pickle", help="Path to the processed status pickle file", default="status.pkl")
    parser.add_argument("--logger_path", help="Path to the logger", default="log.txt")
    parser.add_argument("--picture_url_column", help="Column name for picture URLs in the CSV file",
                        default="picture_url")
    parser.add_argument("--picture_description_column", help="Column name for picture descriptions in the CSV file",
                        default="picture_description")
    return parser.parse_args()


def main():
    """
    Main function for running the Image Captions Job Manager.
    """
    args = parse_arguments()
    job_manager = ABNBImageCaptionsJobManager(
        logger_path=args.logger_path,
        file_processing_status_path=args.files_pickle,
        picture_url_column=args.picture_url_column,
        picture_description_column=args.picture_description_column,
        batch_size=16,
        num_workers=4
    )
    job_manager.process_files(args.files, args.out_paths)


if __name__ == "__main__":
    main()
