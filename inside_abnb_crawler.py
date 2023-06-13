import os
import time
import shutil
import gzip
import requests
from urllib.parse import quote
from bs4 import BeautifulSoup
from file_logger import FileLogger
from abc import ABC, abstractmethod
import file_utils
import argparse


class InsideABNBCrawler(ABC):
    """
    Base class for Inside Airbnb crawlers.
    """
    inside_airbnb_url = "http://insideairbnb.com/get-the-data/"
    logger_name = "iabnb_crawler"

    def __init__(self, log_path, parent_dir):
        self.logger = FileLogger(self.logger_name, log_path)
        self.start_time = time.time()
        self.parent_dir = parent_dir
        self.csv_urls = []
        self.url2local_path = []
        self.soup = None
        self.response = None

    def run(self):
        """
        Run the Inside Airbnb crawler.
        """
        self.crawl_valid_urls()
        self.url2local_path = [(url, self.get_local_path(url)) for url in self.csv_urls]
        self.filter_existing_csvs()
        self.makedirs_by_urls()
        self.download_csvs()

    def makedirs_by_urls(self):
        for url, local_path in self.url2local_path:
            file_utils.makedirs_if_not_exists(local_path)

    def filter_existing_csvs(self):
        """
        Filter existing CSV files to avoid duplicate downloads.
        """
        # TODO implement
        pass

    def url_is_valid(self, url):
        return True

    @abstractmethod
    def iter_urls(self):
        """
        Iterate over the URLs on the Inside Airbnb website. This is an abstract method that is implemented differently
        according to the desired data type.

        Yields:
        str: URL string.
        """
        pass

    def crawl_valid_urls(self):
        """
        Crawl the Inside Airbnb website to extract valid CSV file URLs.
        """
        self.response = requests.get(self.inside_airbnb_url)
        self.soup = BeautifulSoup(self.response.text, "html.parser")
        for url in self.iter_urls():
            if self.url_is_valid(url):
                self.logger.info(f"Valid URL {url}")
                self.csv_urls.append(url)
            else:
                self.logger.error(f"Invalid URL {url}")

    def download_csvs(self):
        """
        Download the CSV files from the URLs and save them locally.
        """
        for url, local_path in self.url2local_path:
            print(url)
            self.download_csv_from_gz_url(url, local_path)

    def download_csv_from_gz_url(self, url, local_path):
        """
        Download a CSV file from a gzipped URL and save it locally.

        Args:
            url (str): URL of the gzipped CSV file.
            local_path (str): Local path to save the CSV file.
        """
        self.logger.info(f"trying to download {url}")
        response = requests.get(url, stream=True)
        with open(local_path, 'wb') as f:
            with gzip.GzipFile(fileobj=response.raw) as gzipped_file:
                shutil.copyfileobj(gzipped_file, f)

        response.close()
        self.logger.info(f"{url} file downloaded and saved successfully to {local_path}")

    @abstractmethod
    def get_local_path(self, url: str) -> str:
        """
        Get a local path to save the CSV file for a given URL based on its features.

        Args:
            url (str): URL of the CSV file.

        Returns:
            str: Local path to save the CSV file.
        """
        pass


class IABNBListingsCrawler(InsideABNBCrawler):
    listing_pattern = "http://data.insideairbnb.com/*data/listings.csv"
    invalid_str_for_extraction = ["href", "Summary", "GIS files", "GeoJSON", "Review", "Calendar", "Listing data",
                                  "Listings data"]

    def iter_urls(self):
        for tag in set(self.soup.find_all("td", string=self.soup.find_all("td", string=self.listing_pattern))):
            tag_str = str(tag)
            self.logger.info(f"analyzing tag {tag_str}")
            if not any(substring in tag_str for substring in self.invalid_str_for_extraction):
                url = tag.nextSibling.contents[0].attrs["href"]
                yield url[:5] + quote(url[5:].encode('latin-1').decode('utf-8'))

    def get_local_path(self, url):
        url_info = url.split("/")
        sub_categories = url_info[3:-2]
        return os.path.join(*[self.parent_dir] + sub_categories + [f"{'.'.join(sub_categories)}.csv"])


def main(log_path, parent_dir):
    crawler = IABNBListingsCrawler(log_path, parent_dir)
    crawler.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inside Airbnb CSV Crawler")
    parser.add_argument("--log-path", type=str, default="crawler_log.log", help="Path to the crawler log file")
    parser.add_argument("--parent-dir", type=str, default=".", help="Parent directory to save the CSV files")

    args = parser.parse_args()
    main(args.log_path, args.parent_dir)
