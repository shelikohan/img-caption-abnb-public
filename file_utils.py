import os
import glob
import json
import pickle
import subprocess
from abc import abstractmethod, ABC
import psutil


def makedirs_if_not_exists(path):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)


def get_all_sub_files(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    return file_list


def extract_csv_files(directory):
    csv_files = []
    for root, dirs, files in os.walk(directory):
        csv_files.extend(glob.glob(os.path.join(root, '*.csv')))
    return csv_files


class FileHandler(ABC):
    def __init__(self, path, default_output=None, logger=None):
        self.path = path
        self.logger = logger
        self.default_output = default_output
        self.data = self.load()

    def load(self):
        try:
            with open(self.path, self.r) as f:
                return self._load(f)
        except FileNotFoundError:
            if self.logger:
                self.logger.info(f"could not find existing {self.path}, returns {self.default_output}")
            return self.default_output

    def save(self, content):
        with open(self.path, self.w) as f:
            if self.logger:
                self.logger.info(f"saved pickle to {self.path}")
            self._save(content, f)

    @abstractmethod
    def _save(self, content, file):
        pass

    @abstractmethod
    def _load(self, file):
        pass

    @property
    @abstractmethod
    def w(self):
        pass

    @property
    @abstractmethod
    def r(self):
        pass

    def save_data(self):
        self.save(self.data)

def get_gpu_memory_usage():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
        memory_used = [int(x) for x in output.decode().strip().split('\n')]
        return memory_used
    except: #no gpu
        return None


def get_cpu_memory_usage():
    process = psutil.Process()
    memory_usage = process.memory_info().rss
    return memory_usage


def get_ram_usage():
    ram = psutil.virtual_memory()
    memory_usage = ram.used
    return memory_usage


class PickleFileHandler(FileHandler):
    """
    save/load safely from pickles
    """
    w = "wb"
    r = "rb"

    def _load(self, file):
        return pickle.load(file)

    def _save(self, content, file):
        pickle.dump(content, file)


class JsonFileHandler(FileHandler):
    """
    save/load safely from pickles
    """
    w = "w"
    r = "r"

    def _load(self, file):
        return json.load(file)

    def _save(self, content, file):
        file.write(json.dumps(content))
