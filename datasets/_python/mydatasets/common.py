import os
from dataclasses import dataclass


@dataclass
class DatasetConf:
    _target_: str
    dataset_type: str


def get_datasets_files_root_dir() -> str:
    package_dir: str = os.path.dirname(__file__)
    datasets_files_dir: str = os.path.join(package_dir, "..", "..", "_files")
    datasets_files_dir = os.path.abspath(datasets_files_dir)
    if not os.path.exists(datasets_files_dir):
        raise FileNotFoundError("datasets/_files does not exists")
    return datasets_files_dir
