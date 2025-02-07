import csv
import os
from typing import Literal
import mydatasets.common


def load_template_dataset(dataset_type: Literal["train", "test", "val"]):
    datasets_root_dir: str = mydatasets.common.get_datasets_files_root_dir()
    dataset_p: str = os.path.join(datasets_root_dir, "template", f"{dataset_type}.csv")
    dataset: list[dict[str, str]] = list()
    with open(dataset_p, "r") as f:
        reader = csv.DictReader(f, ["model", "plate"])
        dataset = [r for r in reader]
    return dataset
