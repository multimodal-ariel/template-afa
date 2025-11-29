# %%
from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path
import pickle as pkl
from typing import Any, Callable, Optional

import mydatasets
import tensordict as thd
import torch as th
import torch.utils.data as th_data
import torchvision as thv
import PIL.Image

# %%
os.makedirs("tmp", exist_ok=True)


# %%
class OmniglotLanguage(thv.datasets.VisionDataset):
    language_to_image_paths: dict[str, list[str]]
    languages: list[str]
    img_label_pairs: list[tuple[str, int]]

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(
            os.path.join(root, thv.datasets.Omniglot.folder),
            transform=transform,
            target_transform=None,
        )
        # takes care of both download and check integrity
        thv.datasets.Omniglot(root, background=True, download=download)
        thv.datasets.Omniglot(root, background=False, download=download)
        # acquire all languages
        language_to_image_paths: dict[str, list[str]] = defaultdict(list)
        [
            language_to_image_paths[_lang].extend(
                list(
                    filter(
                        lambda _p: os.path.isfile(_p)
                        and os.path.splitext(_p)[-1] == ".png",
                        [
                            os.path.abspath(
                                os.path.join(self.root, _folder, _lang, _character, _fn)
                            )
                            for _fn in sorted(
                                os.listdir(
                                    os.path.join(self.root, _folder, _lang, _character)
                                )
                            )
                        ],
                    )
                )
            )
            for _folder in ("images_background", "images_evaluation")
            for _lang in sorted(os.listdir(os.path.join(self.root, _folder)))
            for _character in sorted(
                os.listdir(os.path.join(self.root, _folder, _lang))
            )
        ]
        # language_to_image_paths: dict[str, list[str]] = defaultdict(list)
        # [
        #     language_to_image_paths[_lang].extend(
        #         os.path.abspath(
        #             os.path.join(self.root, _folder, _lang, _character, _fn)
        #         )
        #         for _fn in sorted(
        #             os.listdir(os.path.join(self.root, _folder, _lang, _character))
        #         )
        #     )
        #     for _folder in ("images_background", "images_evaluation")
        #     for _lang in sorted(os.listdir(os.path.join(self.root, _folder)))
        #     for _character in sorted(
        #         os.listdir(os.path.join(self.root, _folder, _lang))
        #     )
        # ]
        # sort language according to class path
        language_to_image_paths = {
            _lang: language_to_image_paths[_lang]
            for _lang in sorted(language_to_image_paths)
        }
        # class label to langauge
        languages: list[str] = list(language_to_image_paths.keys())
        # index to img label pair
        img_label_pairs = [
            (_imgp, _c)
            for _c, (_lang, _imgps) in enumerate(language_to_image_paths.items())
            for _imgp in _imgps
        ]
        # asseign to instance variable
        self.language_to_image_paths = language_to_image_paths
        self.languages = languages
        self.img_label_pairs = img_label_pairs

    def __len__(self) -> int:
        return len(self.img_label_pairs)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_path, image_label = self.img_label_pairs[index]
        image = PIL.Image.open(image_path, mode="r").convert("L")
        if self.transform:
            image = self.transform(image)
        return image, image_label


# %%
# _data = OmniglotLanguage(root="./tmp", download=True)
_data = OmniglotLanguage(
    root="./tmp",
    transform=thv.transforms.Compose(
        [
            thv.transforms.PILToTensor(),
            thv.transforms.Grayscale(),
            thv.transforms.Resize((64, 64)),
        ]
    ),
    download=True,
)

# %%
_tdata, _vdata, _tstdata = th_data.random_split(
    _data, (0.7, 0.2, 0.1), th.Generator().manual_seed(279)
)

# %%
tdata: thd.TensorDict = thd.cat(  # type:ignore
    [
        thd.make_tensordict(
            {"xs": _x[:, 0].flatten(1, 2), "ys": th.as_tensor(_y)},
            batch_size=(len(_x),),
        )
        for _x, _y in th_data.DataLoader(
            _tdata,
            batch_size=36,
            shuffle=False,
            drop_last=False,
            num_workers=36,
        )
    ]
)
vdata: thd.TensorDict = thd.cat(  # type:ignore
    [
        thd.make_tensordict(
            {"xs": _x[:, 0].flatten(1, 2), "ys": th.as_tensor(_y)},
            batch_size=(len(_x),),
        )
        for _x, _y in th_data.DataLoader(
            _vdata,
            batch_size=36,
            shuffle=False,
            drop_last=False,
            num_workers=36,
        )
    ]
)
tstdata: thd.TensorDict = thd.cat(  # type:ignore
    [
        thd.make_tensordict(
            {"xs": _x[:, 0].flatten(1, 2), "ys": th.as_tensor(_y)},
            batch_size=(len(_x),),
        )
        for _x, _y in th_data.DataLoader(
            _tstdata,
            batch_size=8,
            shuffle=False,
            drop_last=False,
            num_workers=36,
        )
    ]
)

# %%
with open(
    os.path.join(
        mydatasets.common.get_datasets_files_root_dir(), "aaco", "omniglot.pkl"
    ),
    mode="wb",
) as f:
    pkl.dump(
        {
            "train": (tdata["xs"].numpy(), tdata["ys"].numpy()),
            "valid": (vdata["xs"].numpy(), vdata["ys"].numpy()),
            "test": (tstdata["xs"].numpy(), tstdata["ys"].numpy()),
        },
        f,
    )

# %%
