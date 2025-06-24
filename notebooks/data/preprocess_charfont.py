# %%
from __future__ import annotations

import os
import pickle as pkl

import mydatasets
import numpy as np
import pandas as pd
import torch as th
import torch.utils.data as th_data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# %%
filenames_l: list[str] = list(
    filter(
        lambda p: str.lower(os.path.splitext(p)[-1]) == ".csv",
        os.listdir("character-font"),
    )
)
filenames_l.sort()

# %%
rdata_df: pd.DataFrame = pd.concat(
    [
        pd.read_csv(os.path.join("character-font", _fn)).infer_objects()
        for _fn in filenames_l
    ]
)

# %%
print(f"DataFrame shape: {rdata_df.shape}")
print(f"Data types overview:")
print(rdata_df.dtypes.value_counts())

# %%
# Preprocessing: Encode m_label to start from 0
label_encoder = LabelEncoder()
y: np.ndarray = label_encoder.fit_transform(rdata_df["m_label"])
print(
    f"Original m_label range: {rdata_df['m_label'].min()} to {rdata_df['m_label'].max()}"
)
print(f"Encoded y range: {y.min()} to {y.max()}")
print(f"Number of classes: {len(np.unique(y))}")

# %%
# Separate features
object_features = ["font", "fontVariant"]
numeric_features = [
    col for col in rdata_df.columns if col not in object_features + ["m_label"]
]

print(f"Object features: {object_features}")
print(f"Number of numeric features: {len(numeric_features)}")

# %%
# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(drop="first", sparse_output=False), object_features),
        ("passthrough", "passthrough", numeric_features),
    ]
)

# %%
# Fit and transform the features
X_processed: np.ndarray = preprocessor.fit_transform(rdata_df)
print(f"Original feature shape: {rdata_df.shape}")
print(f"Processed feature shape: {X_processed.shape}")

# %%
# Convert to numpy arrays
X = X_processed.astype(np.float32)
y = y.astype(np.int64)

print(f"Final X shape: {X.shape}, dtype: {X.dtype}")
print(f"Final y shape: {y.shape}, dtype: {y.dtype}")
print(f"X range: {X.min():.3f} to {X.max():.3f}")
print(f"y range: {y.min()} to {y.max()}")

# %%
tidxs, vidxs, tstidxs = [
    _subset.indices
    for _subset in th_data.random_split(
        th.arange(len(X)), (0.8, 0.1, 0.1), generator=th.Generator().manual_seed(279)
    )
]
with open(
    os.path.join(
        mydatasets.common.get_datasets_files_root_dir(), "aaco", "charfont.pkl"
    ),
    mode="wb",
) as f:
    pkl.dump(
        {
            "train": (X[tidxs], y[tidxs]),
            "valid": (X[vidxs], y[vidxs]),
            "test": (X[tstidxs], y[tstidxs]),
        },
        f,
    )

# %%
tdata, vdata, tstdata = mydatasets.aaco.load_aaco_data("charfont", to_normalize=True)

# %%
# Optional: Save preprocessed data
# np.save('X_charfont.npy', X)
# np.save('y_charfont.npy', y)
# print("Preprocessed data saved!")

# %%
