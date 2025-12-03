# %%
from __future__ import annotations

import os
import pickle as pkl

import mydatasets
import pandas as pd
import tensordict as thd
import torch as th
import torch.utils.data as th_data

# %%
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10311597
# https://github.com/Leo-Thomas/EngineFaultDB
# Regarding the variables of the dataset, these are:
# Manifold Absolute Pressure (MAP): A measure of the pressure within the intake manifold, crucial for the engine's electronic control system to regulate fuel injection and ignition timing, measured in kilopascals (kPa).
# Throttle Position Sensor (TPS): Provides information about the position of the throttle, influencing fuel injection, ignition timing, and other engine parameters. It is measured as a percentage.
# Force: Represents the engine's torque or rotational force. It is measured in newtons (N).
# Power: Quantifies the rate at which work is done or energy is transferred in the engine. It is measured in kilowatts (kW).
# Revolutions Per Minute (RPM): Indicates the engine's speed, detailing how many times the engine's crankshaft rotates per minute.
# Fuel consumption L/H: Illustrates the engine's fuel consumption rate.
# Fuel consumption L/100KM: Relays the engine's fuel efficiency over a given distance.
# Speed (km/h): The vehicle's travel speed.
# Carbon monoxide (CO): CO concentration in the exhaust gases. It is measured as a percentage.
# Hydrocarbons (HC): Concentration of unburnt hydrocarbons in the exhaust. It is measured in parts per million (ppm).
# Carbon dioxide (CO2): CO2 concentration in the exhaust, indicative of combustion efficiency. It is measured as a percentage.
# Oxygen (O2): Oxygen amount in the exhaust, offering insights into the combustion process. It is measured as a percentage.
# Lambda: The air-fuel equivalence ratio.
# Air-Fuel Ratio (AFR): Ratio of air to fuel in the combustion chambers.
# The percentage distribution of data according to each fault type is as follows:
data_df: pd.DataFrame = pd.read_csv("engine-failure/EngineFaultDB_Final.csv")

# %%
feature_columns = [
    "MAP",
    "TPS",
    "Force",
    "Power",
    "RPM",
    "Consumption L/H",
    "Consumption L/100KM",
    "Speed",
    "CO",
    "HC",
    "CO2",
    "O2",
    "Lambda",
    "AFR",
]
target_columns = [
    "Fault",
]

# %%
# stdsclr = skl_preproc.StandardScaler()
# xs: th.Tensor = th.as_tensor(stdsclr.fit_transform(data_xs_df), dtype=th.float32)
# ys: th.Tensor = th.as_tensor(data_df[target_columns].to_numpy(), dtype=th.long)
xs: th.Tensor = th.as_tensor(data_df[feature_columns].to_numpy(), dtype=th.float32)
ys: th.Tensor = th.as_tensor(data_df[target_columns].to_numpy(), dtype=th.long)

# %%
data = thd.make_tensordict({"xs": xs, "ys": ys}, batch_size=(len(data_df),))
rg = th.Generator().manual_seed(279)
tidxs, vidxs, tstidxs = [
    _subset.indices
    for _subset in th_data.random_split(
        th.arange(len(data_df)),
        (0.8, 0.1, 0.1),
        generator=rg,
    )
]
tdata = data[tidxs]
vdata = data[vidxs]
tstdata = data[tstidxs]

# %%
with open(
    os.path.join(
        mydatasets.common.get_datasets_files_root_dir(), "aaco", "engine-fault.pkl"
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
