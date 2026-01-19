import glob
import os
import re
import sys

import hydra as hd
import mylib
import numpy as np
import torch
from omegaconf import OmegaConf
from tafa_interpreter import GlobalLogicExtractor

SEARCH_PATTERN = "engine-fault_l4"
TEMPLATE_DATASET = "engine-cnnet"
TEMPLATE_TIMESTAMP = "20260110_215609"

# Path config
PROJ_ROOT = mylib.utils.get_project_root_dir()
MODELS_PARENT_DIR = os.path.join(
    mylib.utils.get_project_root_dir(), "notebooks/policy/tree_viz_share/models"
)


ENGINE_FEATURE_NAMES = [
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

if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)


try:
    OmegaConf.register_new_resolver(
        name="get_cls", resolver=lambda cls: hd.utils.get_class(cls), replace=True
    )
except Exception as e:
    pass


def load_init_fidx(cfg):
    if "init_fidx" in cfg:
        val = int(cfg.init_fidx)
        print(f"[Setup] Found init_fidx: {val}")
        return val
    else:
        print("[Warning] 'init_fidx' not found in config")
        return 0


def load_real_training_data(cfg, shuffle_idxs_path):
    print(f"[Setup] Loading dataset using configuration...")
    try:
        # 1. Load full dataset using hydra
        # This returns (_tdata, vdata, tstdata)
        _tdata, vdata, tstdata = hd.utils.call(cfg.data)

        # 2. Load shuffle indices
        if os.path.exists(shuffle_idxs_path):
            print(f"[Setup] Loading shuffle indices from {shuffle_idxs_path}")
            shuffle_idxs = torch.load(shuffle_idxs_path)

            # 3. Extract the training subset (tdata)
            cutoff = len(_tdata) // 2
            tdata_idxs = shuffle_idxs[:cutoff]
            tdata = _tdata[tdata_idxs]

            print(f"[Setup] Successfully extracted 'tdata' with {len(tdata)} samples.")

            # Helper to convert to numpy
            X = (
                tdata["xs"].numpy()
                if hasattr(tdata["xs"], "numpy")
                else np.array(tdata["xs"])
            )
            y = (
                tdata["ys"].numpy()
                if hasattr(tdata["ys"], "numpy")
                else np.array(tdata["ys"])
            )

            return X, y
        else:
            print(
                f"[Warning] Shuffle indices not found at {shuffle_idxs_path}. Using full _tdata."
            )
            X = (
                _tdata["xs"].numpy()
                if hasattr(_tdata["xs"], "numpy")
                else np.array(_tdata["xs"])
            )
            y = (
                _tdata["ys"].numpy()
                if hasattr(_tdata["ys"], "numpy")
                else np.array(_tdata["ys"])
            )
            return X, y

    except Exception as e:
        print(f"[Error] Failed to load real training data: {e}")
        return None, None


def load_classifier(cfg, classifier_ckpt_path):
    print(f"[Setup] Loading classifier from {classifier_ckpt_path}...")
    try:
        if not os.path.exists(classifier_ckpt_path):
            print(f"[Error] Classifier checkpoint not found.")
            return None

        # Instantiate using Hydra config
        classifier = hd.utils.call(cfg.classifier)

        # Load weights
        state_dict = torch.load(classifier_ckpt_path, map_location="cpu")
        classifier.load_state_dict(state_dict)

        print(f"[Setup] Classifier loaded successfully.")
        return classifier
    except Exception as e:
        print(f"[Error] Failed to load classifier: {e}")
        return None


def process_model(model_dir):
    exp_name = os.path.basename(model_dir)
    print(f"\n{'='*60}")
    print(f"Processing: {exp_name}")
    print(f"{'='*60}")

    match = re.search(r"accuracy_(\d+)$", exp_name)
    if not match:
        return

    base_idx = match.group(1)
    print(f"[Info] Detected base_idx: {base_idx}")

    mktmpl_dir = os.path.join(
        PROJ_ROOT,
        f"experiments/make_template/outputs/{TEMPLATE_DATASET}/{TEMPLATE_TIMESTAMP}/{base_idx}",
    )

    tmpls_path = os.path.join(mktmpl_dir, "tmpls.pt")
    config_path = os.path.join(mktmpl_dir, ".hydra", "config.yaml")
    shuffle_idxs_path = os.path.join(mktmpl_dir, "tdata_shuffle_idxs.pt")
    classifier_ckpt_path = os.path.join(mktmpl_dir, "classifier.pt")

    if not os.path.exists(config_path):
        print(f"[Skip] Config file not found at {config_path}")
        return

    cfg = OmegaConf.load(config_path)

    if not os.path.exists(tmpls_path):
        print(f"[Skip] Templates file not found at {tmpls_path}")
        return

    tmpls = torch.load(tmpls_path, map_location="cpu")
    if isinstance(tmpls, list):
        tmpls = torch.stack(tmpls)
    n_features = tmpls.shape[1]

    init_fidx = load_init_fidx(cfg)

    if "engine" in exp_name.lower():
        feature_names = ENGINE_FEATURE_NAMES
        if len(feature_names) < n_features:
            feature_names += [f"x_{i}" for i in range(len(feature_names), n_features)]
    else:
        feature_names = [f"x_{i}" for i in range(n_features)]

    interpreter = GlobalLogicExtractor(
        model_dir=model_dir, tmpls_path=tmpls_path, feature_names=feature_names
    )

    # extract rule
    interpreter.extract(init_feature_idx=init_fidx)

    # load
    X_val, y_val = load_real_training_data(cfg, shuffle_idxs_path)

    # validate
    interpreter.validate_with_data(X_val)
    interpreter.verify_consistency(X_val, init_fidx)

    # eval
    classifier = load_classifier(cfg, classifier_ckpt_path)
    accuracy_str = ""
    if classifier:
        acc = interpreter.evaluate_static_accuracy(
            X_val,
            y_val,
            classifier,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        accuracy_str = f"\nValidation Accuracy (Static Rules): {acc*100:.2f}%"

    report = interpreter.generate_report()
    full_output = report + accuracy_str

    # Extract dataset name
    parts = exp_name.split("_")
    if len(parts) >= 3:
        dataset_folder_name = "_".join(parts[:3])  # e.g. cube_20_0.3
    else:
        dataset_folder_name = parts[0]

    output_dir = os.path.join(os.getcwd(), "outputs", dataset_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{exp_name}.txt")

    with open(output_path, "w") as f:
        f.write(full_output)

    print(f"[Success] Results saved to {output_path}")


def main():
    print(f"Searching for models in {MODELS_PARENT_DIR}")
    print(f"Matching pattern: '{SEARCH_PATTERN}*'")

    search_path = os.path.join(MODELS_PARENT_DIR, f"{SEARCH_PATTERN}*")
    model_dirs = sorted(glob.glob(search_path))

    for model_dir in model_dirs:
        if os.path.isdir(model_dir):
            try:
                process_model(model_dir)
            except Exception as e:
                import traceback

                traceback.print_exc()


if __name__ == "__main__":
    main()
