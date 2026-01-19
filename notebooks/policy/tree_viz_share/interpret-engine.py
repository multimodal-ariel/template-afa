# %%
import glob
import os
import re
import time
from collections import defaultdict

import hydra as hd
import joblib
import mylib
import numpy as np
import pandas as pd
import torch as th
import tqdm.auto as tqdm
from omegaconf import OmegaConf
from sklearn.tree import _tree as skl_tree_prtdtree

# %%
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


# %%
class GlobalLogicExtractor:
    def __init__(self, model_dir, tmpls_path, feature_names=None):

        self.model_dir = model_dir
        self.tmpls = th.load(tmpls_path, map_location="cpu")
        if isinstance(self.tmpls, list):
            self.tmpls = th.stack(self.tmpls)

        self.n_features = self.tmpls.shape[1]
        self.models = self._load_models()

        if feature_names:
            self.feature_names = feature_names[: self.n_features]
        else:
            self.feature_names = [f"x_{i}" for i in range(self.n_features)]

        print(
            f"[TAFA-Interp] Loaded {len(self.models)} stage trees and {len(self.tmpls)} templates."
        )

    def _load_models(self):
        models = {}
        for filename in os.listdir(self.model_dir):
            if filename.startswith("student_") and filename.endswith(".joblib"):
                try:
                    key = filename.replace("student_", "").replace(".joblib", "")
                    model = joblib.load(os.path.join(self.model_dir, filename))
                    if hasattr(model, "tree"):
                        models[key] = model.tree
                    else:
                        models[key] = model
                except Exception as e:
                    print(f"Warning: Could not load {filename}: {e}")
        return models

    def extract(self, init_feature_idx):
        if init_feature_idx >= self.n_features:
            print(
                f"[Warning] Init feature index {init_feature_idx} >= n_features {self.n_features}. Defaulting to 0."
            )
            init_feature_idx = 0

        print(
            f"[TAFA-Interp] Starting extraction with initial feature: {self.feature_names[init_feature_idx]}"
        )

        initial_mask = np.zeros(self.n_features, dtype=int)
        initial_mask[init_feature_idx] = 1

        self.extracted_paths = []
        self._trace_ensemble(current_mask=initial_mask, current_rules=[])

        print(
            f"[TAFA-Interp] Extraction complete. Found {len(self.extracted_paths)} logical paths."
        )

    def _trace_ensemble(self, current_mask, current_rules):
        k = int(current_mask.sum())
        tree_key = f"{float(k-1)}"

        if tree_key not in self.models:
            self._record_path(current_rules, current_mask, "Max Depth / No Tree")
            return

        tree_model = self.models[tree_key]
        self._trace_single_tree(tree_model.tree_, 0, current_mask, current_rules)

    def _trace_single_tree(self, tree, node_idx, current_mask, current_rules):
        if tree.children_left[node_idx] == skl_tree_prtdtree.TREE_LEAF:
            node_value = tree.value[node_idx]
            class_idx = np.argmax(node_value)

            if class_idx >= len(self.tmpls):
                return  # Safety

            predicted_template = self.tmpls[class_idx].numpy()

            new_features_mask = np.maximum(predicted_template - current_mask, 0)

            if new_features_mask.sum() == 0:
                self._record_path(current_rules, current_mask, "Policy Stop")
            else:
                next_mask = np.maximum(current_mask, predicted_template)
                self._trace_ensemble(next_mask, current_rules)
            return

        feature_idx = tree.feature[node_idx]
        threshold = tree.threshold[node_idx]

        if feature_idx >= self.n_features:
            real_feat_idx = feature_idx - self.n_features
            mask_val = current_mask[real_feat_idx]

            if mask_val <= threshold:
                self._trace_single_tree(
                    tree, tree.children_left[node_idx], current_mask, current_rules
                )
            else:
                self._trace_single_tree(
                    tree, tree.children_right[node_idx], current_mask, current_rules
                )
            return

        is_observed = current_mask[feature_idx] == 1
        feature_name = self.feature_names[feature_idx]

        if is_observed:
            rule_left = {
                "idx": feature_idx,
                "name": feature_name,
                "op": "<=",
                "val": threshold,
            }
            self._trace_single_tree(
                tree,
                tree.children_left[node_idx],
                current_mask,
                current_rules + [rule_left],
            )

            rule_right = {
                "idx": feature_idx,
                "name": feature_name,
                "op": ">",
                "val": threshold,
            }
            self._trace_single_tree(
                tree,
                tree.children_right[node_idx],
                current_mask,
                current_rules + [rule_right],
            )

        else:
            if 0.0 <= threshold:
                self._trace_single_tree(
                    tree, tree.children_left[node_idx], current_mask, current_rules
                )
            else:
                self._trace_single_tree(
                    tree, tree.children_right[node_idx], current_mask, current_rules
                )

    def _record_path(self, rules, final_mask, stop_reason):
        indices = np.where(final_mask == 1)[0]
        outcome_set = tuple(sorted(indices.tolist()))
        self.extracted_paths.append(
            {"rules": rules, "outcome": outcome_set, "count": 0}
        )

    def validate_with_data(self, X_val):
        print(f"[TAFA-Interp] Validating logic against {len(X_val)} samples...")

        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.values

        matched_count = 0

        for path_data in self.extracted_paths:
            rules = path_data["rules"]
            matches = np.ones(len(X_val), dtype=bool)

            if rules:
                for rule in rules:
                    col_data = X_val[:, rule["idx"]]

                    if rule["op"] == "<=":
                        matches &= col_data <= rule["val"]
                    else:
                        matches &= col_data > rule["val"]

            path_data["count"] = int(matches.sum())
            matched_count += matches.sum()

        print(
            f"[TAFA-Interp] Validation coverage: {matched_count}/{len(X_val)} ({(matched_count/len(X_val)*100):.1f}%) matches."
        )

    def verify_consistency(self, X_val, init_feature_idx):
        print(f"\n[TAFA-Interp] Running Consistency Check")
        t0 = time.time()

        dynamic_outcomes = []

        init_mask = np.zeros(self.n_features, dtype=np.float32)
        init_mask[init_feature_idx] = 1.0

        X_val = X_val.astype(np.float32)

        for i in range(len(X_val)):
            x = X_val[i]
            mask = init_mask.copy()

            steps = 0
            while steps < 20:  # max 20 for cube, also work w engine dataset
                k = int(mask.sum())
                tree_key = f"{float(k-1)}"

                if tree_key not in self.models:
                    break

                tree_model = self.models[tree_key]

                # Input: [Values * Mask, Mask]
                masked_x = x * mask
                state = np.concatenate([masked_x, mask]).reshape(1, -1)

                class_idx = tree_model.predict(state)[0]
                template = self.tmpls[class_idx].numpy()

                new_features = np.maximum(template - mask, 0)
                if new_features.sum() == 0:
                    break  # Stop

                mask = np.maximum(mask, template)
                steps += 1

            indices = np.where(mask == 1)[0]
            dynamic_outcomes.append(tuple(sorted(indices.tolist())))

        static_outcomes = [None] * len(X_val)
        matched_indices = np.zeros(len(X_val), dtype=bool)

        for path_data in self.extracted_paths:
            rules = path_data["rules"]
            matches = np.ones(len(X_val), dtype=bool)

            if rules:
                for rule in rules:
                    col_data = X_val[:, rule["idx"]]
                    if rule["op"] == "<=":
                        matches &= col_data <= rule["val"]
                    else:
                        matches &= col_data > rule["val"]

            current_matches = np.where(matches)[0]
            for idx in current_matches:
                static_outcomes[idx] = path_data["outcome"]
                matched_indices[idx] = True

        # compare
        correct_features = 0
        correct_set = 0
        valid_samples = 0

        total_dynamic_len = 0
        total_static_len = 0

        mismatches = []

        for i in range(len(X_val)):
            if not matched_indices[i]:
                continue

            valid_samples += 1
            dyn = dynamic_outcomes[i]
            stat = static_outcomes[i]

            total_dynamic_len += len(dyn)
            total_static_len += len(stat)

            if len(dyn) == len(stat):
                correct_features += 1

            if dyn == stat:
                correct_set += 1
            else:
                mismatches.append((i, dyn, stat))

        t1 = time.time()
        print(f"Consistency Check Complete in {t1-t0:.2f}s")
        print(f"  Samples Compared: {valid_samples}")

        avg_dyn = total_dynamic_len / valid_samples if valid_samples > 0 else 0
        avg_stat = total_static_len / valid_samples if valid_samples > 0 else 0

        print(f"  Average Acquired Features (Dynamic Tree): {avg_dyn:.4f}")
        print(f"  Average Acquired Features (Static Rules): {avg_stat:.4f}")

        print(
            f"  Same # Features:  {correct_features}/{valid_samples} ({(correct_features/valid_samples*100):.1f}%)"
        )
        print(
            f"  Exact Set Match:  {correct_set}/{valid_samples} ({(correct_set/valid_samples*100):.1f}%)"
        )

        if mismatches:
            print(f"\n[Debug] Mismatch Details (First {min(5, len(mismatches))}):")
            for idx, dyn, stat in mismatches[:5]:
                print(f"  Sample {idx}:")
                print(f"    Dynamic (Actual): {dyn} (Len: {len(dyn)})")
                print(f"    Static (Rules)  : {stat} (Len: {len(stat)})")

        if correct_set < valid_samples:
            print("\nDiscrepancies found")
        else:
            print("Extracted logic is 100%")

    def evaluate_static_accuracy(self, X_val, y_val, classifier, device="cpu"):
        """
        Evaluates
        """
        print(f"\n[TAFA-Interp] Evaluating Predictive Accuracy of Static Rules...")

        static_outcomes = [None] * len(X_val)
        matched_indices = np.zeros(len(X_val), dtype=bool)

        for path_data in self.extracted_paths:
            rules = path_data["rules"]
            matches = np.ones(len(X_val), dtype=bool)

            if rules:
                for rule in rules:
                    col_data = X_val[:, rule["idx"]]
                    if rule["op"] == "<=":
                        matches &= col_data <= rule["val"]
                    else:
                        matches &= col_data > rule["val"]

            current_matches = np.where(matches)[0]
            for idx in current_matches:
                static_outcomes[idx] = path_data["outcome"]
                matched_indices[idx] = True

        masks = np.zeros_like(X_val)
        for i in range(len(X_val)):
            if matched_indices[i]:
                masks[i, static_outcomes[i]] = 1
            else:
                pass

        X_tensor = th.tensor(X_val, dtype=th.float32).to(device)
        M_tensor = th.tensor(masks, dtype=th.float32).to(device)
        y_tensor = th.tensor(y_val, dtype=th.long).to(device)

        # eval
        classifier.eval()
        classifier.to(device)

        with th.no_grad():
            x_masked = X_tensor * M_tensor
            try:
                logits = classifier(x_masked, M_tensor)
            except TypeError:
                logits = classifier(x_masked)

            preds = logits.argmax(dim=1)
            correct = (preds == y_tensor).sum().item()
            accuracy = correct / len(y_tensor)

        print(f"  Validation Accuracy (Static Rules): {accuracy*100:.2f}%")
        return accuracy

    def _simplify_path_constraints(self, rules):
        intervals = defaultdict(lambda: [-float("inf"), float("inf")])

        for rule in rules:
            idx = rule["idx"]
            val = rule["val"]
            op = rule["op"]

            if op == "<=":
                intervals[idx][1] = min(intervals[idx][1], val)
            elif op == ">":
                intervals[idx][0] = max(intervals[idx][0], val)

        return dict(intervals)

    def _aggregate_paths(self, paths):
        current_paths = []
        for path in paths:
            if path["count"] == 0:
                continue

            intervals = self._simplify_path_constraints(path["rules"])
            current_paths.append(
                {
                    "intervals": intervals,
                    "count": path["count"],
                    "outcome": path["outcome"],
                }
            )

        while True:
            merged_occurred = False
            next_pass_paths = []
            used_indices = set()
            n = len(current_paths)

            for i in range(n):
                if i in used_indices:
                    continue

                path_a = current_paths[i]
                merged_a = False

                for j in range(i + 1, n):
                    if j in used_indices:
                        continue

                    path_b = current_paths[j]

                    all_feats = set(path_a["intervals"].keys()) | set(
                        path_b["intervals"].keys()
                    )
                    diff_feats = []

                    for f in all_feats:
                        range_a = path_a["intervals"].get(
                            f, [-float("inf"), float("inf")]
                        )
                        range_b = path_b["intervals"].get(
                            f, [-float("inf"), float("inf")]
                        )

                        if not (
                            np.isclose(range_a[0], range_b[0], atol=1e-6)
                            and np.isclose(range_a[1], range_b[1], atol=1e-6)
                        ):
                            diff_feats.append(f)

                    if len(diff_feats) == 1:
                        f_diff = diff_feats[0]
                        range_a = path_a["intervals"].get(
                            f_diff, [-float("inf"), float("inf")]
                        )
                        range_b = path_b["intervals"].get(
                            f_diff, [-float("inf"), float("inf")]
                        )

                        new_range = None
                        if np.isclose(range_a[1], range_b[0], atol=1e-6):
                            new_range = [range_a[0], range_b[1]]
                        elif np.isclose(range_b[1], range_a[0], atol=1e-6):
                            new_range = [range_b[0], range_a[1]]

                        if new_range is not None:
                            merged_intervals = path_a["intervals"].copy()
                            merged_intervals[f_diff] = new_range

                            new_path = {
                                "intervals": merged_intervals,
                                "count": path_a["count"] + path_b["count"],
                                "outcome": path_a["outcome"],
                            }

                            next_pass_paths.append(new_path)
                            used_indices.add(i)
                            used_indices.add(j)
                            merged_occurred = True
                            merged_a = True
                            break

                if not merged_a:
                    next_pass_paths.append(path_a)

            current_paths = next_pass_paths
            if not merged_occurred:
                break

        return current_paths

    def generate_report(self):
        grouped = defaultdict(list)
        total_obs = 0
        total_features_acquired = 0

        for path in self.extracted_paths:
            if path["count"] > 0:
                grouped[path["outcome"]].append(path)
                count = path["count"]
                total_obs += count
                total_features_acquired += count * len(path["outcome"])

        report_lines = ["TAFA-Interp Report", "=" * 30]

        avg_features = total_features_acquired / total_obs if total_obs > 0 else 0
        report_lines.append(f"Total Validation Samples: {total_obs}")
        report_lines.append(f"Average Acquired Features: {avg_features:.4f}")
        report_lines.append("=" * 30)

        sorted_outcomes = sorted(
            grouped.items(), key=lambda x: sum(p["count"] for p in x[1]), reverse=True
        )

        for outcome, raw_paths in sorted_outcomes:
            # Aggregate Paths first
            agg_paths = self._aggregate_paths(raw_paths)

            cnt = sum(p["count"] for p in agg_paths)
            prob = (cnt / total_obs * 100) if total_obs else 0
            names = [self.feature_names[i] for i in outcome]

            report_lines.append(
                f"\nACQUIRE (Size: {len(outcome)}): {{ {', '.join(names)} }} (Freq: {prob:.1f}%)"
            )
            agg_paths.sort(key=lambda x: x["count"], reverse=True)

            for i, path in enumerate(agg_paths):
                p_prob = path["count"] / total_obs * 100

                # Format Intervals
                intervals = path["intervals"]
                if not intervals:
                    rule_str = "DEFAULT (Always True)"
                else:
                    rule_strs = []
                    # Sort by feature index
                    for f_idx in sorted(intervals.keys()):
                        f_name = self.feature_names[f_idx]
                        rng = intervals[f_idx]
                        min_v, max_v = rng[0], rng[1]

                        if min_v > -float("inf") and max_v < float("inf"):
                            rule_strs.append(f"{min_v:.4f} < {f_name} <= {max_v:.4f}")
                        else:
                            parts = []
                            if min_v > -float("inf"):
                                parts.append(f"{f_name} > {min_v:.4f}")
                            if max_v < float("inf"):
                                parts.append(f"{f_name} <= {max_v:.4f}")

                            if parts:
                                rule_strs.append(" AND ".join(parts))

                    rule_str = (
                        " AND ".join(rule_strs)
                        if rule_strs
                        else "DEFAULT (Always True)"
                    )

                report_lines.append(f"  Rule {i+1} ({p_prob:.1f}%): IF {rule_str}")

        return "\n".join(report_lines)


# %%
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
            shuffle_idxs = th.load(shuffle_idxs_path)

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
        state_dict = th.load(classifier_ckpt_path, map_location="cpu")
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
        mylib.utils.get_project_root_dir(),
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

    tmpls = th.load(tmpls_path, map_location="cpu")
    if isinstance(tmpls, list):
        tmpls = th.stack(tmpls)
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
            device="cuda" if th.cuda.is_available() else "cpu",
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

    output_dir = os.path.join(os.getcwd(), dataset_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{exp_name}.txt")

    with open(output_path, "w") as f:
        f.write(full_output)

    print(f"[Success] Results saved to {output_path}")


# %%
MODELS_PARENT_DIR = "notebooks/policy/tree_viz_share/models"
SEARCH_PATTERN = "engine-fault_l4"
TEMPLATE_DATASET = "engine-cnnet"
TEMPLATE_TIMESTAMP = "20260110_215609"
MODELS_PARENT_DIR = os.path.join(mylib.utils.get_project_root_dir(), MODELS_PARENT_DIR)

# %%
search_path = os.path.join(MODELS_PARENT_DIR, f"{SEARCH_PATTERN}*")
model_dirs = sorted(glob.glob(search_path))

for model_dir in tqdm.tqdm(model_dirs):
    if os.path.isdir(model_dir):
        try:
            process_model(model_dir)
        except Exception as e:
            import traceback

            traceback.print_exc()


# %%
