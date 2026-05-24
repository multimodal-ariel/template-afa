# Information Templates: A New Paradigm for Intelligent Active Feature Acquisition

Code for **Template-based Active Feature Acquisition (TAFA)**, a non-greedy active feature acquisition framework that learns a small library of informative feature subsets ("templates") and uses them to guide inference-time acquisition.

The core loop is:

```text
predictor -> template search/refinement -> template policy -> feature acquisition -> prediction -> optional interpretable distillation
```

## Authors

#### Hung-Tien Huang, Dzung Dinh, Junier B. Oliva
#### Department of Computer Science, University of North Carolina at Chapel Hill

Paper metadata: arXiv `2508.18380v2`. 

## High-level flow

1. **Predictor**: train or load a model that can predict using arbitrary observed feature subsets.
2. **Template search**: build a library of feature templates that are jointly informative under a loss-plus-cost objective.
3. **Mutation-guided greedy optimization**: iteratively mutate promising templates and select the best candidates.
4. **Continuous refinement**: optionally refine templates and actors with a Gumbel-Softmax relaxation.
5. **Template policy**: choose a template for the current instance and acquire missing features from that template.
6. **Interpretable distillation**: distill template policies into step-wise decision trees with explicit acquisition rules.
7. **Benchmarking**: compare against RL-free and static AFA baselines on accuracy, acquisition cost, and runtime.

## Repo layout

- `datasets/` package for dataset loaders and cached benchmark files.
- `datasets/_files/` local cached data files used by experiments.
- `models/` package with subset-feature classifiers, neural nets, and model protocols.
- `libs/internal/tafalib/` core TAFA utilities for template generation, cost estimates, rollout, and evaluation.
- `libs/internal/mylib/` shared local helpers.
- `libs/external/` local copies/wrappers for baseline methods such as AACO/ACO, DiFA, DIME, JAFA, and SEFA.
- `experiments/make_template/` template-bank training and evaluation.
- `experiments/policy/` policy learning and student distillation experiments.
- `experiments/baselines/` reproduction scripts/configs for baseline methods.
- `experiments/pretrain/` subset-predictor pretraining experiments.
- `notebooks/` data preparation, exploration, visualization, and analysis notebooks/scripts.
- `requirements.txt` Python dependencies.
- `setup.sh` install script for all local packages.

## Main components

- **Template search**: implemented primarily in `libs/internal/tafalib/makers/` and launched through `experiments/make_template/train.py`.
- **kNN template policy**: scores templates using nearest-neighbor estimates of loss plus remaining acquisition cost.
- **Actor template policy**: learns a differentiable actor over templates using Gumbel-Softmax.
- **Interpretable TAFA**: trains step-wise decision-tree policies from expert rollouts.
- **Baselines**: includes static acquisition, DIME, DiFA, ACO/AACO-style methods, JAFA, and SEFA-related code.

## Quick start

```bash
conda create -n tmplafa python=3.13
conda activate tmplafa
./setup.sh
```

`setup.sh` installs dependencies and then installs local packages in editable mode:

```text
datasets -> mydatasets
models -> mymodels
libs/internal -> mylib, tafalib
libs/external -> baseline libraries
```

## Running a template experiment

Most experiments are Hydra-based. A small starting point is the cube benchmark:

```bash
cd experiments/make_template
python train.py -m -cp=conf/cube -cn=startup hydra/launcher=joblib
```

Common dataset config directories include:

```text
cube/
```

Swap `conf/cube` and `startup` for the dataset and config variant you want to run.

## Outputs

Hydra writes run artifacts under the active output directory. Template runs save files such as:

- `.hydra/config.yaml`
- `tmpls.pt`
- `tpcomp.pt`
- `tdata_shuffle_idxs.pt`
- `tclassifier.pt`
- `vclassifier.pt`
- TensorBoard logs
- CSV logs

## Notes
- Some notebooks and configs expect local cached datasets or credentials for external downloads.
- To add another package as a submodule, use `git submodule add git://github.com/{username}/{repo_name} third_party/{repo_name}`.
