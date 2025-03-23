# %%
import collections
import itertools
import os


# %%
def safe_makedirs(path_):
    if not os.path.exists(path_):
        try:
            os.makedirs(path_)
        except FileExistsError:
            pass


create_dirs = ["configs/"]
for c in create_dirs:
    safe_makedirs(c)


def get_gpu(combo):
    if combo["problem"] == "eddi":
        return "gypsum-1080ti-phd"
    if combo["data"] in {"parkinsons", "physio-mortality", "grid"}:
        return "gypsum-1080ti-phd"
    if combo["data"] == "stl" and combo["problem"] == "gsmrl":
        return "gypsum-m40-phd"
    return "gpu-long"


def get_time(combo):
    if "gpu" not in combo:
        combo["gpu"] = get_gpu(combo)
    #
    if "gypsum" in combo["gpu"]:
        return "7-00"
    return "14-00"


def get_memory(combo):
    return 30000


def get_cpu(combo):
    return 3


class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


def get_run_id():
    
    filename = "logs/expts.txt"
    if os.path.isfile(filename) is False:
        with open(filename, "w") as f:
            f.write("")
        return 0
    else:
        with open(filename, "r") as f:
            expts = f.readlines()
        run_id = len(expts)
    return run_id


def is_valid(combo):
    # Exceptions for n_features
    if combo["problem"] == "vaeac":
        if combo["n_features"] != -1:
            return False
    if combo["problem"] in {"jafa", "gsmrl", "difa", "random"}:
        valid_features = n_features[combo["data"]]
        if combo["n_features"] not in valid_features:
            return False
    if combo["problem"] == "eddi":
        valid_features = max(n_features[combo["data"]])
        if combo["n_features"] != valid_features:
            return False
    return True


def get_fixed_params(combo):
    fixed_params = ["--neptune", "--cuda"]
    if combo["problem"] == "vaeac":
        fixed_params.append("--save")
    if combo["data"] in {
        "mnist",
        "cifar10",
        "cifar100",
        "fashionmnist",
        "svhn",
        "stl",
        "food",
    }:
        fixed_params.append("--cnn")
    if combo["data"] in {"cifar10", "cifar100", "svhn", "stl", "food"}:
        fixed_params.append("--augmentation")
    if combo["data"] in {"bank", "physio-mortality", "adult", "in-hospital-mortality"}:
        fixed_params.append("--use_f1_score")
    return "  ".join(fixed_params)


def set_problem_specific_params(combo):
    # set iters
    if combo["problem"] in n_iters:
        combo["iters"] = n_iters[combo["problem"]]
    elif combo["data"] in n_iters:
        combo["iters"] = n_iters[combo["data"]]
    else:
        combo["iters"] = n_iters["default"]

    # set batch_size
    combo["batch_size"] = bss[combo["data"]] if combo["data"] in bss else bss["default"]

    # set imputation model
    if combo["problem"] in {"vaeac", "jafa"}:
        combo["imputation_model"] = "none"
    else:
        combo["imputation_model"] = "models/{}.pt".format(
            imputation_models[combo["data"]]
        )

    # set pretrainiters
    if combo["problem"] == "vaeac":
        combo["pretrain_iters"] = -1
    else:
        combo["pretrain_iters"] = (
            pretrain_iters[combo["data"]]
            if combo["data"] in pretrain_iters
            else pretrain_iters["default"]
        )

    # set weights
    combo["weight"] = (
        weights[combo["data"]] if combo["data"] in weights else weights["default"]
    )

    # set lr
    if combo["problem"] == "vaeac":
        combo["lr"] = lrs["default"]
    else:
        combo["lr"] = lrs[combo["data"]] if combo["data"] in lrs else lrs["default"]

    # set policy lr
    combo["policy_lr"] = (
        policy_lrs[combo["data"]]
        if combo["data"] in policy_lrs
        else policy_lrs["default"]
    )

    # set policy base lr
    combo["policy_base_lr"] = (
        policy_base_lrs[combo["data"]]
        if combo["data"] in policy_base_lrs
        else policy_base_lrs["default"]
    )

    # set grad norms
    combo["grad_norm"] = (
        grad_norms[combo["data"]]
        if combo["data"] in grad_norms
        else grad_norms["default"]
    )


# %%
policy_lrs = {
    "default": 2e-4,
    "cifar10": 1e-2,
    "cifar100": 1e-2,
    "stl": 2e-3,
    "food": 1e-2,
    "mnist": 1e-2,
    "fashionmnist": 1e-2,
}
bss = {
    "default": 512,
    "cifar10": 256,
    "stl": 64,
    "food": 256,
    "cifar100": 256,
    "svhn": 256,
}

policy_base_lrs = {
    "default": 1e-5,
}
lrs = {
    "default": 1e-4,
    "cifar10": 1e-3,
    "svhn": 1e-3,
    "stl": 1e-3,
    "food": 1e-3,
    "cifar100": 1e-3,
    "mnist": 1e-2,
    "fashionmnist": 1e-2,
    "parkinsons": 2e-4,
}

pretrain_iters = {
    "default": 2000,
    "parkinsons": 2000,
    "gas": 100,
    "cifar10": 200,
    "stl": 100,
    "food": 200,
    "svhn": 200,
    "cifar100": 200,
    "mnist": 300,
    "fashionmnist": 300,
}
grad_norms = {
    "default": 10.0,
    "cifar10": 100,
    "svhn": 100,
    "stl": 100,
    "food": 100,
    "cifar100": 100,
}
n_iters = {
    "vaeac": 1000,
    "eddi": 1,
    "random": 10,
    "default": 5000,
    "gas": 20000,
    "parkinsons": 20000,
    "cifar10": 1000,
    "stl": 1000,
    "food": 1000,
    "svhn": 1000,
    "cifar100": 1000,
    "mnist": 4000,
    "fashionmnist": 2000,
}
n_features = {
    # "gas": [2, 3, 4],
    "grid": [4, 6, 8],
    # "mnist16": [8, 16, 32],
    "mnist": [12, 24],
    "fashionmnist": [32, 64],
    "cifar10": [64, 128],
    "svhn": [64, 128],
    "stl": [32, 64, 128],
    "food": [32, 64, 128],
    "cifar100": [32, 64, 128],
    "parkinsons": [6, 8, 10],
    "physio-mortality": [4, 8, 12],
    "in-hospital-mortality": [4, 6, 8, 10],
    # "wine": [2, 4, 6, 8],
    # "adult": [2, 3, 4],
    # "bank": [2, 3, 4],
    # "blog": [4, 8, 16, 24],
}
imputation_models = {
    "gas": "2987533",
    "mnist16": "2996256",
    "cifar10": "2990878",
    "wine": "2987536",
    "grid": "2987534",
    "physio-mortality": "3010303",
    "parkinsons": "3008271",
    "mnist": "2996257",
    "fashionmnist": "2992197",
    "cifar100": "2990873",
    "adult": "2992975",
    "bank": "2992984",
    "blog": "2993284",
    "in-hospital-mortality": "3011569",
    "svhn": "3044539",
    "stl": "3045959",
}

# Formula 1/class_freq
weights = {
    "default": "none",
    "adult": "1.32 4.17",
    "bank": "1.13 8.93",
    "physio-mortality": "1.17 6.93",
    "in-hospital-mortality": "1.16 7.14",
}

hyperparameters = [
    [
        ("data",),
        [
            # "in-hospital-mortality",
            "svhn",
            "physio-mortality",
            # "gas",
            # "mnist16",
            "fashionmnist",
            "grid",
            "parkinsons",
            # "wine",
            "mnist",
            "cifar10",
            # "adult",
            # "blog",
            # "bank",
        ],
    ],
    [("problem",), ["difa", "random", "gsmrl", "jafa", "vaeac", "eddi"]],
    [("seed",), [998]],
    [("n_features",), [idx for idx in range(-1, 201)]],
]
other_dependencies = {
    "memory": get_memory,
    "n_cpu": get_cpu,
    "valid": is_valid,
    "fixed_params": get_fixed_params,
    "gpu": get_gpu,
    "time": get_time,
}

# %%
run_id = int(get_run_id())

key_hyperparameters = [x[0] for x in hyperparameters]
value_hyperparameters = [x[1] for x in hyperparameters]
combinations: list[tuple[str, str, int, int]] = list(
    itertools.product(*value_hyperparameters)
)

scripts = []
gpu_counts = collections.defaultdict(int)

# %%
for combo in combinations:
    _data, _prob, _rseed, _nfeats = combo
    # Write the scheduler scripts
    template_name = "references/template.sh"
    with open(template_name, "r") as f:
        schedule_script = f.read()

    combo = {k[0]: v for (k, v) in zip(key_hyperparameters, combo)}

    for k, v in other_dependencies.items():
        combo[k] = v(combo)
    # set other params
    set_problem_specific_params(combo)
    if not combo["valid"]:
        # print(combo)
        continue
    # print(combo)
    combo["run_id"] = run_id
    gpu_counts[combo.get("gpu", "gpu-long")] += 1

    for k, v in combo.items():
        if "{%s}" % k in schedule_script:
            schedule_script = schedule_script.replace("{%s}" % k, str(v))

    schedule_script += "\n"

    # Write schedule script
    safe_makedirs(f"configs/{_data}")
    script_name = f"configs/{_data}/difa_{run_id}.sh"
    with open(script_name, "w") as f:
        f.write(schedule_script)
    scripts.append(script_name)

    # # Making files executable
    # subprocess.check_output("chmod +x %s" % script_name, shell=True)

    # # Update experiment logs
    # output = (
    #     "Script Name = "
    #     + script_name
    #     + ", Time Now= "
    #     + datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    #     + "\n"
    # )
    # with open("logs/expts.txt", "a") as f:
    #     f.write(output)
    # For the next job
    run_id += 1


# %%
# print(gpu_counts)
# # schedule jobs
# excludes = "--exclude="
# for script in scripts:  # --exclude=node078
#     command = "sbatch {} {}".format(excludes, script)
#     # print(command)
#     print(subprocess.check_output(command, shell=True))


# %%
