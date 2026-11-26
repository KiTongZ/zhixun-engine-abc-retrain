"""Train three CPU models from random initialization on the official public splits."""
import argparse
import copy
import json
from pathlib import Path
import re
import time

from baselines.model import Model, checkpoint_digest
from baselines.data import read_dataset
from baselines.gas_pair import apply_margin, distances, features
from bearing_sampling import bearing_groups
import numpy as np

PACKAGES = {"A": "cpu-a-guide", "B": "cpu-b-rul", "C": "cpu-c-guide"}
CLASSES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
SEED = 42
# transform, layer width, learning rate, batch size, sampling power, epochs
SEARCH = {
    "A": [
        ("raw", 32, .003, 128, 0, 20),
        ("raw", 64, .003, 64, .5, 100),
        ("asinh", 64, .003, 64, 0, 100),
        ("asinh", 64, .003, 64, .75, 100),
        ("shape", 64, .003, 64, 0, 100),
        ("shape", 64, .003, 64, .75, 100),
    ],
    "B": [
        ("raw", 24, .003, 128, 0, 15),
        ("raw", 48, .003, 256, .35, 36),
        ("asinh", 48, .003, 256, 0, 45),
        ("asinh", 48, .003, 256, .35, 45),
    ],
    "C": [
        ("raw", 32, .002, 256, 0, 15),
        ("raw", 64, .002, 512, 0, 30),
        ("engine", 64, .002, 512, 0, 30),
    ],
}


def contract(key):
    return ("classification", CLASSES) if key == "A" else ("regression", [])


def metrics(key, target, predictions):
    if key != "A":
        error = predictions - target
        return {"rmse": float(np.sqrt(np.mean(error ** 2))), "mae": float(np.mean(np.abs(error)))}
    n = len(CLASSES)
    labels = predictions.argmax(1)
    matrix = np.bincount(target * n + labels, minlength=n*n).reshape(n, n)
    diagonal = matrix.diagonal()
    f1 = 2 * diagonal / (matrix.sum(0) + matrix.sum(1)).clip(1)
    acetaldehyde = CLASSES.index("Acetaldehyde")
    ethanol = CLASSES.index("Ethanol")
    return {
        "macro_f1": float(f1.mean()), "accuracy": float((labels == target).mean()),
        "acetaldehyde_f1": float(f1[acetaldehyde]),
        "acetaldehyde_precision": float(diagonal[acetaldehyde] / max(1, matrix[:, acetaldehyde].sum())),
        "acetaldehyde_recall": float(diagonal[acetaldehyde] / max(1, matrix[acetaldehyde].sum())),
        "acetaldehyde_to_ethanol": int(matrix[acetaldehyde, ethanol]),
        "ethanol_to_acetaldehyde": int(matrix[ethanol, acetaldehyde]),
        "ethanol_f1": float(f1[ethanol]),
        "confusion_matrix": matrix.tolist(),
    }


def selection_rows(key, data):
    if key == "A":
        selected = np.zeros(len(data["y"]), dtype=bool)
        for label in range(len(CLASSES)):
            rows = np.flatnonzero(data["y"] == label)
            selected[rows[:len(rows)//2]] = True
        return selected
    if key == "B":
        groups = bearing_groups(data["ids"])
    else:
        groups = np.array([re.sub(r"[_-](?:c|t)?\d+$", "", value) for value in data["ids"]])
    if groups is None or len(np.unique(groups)) < 2:
        raise ValueError(f"{key}: cannot create a device-disjoint validation split")
    return np.isin(groups, np.unique(groups)[::2])


def sampling_probabilities(key, data, power):
    if power == 0:
        return None
    groups = data["y"] if key == "A" else bearing_groups(data["ids"])
    if groups is None:
        return None
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = counts[inverse].astype(float) ** (-power)
    return weights / weights.sum()


def fit_epoch(model, x, y, probabilities):
    if probabilities is not None:
        rng = np.random.default_rng([model.meta["seed"], model.meta["epochs"], 719])
        rows = rng.choice(len(x), len(x), replace=True, p=probabilities)
        x, y = x[rows], y[rows]
    return float(model.fit(x, y, 1, normalized=True)[0])


def init_model(key, train, profile):
    transform, hidden, rate, batch, _, _ = profile
    task, classes = contract(key)
    return Model(task, train["paths"], classes, train["x"], train["y"], hidden=hidden, seed=SEED,
                 learning_rate=rate, batch_size=batch, transform=transform)


def rank(key, score):
    if key == "A":
        return score["macro_f1"] + .25 * score["acetaldehyde_f1"]
    return -score["rmse"]


def evaluate(key, validation, model, rows=slice(None)):
    return metrics(key, validation["y"][rows], model.predict(validation["x"][rows]))


def select_gas_pair(model, train, validation, select):
    ae, et = (CLASSES.index(name) for name in ("Acetaldehyde", "Ethanol"))
    x, y = gas_support(train, ae, et)
    if not ((y == ae).any() and (y == et).any()):
        return None
    scale = np.maximum(np.median(np.abs(x), axis=0), 1e-8)
    expected, raw = validation["y"], validation["x"]
    base = model.predict(raw)
    blocks = (select, ~select)
    baselines = [metrics("A", expected[block], base[block]) for block in blocks]
    best, selected = 0., None
    for mode in ("compressed", "direction", "center", "shape", "steady", "rise", "decay",
                 "across_sensors", "within_sensor", "relative_decay", "log_all",
                 "log_steady", "log_rise", "log_decay", "log_rate"):
        support = features(x, scale, mode)
        train_dist = distances(support, support)
        valid_dist = distances(features(raw, scale, mode), support)
        positive = train_dist[train_dist > 1e-9]
        if not positive.size:
            continue
        class_weight = np.where(y == ae, (y == ae).mean(), (y == et).mean())
        for factor in (.1, .4, 1., 3.):
            width = float(factor * np.median(positive))
            kernel = np.exp(-train_dist / width)
            valid_kernel = np.exp(-valid_dist / width)
            for penalty in (.01, .1):
                coef = np.linalg.solve(kernel + np.diag(penalty * class_weight), np.where(y == ae, 1., -1.))
                margin = valid_kernel @ coef
                for threshold in (-.65, -.5, -.375, -.25, -.125, 0., .125, .25, .375, .5):
                    corrected = apply_margin(base, margin, ae, et, threshold)
                    scores = [metrics("A", expected[block], corrected[block]) for block in blocks]
                    value = min(rank("A", score) - rank("A", baseline) for score, baseline in zip(scores, baselines))
                    if value > best and all(gas_pair_allowed(score, baseline) for score, baseline in zip(scores, baselines)):
                        best = value
                        selected = {"class_indices": [ae, et], "mode": mode, "scale": scale.tolist(),
                                    "support": support.tolist(), "width": width, "coef": coef.tolist(),
                                    "threshold": threshold, "penalty": penalty, "width_factor": factor}
    return selected


def gas_pair_allowed(score, baseline):
    return (score["macro_f1"] >= baseline["macro_f1"] and score["acetaldehyde_precision"] >= .7
            and score["ethanol_f1"] >= baseline["ethanol_f1"] - .12)


def gas_support(train, ae, et):
    # A larger stage projection must not create an unbounded dense kernel.
    rows = []
    for label in (ae, et):
        group = np.flatnonzero(train["y"] == label)
        if len(group) > 256:
            group = group[np.linspace(0, len(group) - 1, 256, dtype=int)]
        rows.extend(group)
    rows = np.sort(rows)
    return train["x"][rows], train["y"][rows]


def refit_gas_pair(train, profile):
    ae, et = (CLASSES.index(name) for name in ("Acetaldehyde", "Ethanol"))
    x, y = gas_support(train, ae, et)
    if not ((y == ae).any() and (y == et).any()):
        return None
    scale = np.maximum(np.median(np.abs(x), axis=0), 1e-8)
    support = features(x, scale, profile["mode"])
    train_dist = distances(support, support)
    positive = train_dist[train_dist > 1e-9]
    if not positive.size:
        return None
    width = float(profile["width_factor"] * np.median(positive))
    weight = np.where(y == ae, (y == ae).mean(), (y == et).mean())
    kernel = np.exp(-train_dist / width) + np.diag(profile["penalty"] * weight)
    coef = np.linalg.solve(kernel, np.where(y == ae, 1., -1.))
    return {**profile, "class_indices": [ae, et], "scale": scale.tolist(), "support": support.tolist(),
            "width": width, "coef": coef.tolist()}


PREDICTION_FILES = ("predict.py", "config.json", "baselines/__init__.py", "baselines/data.py",
                    "baselines/model.py", "baselines/gas_pair.py")


def write_artifact(directory, model, key):
    import shutil

    source = Path(__file__).resolve().parent
    directory.mkdir(parents=True)
    (directory / "baselines").mkdir()
    model.save(directory / "checkpoint.npz")
    task, classes = contract(key)
    (directory / "config.json").write_text(json.dumps({"task": task, "classes": classes, "profile": key}), encoding="utf-8")
    shutil.copyfile(source / "artifact_predict.py", directory / "predict.py")
    for name in PREDICTION_FILES[2:]:
        shutil.copyfile(source / name, directory / name)
    if sum(path.stat().st_size for path in directory.rglob("*") if path.is_file()) > 64 * 1024**2:
        raise ValueError("model artifact exceeds 64 MiB")


def experiment(key, packages, output):
    started = time.perf_counter()
    cpu = time.process_time()
    package = packages / (PACKAGES[key] + "-20261110-v1")
    task, classes = contract(key)
    info = json.loads((package / "training.json").read_text(encoding="utf-8"))
    train = read_dataset(package / "public/train.jsonl", task, classes)
    validation = read_dataset(package / "public/validation.jsonl", task, classes, train["paths"])
    for name, data in (("train", train), ("validation", validation)):
        if data["sha256"] != info[name+"_sha256"] or len(data["y"]) != info[name+"_rows"]:
            raise ValueError(f"{key}: {name} does not match official metadata")
    select = selection_rows(key, validation)
    selection_x, selection_y = validation["x"][select], validation["y"][select]
    print(f"{key}: official splits verified, {select.sum()} selection rows / {(~select).sum()} report rows", flush=True)
    best = None
    trials = []
    for profile in SEARCH[key]:
        model = init_model(key, train, profile)
        baseline = evaluate(key, validation, model, select)
        normalized_x = model.normalized(train["x"])
        normalized_selection = model.normalized(selection_x)
        probabilities = sampling_probabilities(key, train, profile[4])
        trial = {"profile": profile, "initial_selection": baseline, "epochs": []}
        trials.append(trial)
        trial_best = float("-inf")
        for _ in range(profile[5]):
            loss = fit_epoch(model, normalized_x, train["y"], probabilities)
            score = metrics(key, selection_y, model.predict(normalized_selection, normalized=True))
            epoch = model.meta["epochs"]
            trial["epochs"].append({"epoch": epoch, "loss": loss, "selection": score})
            value = rank(key, score)
            trial_best = max(value, trial_best)
            if best is None or value > best[0]:
                best = (value, copy.deepcopy(model), (*profile[:5], epoch))
        print(f"{key}: trial {profile} best selection score {trial_best:.5f}", flush=True)
    _, model, chosen = best
    uncorrected = evaluate(key, validation, model)
    pair_report = None
    if key == "A":
        pair = select_gas_pair(model, train, validation, select)
        if pair is not None:
            model.meta["gas_pair"] = pair
            pair_report = {name: pair[name] for name in ("mode", "width", "width_factor", "threshold", "penalty")}
            pair_report["support_rows"] = len(pair["support"])
    artifact = output / key
    write_artifact(artifact, model, key)
    report = {
        "dataset_key": key, "seed": SEED, "initialization": "random", "parent_checkpoint": None,
        "train_sha256": train["sha256"], "validation_sha256": validation["sha256"],
        "train_rows": len(train["y"]), "validation_rows": len(validation["y"]),
        "selection_rows": int(select.sum()), "report_rows": int((~select).sum()),
        "profile": chosen, "checkpoint_sha256": checkpoint_digest(artifact / "checkpoint.npz"),
        "optimizer_steps": model.meta["steps"],
        "uncorrected_public_validation": uncorrected, "gas_pair": pair_report,
        "report_partition": evaluate(key, validation, model, ~select),
        "report_partition_used_in_gas_head_selection": pair_report is not None,
        "entire_public_validation": evaluate(key, validation, model),
        "wall_s": time.perf_counter() - started, "cpu_s": time.process_time() - cpu,
        "trials": trials,
    }
    (output / f"{key}.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"{key}: selected {chosen}; public validation {report['entire_public_validation']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packages", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("datasets", nargs="*", choices=tuple(PACKAGES), default=list(PACKAGES))
    args = parser.parse_args()
    for dataset in args.datasets:
        experiment(dataset, args.packages, args.output)
