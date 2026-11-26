"""SDK workspace v2 entry for the A, B and C industrial models."""
import copy
import json
import os
from pathlib import Path
import sys
import time

from baselines.model import Model
from baselines.data import read_dataset
from artifact_predict import load_artifact, predict_batch
from train_models import (
    PREDICTION_FILES, contract, evaluate, fit_epoch, gas_pair_allowed, rank,
    refit_gas_pair, sampling_probabilities, write_artifact,
)

TASKS = ("classification", "regression")
WORK = Path(os.environ.get("CONTEST_WORK", "/workspace"))
PROFILE_FILE = Path(__file__).with_name("profiles.json")


def train(rnd):
    start = time.perf_counter()
    key = rnd.dataset_key
    task, classes = contract(key)
    if rnd.task.get("task") != task:
        raise ValueError(f"{key}: incompatible task contract")
    train_data = read_dataset(rnd.data.train_path, task, classes)
    validation = read_dataset(rnd.data.validation_path, task, classes, train_data["paths"])
    config = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))[key]
    profile = config["profile"]

    checkpoints = rnd.parent.descriptor["checkpoint_paths"]
    parent_digest = rnd.parent.checkpoint_sha256
    if "checkpoint.npz" not in checkpoints or not parent_digest:
        raise ValueError("a continuation round requires the declared parent checkpoint.npz")
    path = rnd.parent.path("checkpoint.npz", root=os.environ.get("CONTEST_PARENT_ROOT") or None)
    model = Model.load(path)
    if (model.meta["task"], model.meta["paths"], model.meta["classes"]) != (task, train_data["paths"], classes):
        raise ValueError("parent checkpoint does not match the task's input schema")
    origin = "platform_parent"
    default_epochs, allowed_epochs = 1, (1, 2)

    before = evaluate(key, validation, model)
    starting_epoch = model.meta["epochs"]
    starting_steps = model.meta["steps"]
    request = {
        "dataset": key, "origin": origin, "validation": {k: v for k, v in before.items() if k != "confusion_matrix"},
        "train_rows": len(train_data["y"]), "allowed_epochs": allowed_epochs, "recommended_epochs": default_epochs,
    }
    receipt = rnd.model(
        [{"role": "user", "content": "Choose a CPU training budget using the diagnostics. "
          'Keep the parent if the main public metric worsens. '
          'Prefer recommended_epochs unless the validation suggests a change. Reply with {"epochs": integer}. '
          + json.dumps(request, separators=(",", ":"))}],
        max_tokens=min(32, rnd.max_output_tokens), temperature=0, enable_thinking=False,
    )
    try:
        response = json.loads(receipt.text.strip())
    except ValueError:
        response = {}
    epochs = response.get("epochs") if isinstance(response, dict) else None
    if type(epochs) is not int or epochs not in allowed_epochs:
        epochs = default_epochs

    probabilities = sampling_probabilities(key, train_data, profile[4])
    x = model.normalized(train_data["x"])
    history = []
    best = None
    for _ in range(epochs):
        loss = fit_epoch(model, x, train_data["y"], probabilities)
        score = evaluate(key, validation, model)
        history.append({"epoch": model.meta["epochs"], "loss": loss, "metrics": score})
        value = rank(key, score)
        if best is None or value > best[0]:
            best = (value, copy.deepcopy(model), score)
    _, model, after = best

    gas_head = "gas_pair" in model.meta
    if key == "A" and not gas_head and config.get("gas_pair"):
        pair = refit_gas_pair(train_data, config["gas_pair"])
        if pair is not None:
            model.meta["gas_pair"] = pair
            corrected = evaluate(key, validation, model)
            if gas_pair_allowed(corrected, after) and rank(key, corrected) > rank(key, after):
                after, gas_head = corrected, True
            else:
                del model.meta["gas_pair"]

    candidate = WORK / f"candidate-r{rnd.round_index}"
    write_artifact(candidate, model, key)
    details = {
        "dataset_key": key, "origin": origin, "parent_checkpoint_sha256": parent_digest or None,
        "starting_epoch": starting_epoch, "starting_steps": starting_steps, "selected_epoch": model.meta["epochs"],
        "selected_steps": model.meta["steps"], "requested_epochs": epochs, "gas_pair_head": gas_head,
        "validation_before": before, "validation_after": after, "epochs": history,
        "train_sha256": train_data["sha256"], "validation_sha256": validation["sha256"],
        "model_call_id": receipt.call_id, "local_wall_s": time.perf_counter() - start,
    }
    hypothesis = "Public feature training can improve the current industrial model under a single BLAS thread."
    change = f"{origin}: trained {model.meta['epochs'] - starting_epoch} epoch(s), gas_pair_head={gas_head}."
    submitted = rnd.submit_candidate(
        candidate, hypothesis=hypothesis, change_summary=change, configuration=details, seed=model.meta["seed"],
        checkpoint_paths=["checkpoint.npz"], code_paths=list(PREDICTION_FILES),
        predict_argv=["python3", "-B", "/model/predict.py"],
    )
    metric = "macro_f1" if key == "A" else "rmse"
    improves = after[metric] > before[metric] if key == "A" else after[metric] < before[metric]
    chosen = submitted["artifact_id"] if improves else rnd.parent.artifact_id
    details["selection"] = "candidate" if improves else "parent"
    details["selected_validation"] = after if improves else before
    details["selected_artifact_id"] = chosen
    state = WORK / f"state-r{rnd.round_index}"
    state.mkdir()
    (state / "round.json").write_text(json.dumps(details, indent=2, allow_nan=False), encoding="utf-8")
    rnd.save_state(state)
    past = rnd.history()
    print(json.dumps({"dataset": key, "round": rnd.round_index, "origin": origin,
                      "train_rows": len(train_data["y"]), "validation_rows": len(validation["y"]),
                      "wall_s": round(time.perf_counter() - start, 3), "gas_pair_head": gas_head,
                      "selection": details["selection"]}),
          file=sys.stderr, flush=True)
    rnd.finish(
        selected_artifact_id=chosen, hypothesis=hypothesis, change_summary=change,
        reason=f"Model gateway budget was {epochs} epoch(s). Selected {details['selection']} using public {metric}.",
        configuration={**details, "platform_validation": submitted.get("summary")},
        seed=model.meta["seed"], history_ids=[record["id"] for record in past], model_call_ids=list(rnd.call_ids),
    )


_CACHE = {}


def predict(sample, model_dir):
    path = Path(os.environ.get("CONTEST_ARTIFACT_PATH") or (Path(model_dir) / "checkpoint.npz"))
    info = path.stat()
    key = (path, info.st_size, info.st_mtime_ns)
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = load_artifact(path)
    # The SDK sample is an object with inputs/task attributes.
    result = predict_batch(_CACHE[key], [{"inputs": sample.inputs, "task": sample.task}])
    return result[0]["outputs"]
