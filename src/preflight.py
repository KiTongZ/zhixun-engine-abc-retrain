"""Exercise the exported image against the offline SDK host stand-in."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import queue
import subprocess
import threading
import time
import zipfile

from host_fixture import Host

RUN = ["docker", "run", "--rm", "-i", "--platform", "linux/amd64", "--cpus", "4",
       "--memory", "4g", "--memory-swap", "4g", "--network", "none"]
PACKAGES = {"A": "cpu-a-guide", "B": "cpu-b-rul", "C": "cpu-c-guide"}


def mount(path, target, read_only=True):
    flags = ",readonly" if read_only else ""
    return ["--mount", f"type=bind,source={path.resolve()},target={target}{flags}"]


def sha(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def round_test(image, key, package, output, parent, parent_id, index, history):
    task = "classification" if key == "A" else "regression"
    digest = sha(parent / "checkpoint.npz")
    checkpoint = "checkpoint.npz"
    start = {
        "protocol": "contest.harness.workspace.v2", "run_id": "offline-preflight", "stage_id": "public",
        "dataset_key": key, "round_index": index, "rounds_per_dataset": 10,
        "prompt": "Local public data protocol test.", "prompt_version": "offline-preflight",
        "model_name": "base:local-host-stub", "max_output_tokens": 128, "parent_checkpoint_sha256": digest,
        "task_contract": {"dataset_key": key, "task": task, "metric": "macro_f1" if key == "A" else "rmse",
                          "metric_direction": "higher" if key == "A" else "lower",
                          "public_projection": {"train_path": "/data/train.jsonl", "validation_path": "/data/validation.jsonl"}},
        "parent_artifact": {"artifact_id": parent_id, "artifact_sha256": digest,
                            "artifact": {"checkpoint_paths": [checkpoint], "code_paths": []},
                            "files": [{"path": checkpoint, "sha256": digest, "size_bytes": (parent/checkpoint).stat().st_size}]},
        "artifacts": [{"artifact_id": parent_id}], "state_sha256": "", "state_path": None,
        "resource_limits": {"artifact_upload_protocol": "chunked-v1", "artifact_bytes": 64*1024**2,
                            "state_bytes": 64*1024**2, "candidate_upload_bytes_per_round": 64*1024**2,
                            "state_upload_bytes_per_round": 64*1024**2},
        "current_validation": {}, "wall_seconds": 300,
    }
    call_id = f"offline-stub-{key}-{index}"

    def response(messages, options):
        assert messages[0]["role"] == "system" and options["max_tokens"] <= 32
        return {"request_id": call_id, "choices": [{"message": {"content": '{"epochs":1}'}}]}

    work = output / f"round-{index}"
    work.mkdir(parents=True)
    candidate_id = f"offline-candidate-{key}-{index}"
    host = Host(start, response, history, candidate_id=candidate_id)
    command = (RUN + mount(package / "public", "/data") + mount(parent, "/previous")
               + mount(work, "/workspace", False) + [image, "train"])
    before = time.perf_counter()
    with (work / "stderr.log").open("w", encoding="utf-8") as stderr:
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                              text=True, encoding="utf-8") as proc:
            thread = host.serve(proc)
            try:
                proc.wait(timeout=300)
            except subprocess.TimeoutExpired:
                proc.kill()
                raise
            finally:
                thread.join(timeout=10)
    if host.failure or proc.returncode:
        raise RuntimeError(f"{key} round {index}: {host.failure}; see {work / 'stderr.log'}")
    record = host.record
    (work / "finish.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    assert len(host.raw_messages) == 1 and record["model_call_ids"] == [call_id]
    assert record["history_ids"] == [row["id"] for row in history]
    details = record["configuration"]
    assert details["origin"] == "platform_parent" and details["parent_checkpoint_sha256"] == digest
    assert details["selected_epoch"] == details["starting_epoch"] + 1
    assert details["selected_steps"] > details["starting_steps"]
    metric = "macro_f1" if key == "A" else "rmse"
    before_score, after_score = (details[name][metric] for name in ("validation_before", "validation_after"))
    improves = after_score > before_score if key == "A" else after_score < before_score
    assert details["selection"] == ("candidate" if improves else "parent")
    assert record["selected_artifact_id"] == (candidate_id if improves else parent_id)
    candidate = work / f"candidate-r{index}"
    with zipfile.ZipFile(parent / checkpoint) as original, zipfile.ZipFile(candidate / checkpoint) as trained:
        assert original.read("mean.npy") == trained.read("mean.npy")
        assert original.read("scale.npy") == trained.read("scale.npy")
        assert original.read("w1.npy") != trained.read("w1.npy")
        assert original.read("adam_first_w1.npy") != trained.read("adam_first_w1.npy")
    assert sum(path.stat().st_size for path in candidate.rglob("*") if path.is_file()) < 64*1024**2
    return candidate, {"round": index, "seconds": time.perf_counter() - before,
                       "gas_pair_head": details["gas_pair_head"], "candidate_metrics": details["validation_after"],
                       "selected": details["selection"], "selected_artifact_id": record["selected_artifact_id"],
                       "metrics": details["selected_validation"]}


def predictions(command, requests):
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, encoding="utf-8") as proc:
        lines = queue.Queue()

        def collect():
            for line in proc.stdout:
                lines.put(line)
            lines.put(None)

        worker = threading.Thread(target=collect, daemon=True)
        worker.start()
        results = []
        try:
            for request in requests:
                proc.stdin.write(json.dumps(request) + "\n")
                proc.stdin.flush()
                line = lines.get(timeout=50)
                if line is None:
                    raise RuntimeError(proc.stderr.read())
                results.append(json.loads(line))
            proc.stdin.close()
            assert proc.wait(50) == 0, proc.stderr.read()
        finally:
            if proc.poll() is None:
                proc.kill()
        return results


def verify_outputs(key, outputs):
    for row in outputs:
        pred = row["outputs"]
        if key == "A":
            assert len({item["label"] for item in pred}) == len(pred) == 6
            assert all(0 <= item["score"] <= 1 for item in pred)
            assert pred[0]["score"] > pred[1]["score"]
        else:
            assert math.isfinite(pred["value"]) and pred["value"] >= 0


def predict_test(image, key, package, artifact, snapshot):
    task = "classification" if key == "A" else "regression"
    request = []
    labels = set()
    with (package / "public/validation.jsonl").open(encoding="utf-8") as data:
        for line in data:
            row = json.loads(line)
            if key == "A" and row["answer"] in labels:
                continue
            request.append({"inputs": row["inputs"], "task": task})
            labels.add(row["answer"])
            if len(request) == (6 if key == "A" else 8):
                break
    common = RUN + ["--read-only"] + mount(artifact, "/model")
    standalone = common + ["--entrypoint", "python3", image, "-B", "/model/predict.py"]
    observed = predictions(standalone, [*request, {"samples": request}, list(reversed(request))])
    for sequence in (observed[-2], observed[-1]):
        verify_outputs(key, sequence)
    direct = predictions(common + [image, "predict"], request)
    baked = RUN + ["--read-only", "--entrypoint", "python3", image, "-B",
                   f"/opt/retrain/models/{key}/predict.py"]
    baked_outputs = predictions(baked, [{"samples": request}])[0]
    verify_outputs(key, baked_outputs)
    local_baked = predictions(RUN + ["--read-only"] + mount(snapshot, "/model") +
                              ["--entrypoint", "python3", image, "-B", "/model/predict.py"],
                              [{"samples": request}])[0]
    assert baked_outputs == local_baked
    singles, batch, reversed_batch = observed[:-2], observed[-2], observed[-1]
    for single, cli, bundled, reverse in zip(singles, direct, batch, reversed(reversed_batch)):
        if key == "A":
            assert [row["label"] for row in single["outputs"]] == [row["label"] for row in bundled["outputs"]]
            assert single == cli
            for values in (bundled, reverse):
                assert all(math.isclose(left["score"], right["score"], rel_tol=1e-10, abs_tol=1e-12)
                           for left, right in zip(single["outputs"], values["outputs"]))
        else:
            assert all(math.isclose(single["outputs"]["value"], row["outputs"]["value"], rel_tol=1e-10, abs_tol=1e-12)
                       for row in (cli, bundled, reverse))
    return len(request)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("packages", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    described = json.loads(subprocess.check_output(RUN + [args.image, "describe"], text=True))
    assert described["sdk_version"] == "1.0.4"
    assert described["tasks"] == ["classification", "regression"]
    limits_code = ('import ctypes; from pathlib import Path; import numpy as np; '
                   'print(Path("/sys/fs/cgroup/cpu.max").read_text().strip()); '
                   'print(Path("/sys/fs/cgroup/memory.max").read_text().strip()); '
                   'print(Path("/sys/fs/cgroup/memory.swap.max").read_text().strip()); '
                   'library = next((Path(np.__file__).resolve().parent.parent / "numpy.libs").glob("*openblas*.so")); '
                   'lib = ctypes.CDLL(str(library)); '
                   'fn = next(getattr(lib, name) for name in ("scipy_openblas_get_num_threads64_", '
                   '"scipy_openblas_get_num_threads", "openblas_get_num_threads64_", "openblas_get_num_threads") '
                   'if hasattr(lib, name)); print(fn()); print(library.name)')
    limits = subprocess.check_output(RUN + ["--entrypoint", "python3", args.image, "-c", limits_code], text=True)
    quota, period = map(int, limits.splitlines()[0].split())
    assert quota / period == 4 and limits.splitlines()[1:4] == ["4294967296", "0", "1"]
    (args.output / "resources.txt").write_text(limits, encoding="utf-8")
    results = {}
    for key, name in PACKAGES.items():
        package = args.packages / (name + "-20261110-v1")
        parent_id = json.loads((package / "baseline-info.json").read_text(encoding="utf-8"))["id"]
        data_output = args.output / key
        first, first_report = round_test(args.image, key, package, data_output, package, parent_id, 1, [])
        history = [{"id": f"offline-record-{key}-1"}]
        next_parent = first if first_report["selected"] == "candidate" else package
        next_id = first_report["selected_artifact_id"]
        second, second_report = round_test(args.image, key, package, data_output, next_parent, next_id, 2, history)
        count = predict_test(args.image, key, package, second, args.results / key)
        results[key] = {"rounds": [first_report, second_report], "prediction_samples": count}
        print(f"{key}: official parent round, derived parent round and read-only prediction passed.", flush=True)
    (args.output / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
