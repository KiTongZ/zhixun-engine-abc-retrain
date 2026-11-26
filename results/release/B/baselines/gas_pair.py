"""Small RBF classifier for the acetaldehyde/ethanol confusion pair."""
import numpy as np


def features(x, scale, mode):
    values = np.arcsinh(x / scale)
    if mode.startswith("log_"):
        sensors = np.log(np.abs(x / scale).clip(1e-8)).reshape(len(x), 16, 8)
        sensors -= sensors.mean(axis=1, keepdims=True)
        if mode == "log_steady":
            sensors = sensors[:, :, :2]
        elif mode == "log_rise":
            sensors = sensors[:, :, 2:5]
        elif mode == "log_decay":
            sensors = sensors[:, :, 5:8]
        elif mode == "log_rate":
            sensors = np.diff(sensors[:, :, 2:8], axis=2)
        return sensors.reshape(len(x), -1)
    if mode in ("steady", "rise", "decay", "across_sensors", "within_sensor", "relative_decay"):
        sensors = values.reshape(len(values), 16, 8)
        if mode == "across_sensors":
            return (sensors - sensors.mean(axis=1, keepdims=True)).reshape(len(values), 128)
        if mode == "within_sensor":
            return (sensors - sensors.mean(axis=2, keepdims=True)).reshape(len(values), 128)
        if mode == "relative_decay":
            return (sensors[:, :, 2:5] + sensors[:, :, 5:8]).reshape(len(values), 48)
        channels = {"steady": slice(0, 2), "rise": slice(2, 5), "decay": slice(5, 8)}[mode]
        subset = sensors[:, :, channels]
        return (subset - subset.mean(axis=1, keepdims=True)).reshape(len(values), -1)
    if mode == "direction":
        return values / np.linalg.norm(values, axis=1, keepdims=True).clip(1e-8)
    if mode in ("center", "shape"):
        values -= values.mean(axis=1, keepdims=True)
        if mode == "shape":
            values /= values.std(axis=1, keepdims=True).clip(1e-8)
    return values


def distances(a, b):
    return np.maximum(0, (a*a).sum(1)[:, None] + (b*b).sum(1)[None, :] - 2 * (a @ b.T))


def apply_margin(probabilities, margin, ae, et, threshold):
    labels = probabilities.argmax(1)
    swap = ((labels == ae) & (margin <= threshold)) | ((labels == et) & (margin > threshold))
    corrected = probabilities.copy()
    corrected[swap, ae] = probabilities[swap, et]
    corrected[swap, et] = probabilities[swap, ae]
    tied = swap & (probabilities[:, ae] == probabilities[:, et])
    corrected[tied, ae] = np.nextafter(corrected[tied, ae], np.where(margin[tied] > threshold, 1., 0.))
    return corrected


class GasPair:
    def __init__(self, spec):
        self.spec = spec
        self.support = np.asarray(spec["support"])
        self.support_norm = np.einsum("ij,ij->i", self.support, self.support)
        self.scale = np.asarray(spec["scale"])
        self.coef = np.asarray(spec["coef"])

    def predict(self, x, probabilities):
        ae, et = self.spec["class_indices"]
        rows = np.flatnonzero(np.isin(probabilities.argmax(1), (ae, et)))
        for start in range(0, len(rows), 1024):
            batch = rows[start:start+1024]
            values = features(x[batch], self.scale, self.spec["mode"])
            norm = np.einsum("ij,ij->i", values, values)
            distance = np.maximum(0, norm[:, None] + self.support_norm - 2 * (values @ self.support.T))
            margin = np.exp(-distance / self.spec["width"]) @ self.coef
            probabilities[batch] = apply_margin(probabilities[batch], margin, ae, et, self.spec["threshold"])
        return probabilities
