"""DisNet: distance estimation from a monocular camera.

Implements the method of Haseeb, Guan, Ristic-Durrant and Graeser, "DisNet: A novel method for
distance estimation from monocular camera" (2018). A YOLO bounding box is turned into a
six-dimensional feature vector

    v = [1/Bw, 1/Bh, 1/Bd, Ch, Cw, Cb]

with
    Bw, Bh  box width / height divided by the image width / height,
    Bd      box diagonal, computed from the normalised width and height: sqrt(Bw^2 + Bh^2),
    Ch, Cw, Cb  average real-world height / width / breadth of the detected class (cm).

and a small fully connected network maps v to the distance (metres) between camera and object.
There is NO input standardisation: the features are fed to the network as they are.

Notes on the feature order
--------------------------
Equation (1) of the paper lists the features as [1/Bh, 1/Bw, 1/Bd, ...], but the authors' released
code (github.com/guanjianyu/DisNet, both the training script and the inference script) uses
[1/Bw, 1/Bh, 1/Bd, ...] and the pretrained model was trained that way, so this module follows the code.
The same applies to Bd: the code takes the diagonal of the *normalised* box, not the box diagonal
divided by the image diagonal (the two are equal only for square images).

Inference is pure NumPy, so no TensorFlow / Keras is needed to run a trained model. Models are stored as
`.npz`; Keras HDF5 models (such as the authors' `model_3.keras`) can be imported with `DisNet.from_keras_h5`
(needs `h5py`) and re-saved as `.npz`:

    python -m utils.disnet convert model_3.keras generated_files/disnet_model_3.npz
"""
import json
import sys

import numpy as np

# Average (height, width, breadth) in cm of the object classes DisNet was trained for.
# Values from the paper / the authors' implementation (keys are COCO class names).
CLASS_SIZES_CM = {
    "person": (175, 55, 30),
    "bus": (300, 250, 1200),
    "truck": (400, 350, 1500),
    "car": (160, 180, 400),
    "bicycle": (110, 50, 180),
    "motorbike": (110, 50, 180),
    "cat": (40, 20, 50),
    "dog": (50, 30, 60),
    "horse": (180, 60, 200),
    "sheep": (130, 60, 150),
    "cow": (170, 70, 200),
}

# Names used by other label sets (e.g. KITTI) that map unambiguously onto the classes above.
CLASS_ALIASES = {
    "pedestrian": "person",
    "cyclist": "bicycle",
    "motorcycle": "motorbike",
}

_SELU_ALPHA = 1.6732632423543772848170429916717
_SELU_SCALE = 1.0507009873554804934193349852946

_ACTIVATIONS = {
    "linear": lambda x: x,
    "relu": lambda x: np.maximum(x, 0.0),
    "tanh": np.tanh,
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "selu": lambda x: _SELU_SCALE * np.where(x > 0, x, _SELU_ALPHA * np.expm1(np.minimum(x, 0.0))),
}


def canonical_class(name):
    """Map a class name to a key of CLASS_SIZES_CM, or None if DisNet has no size for it."""
    key = str(name).strip().lower()
    key = CLASS_ALIASES.get(key, key)
    return key if key in CLASS_SIZES_CM else None


def box_features(box_w_px, box_h_px, img_w, img_h, class_name):
    """Six DisNet input features for one box (sizes in pixels). Returns None for unsupported classes
    or degenerate boxes."""
    cls = canonical_class(class_name)
    if cls is None or box_w_px <= 0 or box_h_px <= 0:
        return None
    bw = float(box_w_px) / img_w
    bh = float(box_h_px) / img_h
    bd = float(np.sqrt(bw ** 2 + bh ** 2))
    ch, cw, cb = CLASS_SIZES_CM[cls]
    return np.array([1.0 / bw, 1.0 / bh, 1.0 / bd, ch, cw, cb], dtype=np.float32)


class DisNet:
    """Fully connected regression network: features (N, 6) -> distance in metres (N,)."""

    def __init__(self, layers):
        """`layers`: list of (kernel (in, out), bias (out,), activation name)."""
        self.layers = [(np.asarray(w, np.float32), np.asarray(b, np.float32), act) for w, b, act in layers]
        for _, _, act in self.layers:
            if act not in _ACTIVATIONS:
                raise ValueError("Unsupported activation %r (supported: %s)" % (act, sorted(_ACTIVATIONS)))
        assert self.layers[0][0].shape[0] == 6, "DisNet expects 6 input features"
        assert self.layers[-1][0].shape[1] == 1, "DisNet expects a single output (distance)"

    # ------------------------------------------------------------------ inference
    def predict(self, features):
        """features: (6,) or (N, 6) -> distances in metres, shape (N,)."""
        x = np.atleast_2d(np.asarray(features, dtype=np.float32))
        for w, b, act in self.layers:
            x = _ACTIVATIONS[act](x @ w + b)
        return x[:, 0]

    def estimate(self, box_xyxy, class_name, img_w, img_h):
        """Distance in metres for a box (x1, y1, x2, y2 in image pixels), or None if the class has no
        known size or the box is degenerate."""
        x1, y1, x2, y2 = box_xyxy
        feats = box_features(x2 - x1, y2 - y1, img_w, img_h, class_name)
        if feats is None:
            return None
        return float(self.predict(feats)[0])

    # ------------------------------------------------------------------ (de)serialisation
    def save_npz(self, path):
        arrays = {"activations": np.array([a for _, _, a in self.layers])}
        for i, (w, b, _) in enumerate(self.layers):
            arrays["W%d" % i] = w
            arrays["b%d" % i] = b
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        """Load a model saved with `save_npz`, or a Keras HDF5 model (extension .h5 / .keras)."""
        if str(path).endswith(".npz"):
            return cls.from_npz(path)
        return cls.from_keras_h5(path)

    @classmethod
    def from_npz(cls, path):
        data = np.load(path, allow_pickle=False)
        acts = [str(a) for a in data["activations"]]
        return cls([(data["W%d" % i], data["b%d" % i], a) for i, a in enumerate(acts)])

    @classmethod
    def from_keras_h5(cls, path):
        """Import a Sequential Dense model saved by Keras 2.x as HDF5 (no TensorFlow needed, requires h5py)."""
        import h5py

        with h5py.File(path, "r") as f:
            if "model_weights" not in f:
                raise ValueError("%s is not a Keras 2.x HDF5 model (Keras 3 '.keras' zip files are not supported; "
                                 "re-save it with model.save('x.h5') or export the weights to .npz)" % path)
            raw = f.attrs["model_config"]
            config = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            layer_cfgs = config["config"] if isinstance(config["config"], list) else config["config"]["layers"]
            activations = {l["config"]["name"]: l["config"].get("activation", "linear")
                           for l in layer_cfgs if l["class_name"] == "Dense"}

            weights = f["model_weights"]
            layers = []
            for name in weights.attrs["layer_names"]:
                name = name.decode() if isinstance(name, bytes) else name
                if name not in activations:  # InputLayer etc.
                    continue
                group = weights[name]
                kernel = bias = None
                def visit(key, obj):
                    nonlocal kernel, bias
                    if isinstance(obj, h5py.Dataset):
                        if key.endswith("kernel:0"):
                            kernel = obj[()]
                        elif key.endswith("bias:0"):
                            bias = obj[()]
                group.visititems(visit)
                layers.append((kernel, bias, activations[name]))
        return cls(layers)


def _convert(src, dst):
    model = DisNet.from_keras_h5(src)
    model.save_npz(dst)
    print("Saved %s (%d layers)" % (dst, len(model.layers)))


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "convert":
        _convert(sys.argv[2], sys.argv[3])
    else:
        sys.exit("usage: python -m utils.disnet convert <keras_model.h5|.keras> <out.npz>")
