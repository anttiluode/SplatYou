"""The unchanged original SplatWorld decoder, plus honest Sihti-style anchor residues."""
from __future__ import annotations
import hashlib
import os
from pathlib import Path
from urllib.request import urlopen

import numpy as np

LATENT = 128
# A pinned, user-authored upstream commit, not a changing main-branch download.
UPSTREAM_COMMIT = 'cdfa03a69d2412c0112db4da30e298353fd280df'
UPSTREAM_URL = f'https://raw.githubusercontent.com/anttiluode/SplatWorld/{UPSTREAM_COMMIT}/splat_decoder.onnx'
UPSTREAM_SIZE = 7_221_021


def find_model(explicit: str | None = None) -> Path | None:
    choices = [Path(explicit).expanduser()] if explicit else []
    env = os.environ.get('SPLAT_MODEL')
    if env:
        choices.append(Path(env).expanduser())
    here = Path(__file__).resolve().parent.parent
    for parent in (here, here.parent, Path.cwd()):
        choices += [parent / 'splat_decoder.onnx', parent / 'SplatWorld' / 'splat_decoder.onnx',
                    parent / 'SplatWorld2' / 'splat_decoder.onnx', parent / 'SplatWorld3' / 'splat_decoder.onnx']
    for p in choices:
        if p.is_file():
            return p.resolve()
    return None


def download_model(dest: str = 'splat_decoder.onnx') -> Path:
    path = Path(dest)
    if path.exists():
        raise FileExistsError(f'{path} exists; refusing to overwrite')
    print('Downloading pinned original SplatWorld ONNX model (7.2 MB) ...')
    raw = urlopen(UPSTREAM_URL, timeout=45).read()
    if len(raw) != UPSTREAM_SIZE:
        raise RuntimeError(f'Upstream model size changed: got {len(raw)} expected {UPSTREAM_SIZE}')
    # Report the SHA256 for recording, without pretending it was preverified.
    print('Model SHA256:', hashlib.sha256(raw).hexdigest())
    path.write_bytes(raw)
    return path.resolve()


class WaveDecoder:
    """Batch-compatible CPU ONNX decoder. Always genuine: no mock fallback."""
    def __init__(self, model_path: str | Path):
        import onnxruntime as ort
        self.path = Path(model_path)
        self.session = ort.InferenceSession(str(self.path), providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def __call__(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, np.float32)
        if z.ndim == 1:
            z = z[None]
        if z.ndim != 2 or z.shape[1] != LATENT:
            raise ValueError(f'Expected (batch,{LATENT}) latents, got {z.shape}')
        out = self.session.run([self.output_name], {self.input_name: np.ascontiguousarray(z)})[0]
        if out.ndim != 4 or out.shape[1] != 3:
            raise RuntimeError(f'Unexpected ONNX output {out.shape}')
        return np.clip(out.transpose(0, 2, 3, 1), 0, 1).astype(np.float32)


def new_latent(rng: np.random.Generator, radius: float = 8.) -> np.ndarray:
    z = rng.standard_normal(LATENT).astype(np.float32)
    return z * (float(radius) / max(1e-8, np.linalg.norm(z)))


def local_basis(z: np.ndarray, dimensions: int = 8, seed: int = 13) -> np.ndarray:
    """Orthonormal tangent directions around one identity (SplatWorld2 notion)."""
    z = np.asarray(z, np.float64)
    radial = z / (np.linalg.norm(z) + 1e-12)
    rng = np.random.default_rng(seed)
    r = rng.standard_normal((LATENT, dimensions))
    r -= radial[:, None] * (radial @ r)[None, :]
    q, _ = np.linalg.qr(r)
    return q.T.astype(np.float32)


def shell_name(radius: float) -> str:
    """Artistic decoder-state descriptor, NOT calibrated confidence or a phase transition."""
    return 'core' if radius < 15 else 'ghost' if radius < 35 else 'fire'


def image_u8(img: np.ndarray, size: int | None = None) -> np.ndarray:
    import cv2
    rgb = np.uint8(np.clip(img * 255 + .5, 0, 255))
    if size:
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC)
    return rgb
