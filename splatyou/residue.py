"""Lossless telescoping Sihti-style frequency sieve + transported frozen anchor.

This is NOT SihtiImageModel2's learned P3 patch memory. It cannot synthesize
true missing high-frequency data: detail comes only from a provided real anchor.
"""
import cv2
import numpy as np


def bands(img: np.ndarray, sigmas=(1.0, 2.5, 6.0, 13.0)):
    x = np.asarray(img, np.float32)
    out = []
    for s in sigmas:
        smooth = cv2.GaussianBlur(x, (0, 0), float(s), borderType=cv2.BORDER_REFLECT101)
        out.append(x - smooth)
        x = smooth
    return x, out


def compose(core: np.ndarray, residues: list[np.ndarray], gains=None) -> np.ndarray:
    if gains is None:
        gains = [1.] * len(residues)
    x = np.array(core, np.float32, copy=True)
    for r, gain in zip(residues, gains):
        x += np.asarray(r, np.float32) * float(gain)
    return x


def transport_anchor(anchor: np.ndarray, reference: np.ndarray, current: np.ndarray,
                     gain: float = 1.0, max_shift: float = 28.) -> np.ndarray:
    """Warp high bands of a fixed real anchor as ONNX guide changes.

    Flow is current->reference, so each current output pixel samples its
    corresponding source anchor pixel. Avoids feedback-to-feedback blur.
    For a moved/different identity the anchor is invalid; callers must reset.
    """
    h, w = anchor.shape[:2]
    ref = cv2.resize(reference, (w, h), interpolation=cv2.INTER_CUBIC)
    now = cv2.resize(current, (w, h), interpolation=cv2.INTER_CUBIC)
    gray = lambda x: cv2.cvtColor(np.uint8(np.clip(x*255, 0, 255)), cv2.COLOR_RGB2GRAY)
    # Calculate motion at 128px, then scale into the anchor's native canvas.
    work = 128
    now_small = cv2.resize(now, (work, work), interpolation=cv2.INTER_AREA)
    ref_small = cv2.resize(ref, (work, work), interpolation=cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(gray(now_small), gray(ref_small), None, .5, 4, 21, 3, 5, 1.1, 0)
    flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
    flow[:, :, 0] *= (w / work)
    flow[:, :, 1] *= (h / work)
    flow = np.clip(flow, -max_shift, max_shift)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    source_x = xx + flow[:, :, 0]
    source_y = yy + flow[:, :, 1]
    warped = cv2.remap(anchor.astype(np.float32), source_x, source_y,
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
    # A physically supplied anchor may be any image, not necessarily paired.
    # Extract high bands, don't simply paste the anchor RGB onto the face.
    _core, rs = bands(warped)
    detail = compose(np.zeros_like(warped), rs[:2], [1, .6])
    raw = cv2.resize(current, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(raw + float(gain) * detail, 0, 1)
