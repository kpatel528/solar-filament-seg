"""Physics-aware preprocessing: solar-disk localisation + limb-darkening flattening."""
import cv2
import numpy as np

SIZE = 2048
_YY, _XX = np.indices((SIZE, SIZE), dtype=np.float32)


def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _disk_by_threshold(img):
    blur = cv2.GaussianBlur(img, (9, 9), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (cx, cy), r = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
    return float(cx), float(cy), float(r)


DISK_R_RANGE = (870, 930)


def find_disk(img, down=4, n_candidates=8, r_range=None):
    """(cx, cy, r) in full-res px. Hough candidates on a downsampled image, re-ranked by
    limb contrast (bright just inside, dark just outside) to reject the off-limb halo ring.
    `r_range` (full-res px) constrains the radius search; used to refit implausible fits."""
    small = cv2.resize(img, (img.shape[1] // down, img.shape[0] // down), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    n = small.shape[0]
    rmin, rmax = (int(0.36 * n), int(0.49 * n)) if r_range is None else (int(r_range[0] / down), int(r_range[1] / down) + 1)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=20,
                               minRadius=rmin, maxRadius=rmax)
    if circles is None:
        return _disk_by_threshold(img)
    yy, xx = np.indices(small.shape, dtype=np.float32)
    best, best_score = None, -np.inf
    for cx, cy, r in circles[0, :n_candidates]:
        rr = np.hypot(xx - cx, yy - cy) / r
        inner, outer = blur[(rr > 0.90) & (rr < 0.97)], blur[(rr > 1.03) & (rr < 1.10)]
        if inner.size == 0 or outer.size == 0:
            continue
        score = float(np.median(inner)) - float(np.median(outer))
        if score > best_score:
            best, best_score = (cx, cy, r), score
    if best is None:
        return _disk_by_threshold(img)
    cx, cy, r = float(best[0] * down), float(best[1] * down), float(best[2] * down)
    if r_range is None and not (DISK_R_RANGE[0] <= r <= DISK_R_RANGE[1]):
        return find_disk(img, down, n_candidates, r_range=DISK_R_RANGE)
    return cx, cy, r


def radius_map(cx, cy, r):
    """Normalised distance from disk centre (1.0 = limb)."""
    return np.hypot(_XX - cx, _YY - cy) / r


def flatten_disk(img, cx, cy, r, n_bins=200, r_max=0.995, smooth=5):
    """Divide out the radial limb-darkening profile (robust 2-pass mean that ignores dark features).
    Returns (flat, rr): flat ~ 1.0 on quiet Sun, filaments < 1, off-disk = 1.0 (neutral)."""
    f = img.astype(np.float32)
    rr = radius_map(cx, cy, r)
    inside = rr < r_max
    b = np.minimum((rr / r_max * n_bins).astype(np.int32), n_bins - 1)
    bi, fi = b[inside], f[inside]
    prof = np.bincount(bi, fi, n_bins) / np.maximum(np.bincount(bi, minlength=n_bins), 1)
    keep = fi > 0.85 * prof[bi]
    prof = np.bincount(bi[keep], fi[keep], n_bins) / np.maximum(np.bincount(bi[keep], minlength=n_bins), 1)
    k = np.ones(2 * smooth + 1) / (2 * smooth + 1)
    prof = np.maximum(np.convolve(np.pad(prof, smooth, mode="edge"), k, mode="valid"), 1.0)
    flat = np.where(inside, f / prof[b], 1.0).astype(np.float32)
    return flat, rr


def to_uint8(flat):
    """Map flat (0.5..1.5) -> uint8 with quiet Sun ~ 127, for caching / CNN input."""
    return np.clip((flat - 0.5) * 255, 0, 255).astype(np.uint8)
