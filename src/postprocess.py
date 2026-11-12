"""Response map (classical score or CNN probability) -> list of instance RLEs."""
import cv2
import numpy as np
from pycocotools import mask as mask_utils


def instances_from_response(resp, rr, thr, close_px=0, min_area=200, r_limit=0.98, max_instances=150):
    """Threshold -> open -> optional close (bridges small gaps) -> connected components -> RLEs.
    Keeps at most `max_instances` largest components (GT max is 26 per image)."""
    m = ((resp > thr) & (rr < r_limit)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if len(keep) > max_instances:
        keep = sorted(keep, key=lambda i: -stats[i, cv2.CC_STAT_AREA])[:max_instances]
    buf = np.zeros(m.shape, dtype=np.uint8, order="F")
    out = []
    for i in keep:
        x, y, w, h = stats[i, :4]
        buf[y:y + h, x:x + w] = lab[y:y + h, x:x + w] == i
        out.append(mask_utils.encode(buf))
        buf[y:y + h, x:x + w] = 0
    return out
