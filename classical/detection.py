import cv2
import numpy as np

def detect_edges(preprocessed: np.ndarray) -> np.ndarray:
    """
    Edges are computed automatically from gradient median.
    Two Canny thresholds are added because US images do not have equally strong contrast.
    """
    grad_x = cv2.Scharr(preprocessed, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(preprocessed, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(grad_x, grad_y)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    median_val = float(np.median(magnitude))
    lower_1 = int(max(10, 0.55 * median_val))
    upper_1 = int(min(255, 1.45 * median_val + 20))
    lower_2 = int(max(20, 0.80 * median_val))
    upper_2 = int(min(255, 1.80 * median_val + 30))

    edges_1 = cv2.Canny(magnitude, lower_1, upper_1, apertureSize=3, L2gradient=True)
    edges_2 = cv2.Canny(preprocessed, lower_2, upper_2, apertureSize=3, L2gradient=True)
    edges = cv2.bitwise_or(edges_1, edges_2)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel_open, iterations=1)

    return edges


def _normalize_ellipse(cx, cy, w, h, angle) -> dict:
    """
    OpenCV returns width and height of the ellipse. Here I save them as semi-axes a >= b.
    """
    if w >= h:
        a, b = w / 2, h / 2
        angle_norm = angle % 180
    else:
        a, b = h / 2, w / 2
        angle_norm = (angle + 90) % 180

    return {
        "cx": float(cx), "cy": float(cy),
        "a": float(a), "b": float(b),
        "kut": float(angle_norm)
    }


def detect_ellipses_hough(edges: np.ndarray, img_shape: tuple) -> list[dict]:
    """
    Step 4: detection of elliptical shapes.
    In OpenCV I practically use edges + contours + fitEllipse.
    """
    height, width = img_shape[:2]
    min_dim = min(height, width)
    img_area = height * width

    min_a = min_dim * 0.045
    max_a = min_dim * 0.68
    min_area = max(80, img_area * 0.00025)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for k in contours:
        if len(k) < 5:
            continue

        contour_area = cv2.contourArea(k)
        if contour_area < min_area:
            continue

        try:
            ((cx, cy), (w, h), angle) = cv2.fitEllipse(k)
        except cv2.error:
            continue

        e = _normalize_ellipse(cx, cy, w, h, angle)
        a, b = e["a"], e["b"]

        if not (min_a <= a <= max_a):
            continue
        if b < min_dim * 0.025:
            continue
        if a / b > 4.5:
            continue
        if not (0 < cx < width and 0 < cy < height):
            continue

        ellipse_area = np.pi * a * b
        fill_ratio = contour_area / ellipse_area if ellipse_area > 0 else 0

        # Fill ratio is not strict because edges on US image are often not closed.
        if fill_ratio < 0.015:
            continue

        e["area"] = float(contour_area)
        e["fill_ratio"] = float(fill_ratio)
        e["contour"] = k
        candidates.append(e)

    return candidates


def _point_in_ellipse(x: float, y: float, e: dict) -> bool:
    angle = np.deg2rad(e["kut"])
    dx = x - e["cx"]
    dy = y - e["cy"]
    xr = np.cos(angle) * dx + np.sin(angle) * dy
    yr = -np.sin(angle) * dx + np.cos(angle) * dy
    return (xr / e["a"]) ** 2 + (yr / e["b"]) ** 2 <= 1.25


def select_ellipses(candidates: list[dict], img_shape: tuple) -> dict | None:
    """
    Select the best pair from candidates:
    - thorax is the larger ellipse
    - heart is the smaller ellipse whose center is inside or near the thorax
    - CTR must be in a realistic range
    """
    if len(candidates) < 2:
        return None

    height, width = img_shape[:2]
    center_x, center_y = width / 2, height / 2
    diagonal = np.hypot(width, height)

    sorted_candidates = sorted(candidates, key=lambda e: e["a"] * e["b"], reverse=True)
    best_pair = None
    best_score = -1e9

    for thorax in sorted_candidates[:30]:
        area_t = thorax["a"] * thorax["b"]
        d_t = np.hypot(thorax["cx"] - center_x, thorax["cy"] - center_y) / diagonal

        for heart in sorted_candidates:
            if heart is thorax:
                continue

            area_s = heart["a"] * heart["b"]
            if area_s >= area_t * 0.90:
                continue

            ctr = heart["a"] / thorax["a"] if thorax["a"] > 0 else None
            if ctr is None or not (0.18 <= ctr <= 0.85):
                continue

            dx = abs(heart["cx"] - thorax["cx"])
            dy = abs(heart["cy"] - thorax["cy"])
            if dx > thorax["a"] * 0.85 or dy > thorax["b"] * 0.85:
                continue

            if not _point_in_ellipse(heart["cx"], heart["cy"], thorax):
                continue

            # Score favors larger thorax, sensible CTR and candidates near the center.
            pop_t = min(thorax.get("fill_ratio", 0), 1.0)
            pop_s = min(heart.get("fill_ratio", 0), 1.0)
            ctr_score = 1.0 - abs(ctr - 0.45)
            score = (
                2.0 * (area_t / (width * height)) +
                1.2 * ctr_score +
                0.6 * pop_t +
                0.6 * pop_s -
                0.7 * d_t
            )

            if score > best_score:
                best_score = score
                best_pair = {"cardiac": heart, "thorax": thorax}

    if best_pair is not None:
        return best_pair

    # Fallback: if stricter pair not found, take the two largest ellipses.
    if len(sorted_candidates) >= 2:
        return {"cardiac": sorted_candidates[1], "thorax": sorted_candidates[0]}
    return None

def get_ellipse_mask(e: dict, shape: tuple) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if e is None:
        return mask
    cv2.ellipse(
        mask,
        (int(round(e["cx"])), int(round(e["cy"]))),
        (max(1, int(round(e["a"]))), max(1, int(round(e["b"])))),
        e["kut"],
        0, 360, 255, -1
    )
    return mask

def get_ellipse_iou(auto: dict | None, gt: dict | None, shape: tuple) -> float | None:
    if auto is None or gt is None:
        return None

    m1 = get_ellipse_mask(auto, shape) > 0
    m2 = get_ellipse_mask(gt, shape) > 0
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return None
    return float(intersection / union)

