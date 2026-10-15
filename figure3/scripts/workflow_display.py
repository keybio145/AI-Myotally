"""Saved-state Figure 3 display helpers; no segmentation, RF fitting or typing classification."""
import csv
import math
import numpy as np
import cv2
TYPE_COLORS = {
    "Type I": (245, 166, 35),
    "Type IIA": (55, 170, 83),
    "Type IIX": (155, 164, 172),
    "Type IIB": (211, 59, 43),
    "null": (105, 115, 120),
}

TYPE_ORDER = ("Type I", "Type IIA", "Type IIX", "Type IIB")

FILTER_STATUS_ORDER = (
    "manual_keep",
    "manual_reject",
    "rf_reject",
    "geometry_reject",
    "uncertain",
    "retained",
)

FILTER_STATUS_COLORS = {
    "retained": (86, 180, 233),
    "manual_keep": (55, 170, 83),
    "rf_reject": (213, 94, 0),
    "geometry_reject": (220, 20, 60),
    "manual_reject": (204, 121, 167),
    "uncertain": (240, 228, 66),
}

FILTER_STATUS_LABELS = {
    "retained": "Retained",
    "manual_keep": "Keep",
    "rf_reject": "RF exclude",
    "geometry_reject": "Geometry",
    "manual_reject": "Reject",
    "uncertain": "Next label",
}

YELLOW = (230, 205, 0)

GREEN = (35, 170, 85)

GRAY = (112, 120, 132)

RED = (190, 42, 40)

WHITE = (255, 255, 255)

BLACK = (0, 0, 0)

def safe_uint8_gray(image):
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr[..., :3].astype(np.float32).mean(axis=2)
    arr = arr.astype(np.float32, copy=False)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape[:2], dtype=np.uint8)
    low, high = np.percentile(finite, [1.0, 99.8])
    if not np.isfinite(high) or high <= low:
        low, high = float(np.min(finite)), float(np.max(finite))
    if high <= low:
        return np.zeros(arr.shape[:2], dtype=np.uint8)
    return (np.clip((arr - low) / (high - low), 0.0, 1.0) * 255).astype(np.uint8)

def dark_laminin(gray):
    g = np.clip(gray.astype(np.float32) / 255.0 * 0.58, 0.0, 1.0)
    return np.rint(np.dstack([g, g, g]) * 255.0).astype(np.uint8)

def boundary_mask(labels):
    arr = np.asarray(labels)
    if arr.size == 0:
        return np.zeros(arr.shape[:2], dtype=bool)
    b = np.zeros(arr.shape, dtype=bool)
    b |= arr != np.roll(arr, 1, axis=0)
    b |= arr != np.roll(arr, -1, axis=0)
    b |= arr != np.roll(arr, 1, axis=1)
    b |= arr != np.roll(arr, -1, axis=1)
    b &= arr > 0
    b[0, :] = b[-1, :] = False
    b[:, 0] = b[:, -1] = False
    return b

def thicken(mask, amount):
    if amount <= 1:
        return mask.astype(bool)
    kernel = np.ones((int(amount), int(amount)), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

def draw_outline(rgb, labels, color, thickness=5, halo=False):
    b = thicken(boundary_mask(labels), thickness)
    if not np.any(b):
        return rgb
    if halo:
        rgb[thicken(b, thickness + 2)] = BLACK
    rgb[b] = np.asarray(color, dtype=np.uint8)
    return rgb

def blend_mask(rgb, mask, color, alpha):
    mask = np.asarray(mask, dtype=bool)
    if np.any(mask):
        base = rgb[mask].astype(np.float32)
        fg = np.asarray(color, dtype=np.float32)
        rgb[mask] = np.clip(base * (1.0 - alpha) + fg * alpha, 0, 255).astype(np.uint8)
    return rgb

def clamp_roi(roi, shape):
    height, width = shape[:2]
    x, y, w, h = [int(v) for v in roi]
    w = min(max(1, w), width)
    h = min(max(1, h), height)
    x = int(np.clip(x, 0, max(0, width - w)))
    y = int(np.clip(y, 0, max(0, height - h)))
    return x, y, w, h

def integral_sum(integral, x, y, w, h):
    x2 = x + w
    y2 = y + h
    return float(integral[y2, x2] - integral[y, x2] - integral[y2, x] + integral[y, x])

def choose_roi(raw, reference, final_masks, event_mask, roi_size):
    h, w = final_masks.shape[:2]
    roi_w, roi_h = roi_size
    roi_w = min(roi_w, w)
    roi_h = min(roi_h, h)
    tissue = ((raw > 0) | (reference > 0) | (final_masks > 0)).astype(np.uint8)
    boundary = boundary_mask(reference).astype(np.uint8)
    event = event_mask.astype(np.uint8)
    tissue_i = cv2.integral(tissue)
    boundary_i = cv2.integral(boundary)
    event_i = cv2.integral(event)
    area = float(roi_w * roi_h)
    step_x = max(60, roi_w // 8)
    step_y = max(60, roi_h // 8)
    best = None
    best_detail = {}
    for y in range(0, max(1, h - roi_h + 1), step_y):
        for x in range(0, max(1, w - roi_w + 1), step_x):
            tissue_count = integral_sum(tissue_i, x, y, roi_w, roi_h)
            tissue_density = tissue_count / area
            if tissue_density < 0.22:
                continue
            event_count = integral_sum(event_i, x, y, roi_w, roi_h)
            boundary_count = integral_sum(boundary_i, x, y, roi_w, roi_h)
            if event_count < 800 or boundary_count < 2500:
                continue
            event_density = event_count / max(tissue_count, 1.0)
            target_event = math.exp(-((event_density - 0.055) / 0.07) ** 2)
            score = (
                2.7 * target_event
                + 0.55 * math.log1p(event_count)
                + 0.25 * math.log1p(boundary_count)
                + 1.3 * tissue_density
            )
            if event_density > 0.22:
                score -= (event_density - 0.22) * 10.0
            if best is None or score > best:
                best = score
                best_detail = {
                    "x": x,
                    "y": y,
                    "width": roi_w,
                    "height": roi_h,
                    "score": score,
                    "event_density_in_tissue": event_density,
                    "tissue_density": tissue_density,
                    "event_pixels": int(event_count),
                    "boundary_pixels": int(boundary_count),
                }
    if best_detail:
        return clamp_roi(
            (best_detail["x"], best_detail["y"], best_detail["width"], best_detail["height"]),
            final_masks.shape,
        ), best_detail
    return clamp_roi(((w - roi_w) // 2, (h - roi_h) // 2, roi_w, roi_h), final_masks.shape), {
        "roi_source": "center_fallback"
    }

def read_type_assignments(path, sample_id):
    id_to_type = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("Image") and row.get("Image") != sample_id:
                continue
            raw_id = row.get("Mask_ID") or row.get("Fiber_ID")
            type_name = row.get("Fiber_Type") or row.get("Primary_Fiber_Type") or "null"
            try:
                mask_id = int(float(raw_id))
            except (TypeError, ValueError):
                continue
            if type_name not in TYPE_COLORS:
                type_name = "null"
            id_to_type[mask_id] = type_name
    return id_to_type

def manual_labels_from_state(state):
    labels = {}
    for item in state.get("manual_filter_labels", []) or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                labels[int(item[0])] = str(item[1]).lower()
            except (TypeError, ValueError):
                pass
    return labels

def rf_statuses_from_state(state):
    statuses = {}
    for key, value in (state.get("rf_statuses", {}) or {}).items():
        try:
            statuses[int(key)] = str(value).lower()
        except (TypeError, ValueError):
            pass
    return statuses

def status_from_rf(raw_status):
    value = str(raw_status or "").lower()
    if value in {"orange", "rf_reject", "reject", "false", "non_myofiber"}:
        return "rf_reject"
    if value in {"yellow", "uncertain", "next_label"}:
        return "uncertain"
    return "retained"

def build_filter_status_map(reference, manual_labels, rf_statuses, geometry_reject_ids):
    geometry_reject_ids = {int(value) for value in geometry_reject_ids}
    status_map = {}
    for value in np.unique(reference):
        obj_id = int(value)
        if obj_id <= 0:
            continue
        manual = manual_labels.get(obj_id, "")
        if manual == "keep":
            status_map[obj_id] = "manual_keep"
        elif manual == "reject":
            status_map[obj_id] = "manual_reject"
        elif obj_id in geometry_reject_ids:
            status_map[obj_id] = "geometry_reject"
        else:
            status_map[obj_id] = status_from_rf(rf_statuses.get(obj_id))
    return status_map

def filter_preview_panel(background, reference, status_map):
    panel = background.copy()
    present = []
    for status in FILTER_STATUS_ORDER:
        ids = [obj_id for obj_id, mapped in status_map.items() if mapped == status]
        if not ids:
            continue
        present.append(status)
        labels = np.where(np.isin(reference, np.asarray(ids, dtype=reference.dtype)), reference, 0)
        color = FILTER_STATUS_COLORS[status]
        if status != "retained":
            blend_mask(panel, labels > 0, color, 0.62)
        draw_outline(panel, labels, color, thickness=5, halo=False)
    return panel, present

def type_panel(background, masks, id_to_type):
    panel = background.copy()
    for type_name in TYPE_ORDER + ("null",):
        ids = [mid for mid, mapped in id_to_type.items() if mapped == type_name]
        if not ids:
            continue
        mask = np.isin(masks, np.asarray(ids, dtype=masks.dtype))
        blend_mask(panel, mask, TYPE_COLORS[type_name], 0.88)
    draw_outline(panel, masks, (25, 25, 25), thickness=4, halo=False)
    return panel

def build_stage_images(data):
    background = dark_laminin(data["laminin_gray"])
    raw = data["raw"]
    reference = data["reference"]
    final_masks = data["final"]
    removed_by_editing = (raw > 0) & (reference == 0)
    event_mask = (
        (data["non_myofiber"] > 0)
        | (data["geometry"] > 0)
        | removed_by_editing
    )

    initial = background.copy()
    draw_outline(initial, raw, YELLOW, thickness=6, halo=False)

    editing = background.copy()
    if np.any(removed_by_editing):
        blend_mask(editing, removed_by_editing, RED, 0.68)
        removed_labels = np.where(removed_by_editing & (raw > 0), raw, 0)
        draw_outline(editing, removed_labels, RED, thickness=4, halo=False)
    draw_outline(editing, reference, GRAY, thickness=5, halo=False)

    geometry_reject_ids = {
        int(v) for v in np.unique(data["reject_preview"]) if int(v) > 0
    }
    geometry_reject_ids.update(
        int(v) for v in np.unique(data["geometry"]) if int(v) > 0
    )
    non_myofiber_ids = {
        int(v) for v in np.unique(data["non_myofiber"]) if int(v) > 0
    }
    status_map = build_filter_status_map(
        reference,
        data["manual_filter_labels"],
        data["rf_statuses"],
        geometry_reject_ids,
    )
    for obj_id in non_myofiber_ids:
        if obj_id in status_map and status_map[obj_id] == "retained":
            status_map[obj_id] = "rf_reject"
    preview, present_filter_statuses = filter_preview_panel(
        background, reference, status_map
    )

    final = background.copy()
    draw_outline(final, final_masks, GREEN, thickness=6, halo=False)

    typed = type_panel(background, final_masks, data["id_to_type"])
    return {
        "A": initial,
        "B": editing,
        "C": preview,
        "D": final,
        "E": typed,
        "event_mask": event_mask,
        "editing_removed_present": bool(np.any(removed_by_editing)),
        "filter_statuses": present_filter_statuses,
    }
