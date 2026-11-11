"""Stage B: learn the bubble layout (row / column positions) from registered pages."""
import json
import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def flatten(page_bgr):
    """Illumination-flattened (R, G) channels, paper ~ 1.0.
    G is dark on pink print AND pencil; R is dark on pencil/black only."""
    f = page_bgr.astype(np.float32)
    out = []
    for c in (2, 1):
        ch = f[..., c]
        bg = cv2.GaussianBlur(cv2.dilate(ch, np.ones((31, 31), np.uint8)), (0, 0), 15)
        out.append(np.clip(ch / np.maximum(bg, 1.0), 0, 1.2))
    return out[0], out[1]


def median_maps(pages):
    rs, gs = [], []
    for p in pages:
        R, G = flatten(p)
        rs.append((R * 200).astype(np.uint8))
        gs.append((G * 200).astype(np.uint8))
    k = len(rs) // 2
    Rm = np.partition(np.stack(rs), k, axis=0)[k].astype(np.float32) / 200
    Gm = np.partition(np.stack(gs), k, axis=0)[k].astype(np.float32) / 200
    return Rm, Gm


def find_marks(Rm):
    H, W = Rm.shape
    m = (Rm < 0.5).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    marks = []
    for (x, y, w, h, a), (cx, cy) in zip(st[1:], cen[1:]):
        if cx > 0.88 * W and 15 <= w <= 70 and 5 <= h <= 26 and w >= 1.5 * h:
            marks.append((float(cy), float(cx), int(w), int(h)))
    return sorted(marks)


def _ink(Gm):
    ink = np.clip(1.0 - Gm, 0, 1)
    b = (ink > 0.15).astype(np.uint8)
    v = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((81, 1), np.uint8))
    ink[cv2.dilate(v, np.ones((5, 5), np.uint8)) > 0] = 0
    return ink


def _runs(peaks, tol=0.3):
    peaks = list(peaks)
    if len(peaks) < 5:
        return []
    d = np.diff(peaks)
    med = np.median(d)
    runs, cur = [], [peaks[0]]
    for p, dd in zip(peaks[1:], d):
        if abs(dd - med) <= tol * med:
            cur.append(p)
        else:
            runs.append(cur); cur = [p]
    runs.append(cur)
    return [r for r in runs if len(r) >= 5]


SIGMAS = (1.5, 2, 2.5, 3, 3.5, 4, 5, 6)


def find_answer_cols(ink, rows_y, x_max):
    y0, y1 = int(min(rows_y) - 10), int(max(rows_y) + 10)
    prof = ink[y0:y1, :int(x_max)].sum(0)
    for sig in SIGMAS:
        s = gaussian_filter1d(prof, sig)
        pk, _ = find_peaks(s, prominence=0.10 * s.max(), distance=5)
        runs = _runs(pk)
        if len(runs) == 4:
            return [[float(x) for x in r[-5:]] for r in runs], sig
    return None, None


def find_digit_cols(ink, dig_rows, x0, x_max):
    y0, y1 = int(min(dig_rows) - 8), int(max(dig_rows) + 8)
    prof = ink[y0:y1, int(x0):int(x_max)].sum(0)
    for sig in SIGMAS:
        s = gaussian_filter1d(prof, sig)
        pk, _ = find_peaks(s, prominence=0.10 * s.max(), distance=5)
        if len(pk) == 8 and np.std(np.diff(pk)) / np.mean(np.diff(pk)) < 0.2:
            return [float(x + x0) for x in pk], sig
    return None, None


def learn_layout(pages, path=None):
    Rm, Gm = median_maps(pages)
    H, W = Rm.shape
    marks = find_marks(Rm)
    rep = {"n_marks": len(marks)}
    layout = {"W": W, "H": H, "report": rep}
    if len(marks) < 40:
        rep["error"] = "fewer than 40 timing marks found"
        return layout, Rm, Gm
    last = marks[-40:]
    ys = [m[0] for m in last]
    layout["digit_rows"], layout["ans_rows"] = ys[:9], ys[9:39]
    mark_left = float(np.median([m[1] - m[2] / 2 for m in last]))
    ink = _ink(Gm)
    cols, s1 = find_answer_cols(ink, layout["ans_rows"], mark_left - 8)
    dcols, s2 = find_digit_cols(ink, layout["digit_rows"], 0.6 * W, mark_left - 8)
    rep.update(answer_sigma=s1, digit_sigma=s2)
    if cols is None or dcols is None:
        rep["error"] = "could not find regular bubble columns"
    layout["ans_cols"], layout["digit_cols"] = cols, dcols
    if cols:
        sp = np.median([np.diff(c) for c in cols])
        rowsp = np.min(np.diff(layout["ans_rows"]))
        layout["rx"], layout["ry"] = float(0.32 * sp), float(0.38 * rowsp)
    if path:
        json.dump(layout, open(path, "w"), indent=1)
    return layout, Rm, Gm