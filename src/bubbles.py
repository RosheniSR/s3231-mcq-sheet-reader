"""Stage C/D: read pencil marks from a registered page and decode them.

Uses the layout learned by grid.learn_layout (layout.json):
  ans_cols (4 groups x 5 x-positions), ans_rows (30 y-positions),
  digit_cols (8 x-positions), digit_rows (10 y-positions), rx, ry (half-sizes).
Question number = group*30 + row + 1.
"""
import numpy as np
import cv2

LETTERS = "ABCDE"
DEFAULT_PARAMS = dict(a_abs=0.12, a_rel=0.6, d_abs=0.12, d_rel=0.6, bg_sub=True)


def pencil_ink(page_bgr):
    """Pencil-only darkness in [0,1]. Red channel ignores the pink print."""
    r = page_bgr[..., 2].astype(np.float32)
    bg = cv2.GaussianBlur(cv2.dilate(r, np.ones((31, 31), np.uint8)), (0, 0), 15)
    flat = np.clip(r / np.maximum(bg, 1.0), 0, 1.2)
    return np.clip(1.0 - flat, 0, 1)


def _box_means(ink, X, Y, rx, ry):
    ii = cv2.integral(ink)
    H, W = ink.shape
    x0 = np.clip(np.round(X - rx).astype(int), 0, W - 1)
    y0 = np.clip(np.round(Y - ry).astype(int), 0, H - 1)
    x1 = np.clip(np.round(X + rx).astype(int) + 1, 1, W)
    y1 = np.clip(np.round(Y + ry).astype(int) + 1, 1, H)
    x1, y1 = np.maximum(x1, x0 + 1), np.maximum(y1, y0 + 1)
    s = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
    return s / ((x1 - x0) * (y1 - y0))


# Per-form local alignment: the registration can leave a block of bubbles a few pixels off
# (curled paper, perspective). The student-number block (and optionally each answer group) is
# shifted by the (dx, dy) that makes its marks most distinct, i.e. maximises the mean of
# (darkest bubble - median bubble) per row (answers) / per column (digits). Slightly biased to 0.
ALIGN = True
ALIGN_ANSWERS = False   # tested: shifting answer groups locks onto printed structure (blank->A errors), so only the digit block is aligned
SEARCH_DX = np.arange(-10, 11, 2)
SEARCH_DY = np.arange(-14, 15, 2)
SHIFT_PENALTY = 0.004            # relative score loss per pixel of shift


def _best_shift(ink, X, Y, rx, ry, axis):
    """X, Y: (rows, opts) base bubble centres. axis=1: rows are questions; axis=0: columns are digits."""
    dx, dy = np.meshgrid(SEARCH_DX, SEARCH_DY)
    dx, dy = dx.ravel().astype(float), dy.ravel().astype(float)
    m = _box_means(ink, X[None] + dx[:, None, None], Y[None] + dy[:, None, None], rx, ry)   # (S, r, o)
    k = 2 if axis == 1 else 1
    sc = (m.max(k) - np.median(m, k)).mean(1)
    sc = sc * (1 - SHIFT_PENALTY * (np.abs(dx) + np.abs(dy)))
    i = int(np.argmax(sc))
    return float(dx[i]), float(dy[i])


def bubble_scores(page_bgr, layout, align=None):
    """-> A (120,5) answer-bubble darkness, D (8, n_digit_rows) digit-bubble darkness."""
    if not layout.get("ans_cols") or not layout.get("digit_cols"):
        raise ValueError("layout has no bubble columns")
    align = ALIGN if align is None else align
    H, W = layout["H"], layout["W"]
    if page_bgr.shape[:2] != (H, W):
        page_bgr = cv2.resize(page_bgr, (W, H))
    ink = pencil_ink(page_bgr)
    rx, ry = layout["rx"], layout["ry"]
    ac = np.asarray(layout["ans_cols"], float)          # (4,5)
    ar = np.asarray(layout["ans_rows"], float)          # (30,)
    A = np.zeros((4, 30, 5))
    for g in range(4):
        X = np.broadcast_to(ac[g][None, :], (30, 5))
        Y = np.broadcast_to(ar[:, None], (30, 5))
        sx, sy = _best_shift(ink, X, Y, rx, ry, 1) if (align and ALIGN_ANSWERS) else (0.0, 0.0)
        A[g] = _box_means(ink, X + sx, Y + sy, rx, ry)
    A = A.reshape(120, 5)
    dc = np.asarray(layout["digit_cols"], float)        # (8,)
    dr = np.asarray(layout["digit_rows"], float)        # (9,)
    X = np.broadcast_to(dc[:, None], (len(dc), len(dr)))
    Y = np.broadcast_to(dr[None, :], (len(dc), len(dr)))
    rxd = layout.get("rx_d", rx)
    ryd = layout.get("ry_d", ry)
    sx, sy = _best_shift(ink, X, Y, rxd, ryd, 0) if align else (0.0, 0.0)
    D = _box_means(ink, X + sx, Y + sy, rxd, ryd)
    return A, D


def pick_idx(S, t_abs, t_rel, bg_sub=True):
    """Vectorised decision. S: (N, groups, options) -> (N, groups) index, -1 = blank/ambiguous.
    bg_sub removes each form's median bubble darkness (shadows, paper tone)."""
    S = np.asarray(S, float)
    if bg_sub:
        S = S - np.median(S.reshape(len(S), -1), axis=1)[:, None, None]
    o = np.sort(S, axis=-1)
    top, sec = o[..., -1], o[..., -2]
    bad = (top < t_abs) | ((sec >= t_abs) & (sec > t_rel * top))
    return np.where(bad, -1, S.argmax(-1))


def decode(A, D, p=DEFAULT_PARAMS):
    """-> (student_number or None, {"1": "A"|None, ...})."""
    a = pick_idx(A[None], p["a_abs"], p["a_rel"], p["bg_sub"])[0]
    d = pick_idx(D[None], p["d_abs"], p["d_rel"], p["bg_sub"])[0]
    answers = {str(i + 1): (None if k < 0 else LETTERS[k]) for i, k in enumerate(a)}
    sn = None if (d < 0).any() else "".join(str(int(k)) for k in d)
    return sn, answers


def annotate(page_bgr, layout, answers, sn=None):
    """Draw the predicted answers on the registered page (optional extension)."""
    out = page_bgr.copy()
    rx, ry = int(layout["rx"]), int(layout["ry"])
    for g in range(4):
        for r in range(30):
            a = answers.get(str(g * 30 + r + 1))
            if a is None:
                continue
            x, y = layout["ans_cols"][g][LETTERS.index(a)], layout["ans_rows"][r]
            cv2.rectangle(out, (int(x - rx), int(y - ry)), (int(x + rx), int(y + ry)), (0, 160, 0), 2)
    if sn:
        for c, ch in enumerate(sn):
            x, y = layout["digit_cols"][c], layout["digit_rows"][int(ch)]
            cv2.rectangle(out, (int(x - rx), int(y - ry)), (int(x + rx), int(y + ry)), (255, 0, 0), 2)
    return out