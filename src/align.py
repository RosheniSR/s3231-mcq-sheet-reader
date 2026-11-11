"""Stage A: find the answer sheet in a photo/scan and warp it to a canonical A4 page."""
import cv2
import numpy as np

W, H = 1240, 1754          # canonical page (A4 portrait, ~150 dpi)
WORK = 1000                # long side used for page detection


def _order_quad(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def _find_page_quad(small):
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    h, w = gray.shape
    best = None
    cands = [cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]]
    cands += [cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)[1] for t in (110, 140, 170)]
    for mask in cands:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 0.25 * h * w:
            continue
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        quad = None
        for eps in (0.02, 0.03, 0.05, 0.08):
            ap = cv2.approxPolyDP(hull, eps * peri, True)
            if len(ap) == 4:
                quad = ap.reshape(4, 2)
                break
        if quad is None:
            quad = cv2.boxPoints(cv2.minAreaRect(hull))
        q = _order_quad(quad)
        wq = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
        hq = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
        ar = max(wq, hq) / max(1.0, min(wq, hq))
        score = cv2.contourArea(q) * (1.0 if 1.25 < ar < 1.6 else 0.3)
        if best is None or score > best[0]:
            best = (score, q)
    if best is None or best[0] < 0.25 * h * w:
        return None
    return best[1]


def _marks_on_right(page):
    g = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    band = int(0.045 * w)
    right = (g[:, w - band - int(0.01 * w): w - int(0.01 * w)] < 70).mean()
    left = (g[:, int(0.01 * w): int(0.01 * w) + band] < 70).mean()
    return right >= left, right, left


def warp_sheet(img):
    """img: BGR image -> (page 1240x1754 BGR with the black marks on the right, info dict)"""
    info = {}
    h0, w0 = img.shape[:2]
    scale = WORK / max(h0, w0)
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    quad = _find_page_quad(small)
    if quad is None:                       # e.g. flat scan filling the frame
        info["page_found"] = False
        quad = np.array([[0, 0], [small.shape[1] - 1, 0],
                         [small.shape[1] - 1, small.shape[0] - 1],
                         [0, small.shape[0] - 1]], dtype=np.float32)
    else:
        info["page_found"] = True
    quad = quad / scale
    wq = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    hq = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
    if wq > hq:
        quad = np.roll(quad, -1, axis=0)
        info["rotated90"] = True
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    page = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_AREA,
                               borderMode=cv2.BORDER_REPLICATE)
    ok, r, l = _marks_on_right(page)
    info["marks_right_left"] = (round(float(r), 4), round(float(l), 4))
    if not ok:
        page = cv2.rotate(page, cv2.ROTATE_180)
        info["rotated180"] = True
    info["quad"] = quad.tolist()
    return page, info