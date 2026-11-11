"""Registration with a built-in quality check.
SIFT+RANSAC can return a wrong-but-plausible homography (repetitive bubble grids). We score every
registered page by how well its printed (pink) layout matches the blank template, and if the score
is low we try other SIFT settings and the contour-based warp and keep the best candidate."""
import numpy as np
import cv2
import align
import register
from grid import flatten


def _map(page_bgr):
    """Darkness of print + pencil (G channel), downsampled, zero-mean, unit-norm."""
    _, G = flatten(page_bgr)
    m = cv2.resize(1.0 - G, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    m = cv2.GaussianBlur(m, (0, 0), 1.5)
    m = m - m.mean()
    n = float(np.linalg.norm(m))
    return m / n if n > 0 else m


class RobustRegistrar:
    EXTRA_SIFT = ((1000, 0.70, 25), (1800, 0.80, 25), (2600, 0.85, 15))

    def __init__(self, template_bgr, thr=0.35):
        self.base = register.Registrar(template_bgr)
        self.th, self.tw = template_bgr.shape[:2]
        self.tm = _map(template_bgr)
        self.thr = thr

    def score(self, page):
        """Cosine similarity to the template layout, ~0 = misaligned, high = aligned."""
        if page.shape[:2] != (self.th, self.tw):
            page = cv2.resize(page, (self.tw, self.th))
        return float((_map(page) * self.tm).sum())

    def warp(self, img):
        page, info = self.base.warp(img)
        best = (self.score(page), page, info)
        if best[0] < self.thr:                                    # suspicious -> try alternatives
            cands = []
            for ms, ratio, mi in self.EXTRA_SIFT:
                H, n = self.base.homography(img, ms, ratio, mi)
                if H is not None:
                    p = cv2.warpPerspective(img, H, (self.tw, self.th), flags=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_REPLICATE)
                    cands.append((p, {"method": "sift2", "inliers": n}))
            try:
                p, inf = align.warp_sheet(img)
                if p.shape[:2] != (self.th, self.tw):
                    p = cv2.resize(p, (self.tw, self.th))
                cands.append((p, {"method": "contour", "page_found": inf.get("page_found")}))
            except Exception:
                pass
            for p, inf in cands:
                s = self.score(p)
                if s > best[0]:
                    best = (s, p, inf)
        s, page, info = best
        info = dict(info); info["align_score"] = round(s, 3)
        return page, info