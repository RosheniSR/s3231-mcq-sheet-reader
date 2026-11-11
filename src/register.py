"""Register a photo/scan to a reference page with SIFT + RANSAC homography.
Robust to background, shadows, rotation. Falls back to align.warp_sheet."""
import cv2
import numpy as np
import align

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _prep(gray):
    return _clahe.apply(cv2.GaussianBlur(gray, (3, 3), 0))


class Registrar:
    def __init__(self, template_bgr):
        self.template = template_bgr
        self.th, self.tw = template_bgr.shape[:2]
        self.sift = cv2.SIFT_create(nfeatures=5000)
        g = _prep(cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY))
        self.kp_t, self.des_t = self.sift.detectAndCompute(g, None)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.corners_t = np.float32([[0, 0], [self.tw, 0], [self.tw, self.th], [0, self.th]])

    def homography(self, img, maxside=1400, ratio=0.75, min_in=30):
        h0, w0 = img.shape[:2]
        scale = min(1.0, maxside / max(h0, w0))
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
        g = _prep(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        kp, des = self.sift.detectAndCompute(g, None)
        if des is None or len(kp) < 20:
            return None, 0
        pairs = self.bf.knnMatch(des, self.des_t, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < ratio * n.distance]
        if len(good) < 12:
            return None, len(good)
        src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([self.kp_t[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        Hs, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0, maxIters=5000, confidence=0.995)
        if Hs is None:
            return None, 0
        n_in = int(mask.sum())
        if n_in < min_in:
            return None, n_in
        back = cv2.perspectiveTransform(self.corners_t.reshape(-1, 1, 2), np.linalg.inv(Hs)).reshape(-1, 2)
        if not cv2.isContourConvex(back.astype(np.float32)):
            return None, n_in
        S = np.diag([scale, scale, 1.0])
        return Hs @ S, n_in

    def warp(self, img):
        n = 0
        for maxside, ratio, min_in in ((1400, 0.75, 30), (2200, 0.82, 18)):
            Hf, n = self.homography(img, maxside, ratio, min_in)
            if Hf is not None:
                page = cv2.warpPerspective(img, Hf, (self.tw, self.th), flags=cv2.INTER_AREA,
                                           borderMode=cv2.BORDER_REPLICATE)
                return page, {"method": "sift", "inliers": n}
        page, info = align.warp_sheet(img)
        info.update(method="fallback", inliers=n)
        if page.shape[:2] != (self.th, self.tw):
            page = cv2.resize(page, (self.tw, self.th))
        return page, info