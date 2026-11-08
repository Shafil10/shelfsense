"""Re-used footage that the keyframe pHash misses (Sprint 5): cropped, zoomed or shifted copies of an old video.

pHash compares whole pictures, so cropping 10% changes it. Here two keyframes are compared by local features:
ORB keypoints are matched (Lowe ratio test), a homography is fitted with RANSAC and the old keyframe is warped
onto the new one. Honest re-filming of the same shelf also matches (same products, same shelf), so the decision
uses what only a copy has: after alignment, a copy is nearly pixel-identical (high correlation, sub-pixel
alignment error), while a new recording differs in light, product positions and camera angle (parallax).
"""
import cv2
import numpy as np

ORB = cv2.ORB_create(nfeatures=1500)
MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING)
WIDTH = 640


def prepare(frame):
    """Grey keyframe at a fixed width + its ORB features (computed once per keyframe)."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    g = cv2.resize(g, (WIDTH, round(g.shape[0] * WIDTH / g.shape[1])), interpolation=cv2.INTER_AREA)
    kp, des = ORB.detectAndCompute(g, None)
    return {"grey": g, "kp": kp, "des": des}


def compare(a, b, ratio=0.75):
    """How well keyframe b (old) maps onto keyframe a (new). Returns a dict of measures (zeros if no fit)."""
    out = {"good": 0, "inliers": 0, "inlier_share": 0.0, "reproj_px": None, "ncc": 0.0}
    if a["des"] is None or b["des"] is None or len(a["kp"]) < 10 or len(b["kp"]) < 10:
        return out
    pairs = MATCHER.knnMatch(a["des"], b["des"], k=2)
    good = [m for m, *n in pairs if n and m.distance < ratio * n[0].distance]
    out["good"] = len(good)
    if len(good) < 15:
        return out
    pa = np.float32([a["kp"][m.queryIdx].pt for m in good])
    pb = np.float32([b["kp"][m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(pb, pa, cv2.RANSAC, 4.0)
    if H is None:
        return out
    inl = mask.ravel().astype(bool)
    if inl.sum() < 4:
        return out
    proj = cv2.perspectiveTransform(pb[inl].reshape(-1, 1, 2), H).reshape(-1, 2)
    out.update(inliers=int(inl.sum()), inlier_share=round(float(inl.mean()), 3),
               reproj_px=round(float(np.median(np.linalg.norm(proj - pa[inl], axis=1))), 2))
    h, w = a["grey"].shape
    warped = cv2.warpPerspective(b["grey"], H, (w, h))
    valid = cv2.erode(cv2.warpPerspective(np.full(b["grey"].shape, 255, np.uint8), H, (w, h)), np.ones((9, 9), np.uint8)) > 0
    if valid.mean() < 0.3:  # the two pictures overlap too little to compare pixels
        return out
    x = cv2.GaussianBlur(a["grey"], (5, 5), 0)[valid].astype(np.float32)
    y = cv2.GaussianBlur(warped, (5, 5), 0)[valid].astype(np.float32)
    x, y = x - x.mean(), y - y.mean()
    out["ncc"] = round(float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-6)), 3)
    return out


def same_picture(m, cfg):
    """A copy: enough aligned points, sub-pixel-ish alignment and near-identical pixels after alignment."""
    return (m["inliers"] >= cfg["orb_min_inliers"] and m["reproj_px"] is not None
            and m["reproj_px"] <= cfg["orb_max_reproj_px"] and m["ncc"] >= cfg["orb_min_ncc"])


def best_matches(new, old):
    """For each new keyframe, the measures of its best-matching old keyframe (by correlation)."""
    none = {"good": 0, "inliers": 0, "inlier_share": 0.0, "reproj_px": None, "ncc": 0.0}
    return [max((compare(a, b) for b in old), key=lambda m: (m["ncc"], m["inliers"]), default=none) for a in new]
