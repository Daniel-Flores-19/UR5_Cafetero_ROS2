import cv2
import numpy as np


def get_polygons(mask, min_area=200, epsilon=2.0):
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        polygons.append(approx.reshape(-1, 2))
    return polygons


def scanline_fill(mask, spacing=8):
    strokes = []
    for i, y in enumerate(range(0, mask.shape[0], spacing)):
        row = mask[y] > 0
        runs = []
        x = 0
        while x < len(row):
            if row[x]:
                start = x
                while x < len(row) and row[x]:
                    x += 1
                runs.append((start, x - 1))
            else:
                x += 1

        if i % 2 == 1:
            runs = runs[::-1]
        for x1, x2 in runs:
            a, b = (x2, x1) if i % 2 == 1 else (x1, x2)
            strokes.append(((a, y), (b, y)))
    return strokes


def resample_polyline(points, max_spacing, closed=False):
    points = np.asarray(points, dtype=float)
    if closed:
        points = np.vstack([points, points[0]])

    out = []
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        dist = np.linalg.norm(p2 - p1)
        n = max(2, int(np.ceil(dist / max_spacing)) + 1)
        seg = np.linspace(p1, p2, n)
        out.append(seg[:-1] if i < len(points) - 2 else seg)

    return np.vstack(out)