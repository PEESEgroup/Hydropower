"""voronoi_treemap.py - additively-weighted Voronoi (power-diagram) treemap.

Pure numpy. Each cell's area is driven toward a target proportional to its value via
Lloyd relaxation + additive weight adaptation (Balzer/Nocaj-Brandes style). Cells are
convex (intersection of half-planes), clipped to a convex boundary polygon.
No external Voronoi library required.
"""
import numpy as np


def _clip_halfplane(poly, nx, ny, c):
    """Keep the part of convex polygon where nx*x + ny*y <= c (Sutherland-Hodgman)."""
    if len(poly) < 3:
        return np.empty((0, 2))
    out = []
    M = len(poly)
    for k in range(M):
        a = poly[k]
        b = poly[(k + 1) % M]
        fa = nx * a[0] + ny * a[1] - c
        fb = nx * b[0] + ny * b[1] - c
        ina, inb = fa <= 0, fb <= 0
        if ina:
            out.append(a)
        if ina != inb:
            t = fa / (fa - fb)
            out.append(a + t * (b - a))
    return np.array(out) if len(out) >= 3 else np.empty((0, 2))


def poly_area(p):
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def poly_centroid(p):
    if len(p) < 3:
        return p.mean(0) if len(p) else np.zeros(2)
    x, y = p[:, 0], p[:, 1]
    cr = x * np.roll(y, -1) - np.roll(x, -1) * y
    A = cr.sum() / 2.0
    if abs(A) < 1e-12:
        return p.mean(0)
    cx = ((x + np.roll(x, -1)) * cr).sum() / (6 * A)
    cy = ((y + np.roll(y, -1)) * cr).sum() / (6 * A)
    return np.array([cx, cy])


def _cells(sites, w, boundary):
    n = len(sites)
    cells = []
    for i in range(n):
        cell = boundary
        pi = sites[i]
        for j in range(n):
            if i == j:
                continue
            d = sites[j] - pi
            c = 0.5 * (sites[j] @ sites[j] - pi @ pi - (w[j] - w[i]))
            cell = _clip_halfplane(cell, d[0], d[1], c)
            if len(cell) < 3:
                break
        cells.append(cell)
    return cells


def circle_boundary(R=1.0, cx=0.0, cy=0.0, nv=80):
    t = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    return np.column_stack([cx + R * np.cos(t), cy + R * np.sin(t)])


def point_in_poly(p, poly):
    x, y = p
    inside = False
    nn = len(poly)
    j = nn - 1
    for i in range(nn):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside


def voronoi_treemap(values, boundary=None, iters=120, gain=1.0, seed=0, tol=0.01):
    """Return (cells, sites, areas, target). cells[i] is an (M,2) CCW polygon for value i."""
    values = np.asarray(values, float)
    n = len(values)
    if boundary is None:
        boundary = circle_boundary()
    Atot = poly_area(boundary)
    if n <= 1:
        return [boundary.copy()], np.array([boundary.mean(0)]), \
            np.array([Atot]), np.array([Atot]), 0
    target = values / values.sum() * Atot
    bc = boundary.mean(0)
    rad = np.sqrt(Atot / np.pi)

    # init sites by rejection sampling inside the (convex) boundary -> robust for any shape
    rng = np.random.default_rng(seed)
    xmin, ymin = boundary.min(0)
    xmax, ymax = boundary.max(0)
    pts = []
    tries = 0
    while len(pts) < n and tries < n * 4000:
        p = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)])
        if point_in_poly(p, boundary):
            pts.append(p)
        tries += 1
    while len(pts) < n:
        pts.append(bc + rng.uniform(-1e-3, 1e-3, 2))
    sites = np.array(pts)
    w = np.zeros(n)
    eps = 1e-9

    # phase 1: pure Lloyd (w=0) to spread sites evenly
    for _ in range(15):
        cells = _cells(sites, w, boundary)
        for i in range(n):
            if len(cells[i]) >= 3:
                sites[i] = poly_centroid(cells[i])

    # phase 2: radius-based weight adaptation (Nocaj-Brandes) + Lloyd
    best = None
    for it in range(iters):
        cells = _cells(sites, w, boundary)
        areas = np.array([poly_area(c) for c in cells])
        relerr = np.abs(areas - target) / np.maximum(target, eps)
        score = np.median(relerr)
        if best is None or score < best[0]:
            best = (score, [c.copy() for c in cells], areas.copy())
        if relerr.max() < tol:
            break
        for i in range(n):
            if len(cells[i]) >= 3:
                sites[i] = poly_centroid(cells[i])
        # adjust effective radius r_i = sqrt(w_i) by the equiv-circle-radius error
        r = np.sqrt(np.maximum(w - w.min() + eps, eps))
        r += gain * (np.sqrt(target / np.pi) - np.sqrt(np.maximum(areas, eps) / np.pi))
        r = np.maximum(r, eps)
        w = r ** 2
        w -= w.mean()
        # clamp so a power cell cannot swallow its nearest neighbour (avoid empty cells)
        for i in range(n):
            dd = ((sites - sites[i]) ** 2).sum(1)
            dd[i] = np.inf
            j = dd.argmin()
            maxw = w[j] + 0.95 * dd[j]
            if w[i] > maxw:
                w[i] = maxw
    cells = best[1]
    areas = best[2]
    return cells, sites, areas, target, it + 1
