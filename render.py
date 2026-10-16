"""Tiny numpy software renderer (z-buffer, flat/Lambert shading) for viewmodel proof."""
import numpy as np


def normalize(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 1e-12)


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    f = normalize(target - eye)
    r = normalize(np.cross(f, up))
    u = np.cross(r, f)
    # camera looks down -Z in view space
    M = np.eye(4)
    M[0, :3] = r
    M[1, :3] = u
    M[2, :3] = -f
    M[:3, 3] = -M[:3, :3] @ eye
    return M


class Scene:
    def __init__(self, width=960, height=640, fov_deg=38.0):
        self.w, self.h = width, height
        self.f = 1.0 / np.tan(np.radians(fov_deg) * 0.5)
        self.color = np.zeros((height, width, 3), dtype=np.float64)
        self.depth = np.full((height, width), np.inf)

    def clear(self, bg=(0.06, 0.07, 0.09)):
        self.color[:] = bg
        self.depth[:] = np.inf

    def project(self, pts, view):
        p = (view[:3, :3] @ pts.T).T + view[:3, 3]
        z = -p[:, 2]
        z = np.where(np.abs(z) < 1e-6, 1e-6, z)
        x = self.f * p[:, 0] / z
        y = self.f * p[:, 1] / z
        sx = (x * 0.5 + 0.5) * self.w
        sy = (0.5 - y * 0.5) * self.h
        return np.stack([sx, sy], axis=1), z

    def add_mesh(self, verts, tris, view, color, ambient=0.25, light=(0.35, -0.55, 0.75),
                 only_tris=None, alpha=1.0):
        pts, z = self.project(np.asarray(verts), view)
        light = normalize(np.array(light, dtype=np.float64))
        cols = np.asarray(color, dtype=np.float64)
        tris = np.asarray(tris, dtype=np.int64)
        if only_tris is not None:
            tris = tris[np.asarray(only_tris, dtype=bool)]
        # screen-space backface cull
        p0 = pts[tris[:, 0]]
        p1 = pts[tris[:, 1]]
        p2 = pts[tris[:, 2]]
        area = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
        keep = np.abs(area) > 1e-9
        tris, p0, p1, p2, area = tris[keep], p0[keep], p1[keep], p2[keep], area[keep]
        nrm = np.cross(np.asarray(verts)[tris[:, 1]] - np.asarray(verts)[tris[:, 0]],
                       np.asarray(verts)[tris[:, 2]] - np.asarray(verts)[tris[:, 0]])
        nrm = normalize(nrm)
        flip = area < 0
        nrm[flip] *= -1.0
        shade = np.clip(np.abs(nrm @ light), 0, 1)
        tri_color = cols[None, :] * (ambient + (1 - ambient) * shade)[:, None]
        z0, z1, z2 = z[tris[:, 0]], z[tris[:, 1]], z[tris[:, 2]]
        for k in range(len(tris)):
            x0, y0 = p0[k]
            x1, y1 = p1[k]
            x2, y2 = p2[k]
            minx = int(max(0, np.floor(min(x0, x1, x2))))
            maxx = int(min(self.w - 1, np.ceil(max(x0, x1, x2))))
            miny = int(max(0, np.floor(min(y0, y1, y2))))
            maxy = int(min(self.h - 1, np.ceil(max(y0, y1, y2))))
            if maxx < minx or maxy < miny:
                continue
            xs = np.arange(minx, maxx + 1) + 0.5
            ys = np.arange(miny, maxy + 1) + 0.5
            X, Y = np.meshgrid(xs, ys)
            d = area[k]
            w0 = ((x1 - X) * (y2 - Y) - (x2 - X) * (y1 - Y)) / d
            w1 = ((x2 - X) * (y0 - Y) - (x0 - X) * (y2 - Y)) / d
            w2 = 1.0 - w0 - w1
            mask = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            if not mask.any():
                continue
            zz = w0 * z0[k] + w1 * z1[k] + w2 * z2[k]
            sub = self.depth[miny:maxy + 1, minx:maxx + 1]
            upd = mask & (zz < sub)
            if not upd.any():
                continue
            sub[upd] = zz[upd]
            csub = self.color[miny:maxy + 1, minx:maxx + 1]
            c = tri_color[k]
            if alpha >= 1.0:
                csub[upd] = c
            else:
                csub[upd] = csub[upd] * (1 - alpha) + c * alpha

    def add_points(self, pts, view, color=(1.0, 0.2, 0.2), radius=2):
        p, z = self.project(np.asarray(pts), view)
        for (x, y), zz in zip(p, z):
            xi, yi = int(x), int(y)
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx * dx + dy * dy > radius * radius:
                        continue
                    X, Y = xi + dx, yi + dy
                    if 0 <= X < self.w and 0 <= Y < self.h and zz < self.depth[Y, X]:
                        self.depth[Y, X] = zz
                        self.color[Y, X] = color

    def add_lines(self, pts, view, color=(0.9, 0.9, 0.2), width=1):
        p, z = self.project(np.asarray(pts), view)
        for k in range(len(p) - 1):
            x0, y0 = p[k]
            x1, y1 = p[k + 1]
            n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
            for t in np.linspace(0, 1, n):
                X = int(round(x0 + (x1 - x0) * t))
                Y = int(round(y0 + (y1 - y0) * t))
                if 0 <= X < self.w and 0 <= Y < self.h:
                    self.color[Y, X] = color

    def image(self):
        return np.clip(self.color, 0, 1)
