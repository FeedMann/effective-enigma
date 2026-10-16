"""Small, strict reader for Valve SMD files plus rotation helpers.

The retargeter stores all transforms as 4x4 matrices.  SMD angles are radians;
``euler_matrix_zyx`` builds Rz @ Ry @ Rx, the usual Source SMD XYZ Euler
representation.  Parsing is intentionally limited to the SMD v1 nodes,
skeleton, and triangles sections used by this project.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

_NODE_RE = re.compile(r'^\s*(\d+)\s+"((?:[^"\\]|\\.)*)"\s+(-?\d+)\s*$')


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def euler_matrix_zyx(rx, ry, rz):
    """Return the SMD XYZ Euler rotation matrix: Rz(rz) @ Ry(ry) @ Rx(rx)."""
    return _rz(float(rz)) @ _ry(float(ry)) @ _rx(float(rx))


def euler_matrix_yxz(rx, ry, rz):
    """Return a Blender-style YXZ Euler rotation matrix: Rz @ Rx @ Ry."""
    return _rz(float(rz)) @ _rx(float(rx)) @ _ry(float(ry))


def from_matrix_zyx_angles(matrix):
    """Extract (rx, ry, rz) for ``Rz(rz) @ Ry(ry) @ Rx(rx)``."""
    r = np.asarray(matrix, dtype=np.float64)[:3, :3]
    sy = float(np.clip(-r[2, 0], -1.0, 1.0))
    ry = math.asin(sy)
    cy = math.cos(ry)
    if abs(cy) > 1.0e-8:
        rx = math.atan2(r[2, 1], r[2, 2])
        rz = math.atan2(r[1, 0], r[0, 0])
    else:
        # At gimbal lock only rx-rz (or rx+rz) is observable.  Choose rz=0.
        rx = math.atan2(-r[1, 2], r[1, 1])
        rz = 0.0
    return rx, ry, rz


def quat_to_matrix(quaternion_xyzw):
    """Convert an (x, y, z, w) quaternion to a normalized 3x3 matrix."""
    x, y, z, w = (float(v) for v in quaternion_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1.0e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def parse_smd(path, require_triangles=True):
    """Read a Source SMD file into the dictionaries expected by the builder."""
    path = Path(path)
    text = path.read_text(encoding='utf-8-sig', errors='replace')
    lines = text.splitlines()

    nodes = {}
    rest = {}
    verts = []
    normals = []
    uvs = []
    weights = []
    parent_bones = []
    faces = []
    vert_material = []
    mat_per_face = []
    materials = []

    section = None
    skeleton_time = None
    have_triangles = False
    i = 0

    def next_content(start):
        j = start
        while j < len(lines):
            s = lines[j].strip()
            if s and not s.startswith('//'):
                return j, s
            j += 1
        return j, None

    while i < len(lines):
        raw = lines[i].strip()
        i += 1
        if not raw or raw.startswith('//'):
            continue
        if raw in ('nodes', 'skeleton', 'triangles'):
            section = raw
            if raw == 'triangles':
                have_triangles = True
            continue
        if raw == 'end':
            section = None
            skeleton_time = None
            continue

        if section == 'nodes':
            m = _NODE_RE.match(raw)
            if not m:
                raise ValueError(f'{path}: malformed nodes line: {raw!r}')
            bone_id, name, parent = int(m.group(1)), bytes(m.group(2), 'utf-8').decode('unicode_escape'), int(m.group(3))
            if bone_id in nodes:
                raise ValueError(f'{path}: duplicate node id {bone_id}')
            nodes[bone_id] = (name, parent)
            continue

        if section == 'skeleton':
            if raw.startswith('time '):
                try:
                    skeleton_time = int(raw.split()[1])
                except (IndexError, ValueError) as exc:
                    raise ValueError(f'{path}: malformed skeleton time: {raw!r}') from exc
                continue
            fields = raw.split()
            if len(fields) != 7:
                raise ValueError(f'{path}: malformed skeleton transform: {raw!r}')
            bone_id = int(fields[0])
            values = tuple(float(v) for v in fields[1:])
            if skeleton_time == 0:
                rest[bone_id] = values
            continue

        if section == 'triangles':
            # A material line is followed by exactly three SMD vertex lines.
            material = raw.strip().strip('"')
            if material not in materials:
                materials.append(material)
            face = []
            for _ in range(3):
                j, vertex_line = next_content(i)
                if vertex_line is None or vertex_line == 'end':
                    raise ValueError(f'{path}: incomplete triangle for material {material!r}')
                i = j + 1
                fields = vertex_line.split()
                if len(fields) < 9:
                    raise ValueError(f'{path}: malformed SMD vertex: {vertex_line!r}')
                parent = int(fields[0])
                p = tuple(float(v) for v in fields[1:4])
                n = tuple(float(v) for v in fields[4:7])
                uv = tuple(float(v) for v in fields[7:9])
                links = []
                if len(fields) >= 10:
                    link_count = int(fields[9])
                    if link_count < 0 or len(fields) != 10 + link_count * 2:
                        raise ValueError(f'{path}: invalid SMD weight list: {vertex_line!r}')
                    links = [(int(fields[10 + 2 * k]), float(fields[11 + 2 * k])) for k in range(link_count)]
                else:
                    links = [(parent, 1.0)]
                idx = len(verts)
                verts.append(p)
                normals.append(n)
                uvs.append(uv)
                weights.append(links)
                parent_bones.append(parent)
                vert_material.append(material)
                face.append(idx)
            faces.append(face)
            mat_per_face.append(material)
            continue

        # Ignore an optional SMD version declaration and other harmless lines.
        if raw.startswith('version '):
            continue

    if not nodes:
        raise ValueError(f'{path}: no nodes section found')
    if not rest:
        raise ValueError(f'{path}: no time 0 skeleton pose found')
    missing = sorted(set(nodes) - set(rest))
    if missing:
        raise ValueError(f'{path}: missing time 0 transforms for node ids {missing}')
    if require_triangles and (not have_triangles or not faces):
        raise ValueError(f'{path}: no triangle geometry found')

    known = set(nodes)
    for vtx, links in enumerate(weights):
        if not links or any(b not in known for b, _ in links):
            raise ValueError(f'{path}: vertex {vtx} references an unknown or empty bone weight')
        total = sum(w for _, w in links)
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError(f'{path}: vertex {vtx} has invalid total weight {total}')

    return {
        'nodes': nodes,
        'rest': rest,
        'verts': np.asarray(verts, dtype=np.float64),
        'normals': np.asarray(normals, dtype=np.float64),
        'uvs': np.asarray(uvs, dtype=np.float64),
        'weights': weights,
        'parent_bones': parent_bones,
        'faces': faces,
        'vert_material': vert_material,
        'mat_per_face': mat_per_face,
        'materials': materials,
    }
