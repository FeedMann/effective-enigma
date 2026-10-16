"""X-Ray source rig: rebuild a bone rig + animation from OMF (+OGF hierarchy/mesh)."""
import numpy as np
import xray
import smd

MATRIX_BONE = np.array([
    [1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]], dtype=np.float64)


def motion_quat_matrix(q_raw):
    """Plugin convention: Quaternion((q[3], q[0], q[1], -q[2])) / 0x7fff"""
    x, y, z, w = [v / 32767.0 for v in q_raw]
    return smd.quat_to_matrix((x, y, -z, w))


def motion_translation(bm, frame):
    if bm.t_present:
        i = min(frame, len(bm.translate) - 1)
        raw = np.array(bm.translate[i], dtype=np.float64)
        if bm.hq:
            return raw * np.array(bm.t_size) + np.array(bm.t_init)
        return raw * np.array(bm.t_size) + np.array(bm.t_init)
    return np.array(bm.translate[0], dtype=np.float64)


def motion_quat(bm, frame, length):
    if bm.r_absent or not bm.quats:
        return bm.quats[0]
    i = min(frame, len(bm.quats) - 1)
    return bm.quats[i]


class SourceRig:
    """A rig defined purely by an OMF: hierarchy + per-frame local transforms."""

    def __init__(self, bone_order, parents, motions, reference_motion, reference_frame=0):
        self.bone_order = bone_order
        self.parents = parents            # name -> parent name or None
        self.motions = motions
        self.ref_motion = reference_motion
        self.ref_frame = reference_frame
        self.index = {n: i for i, n in enumerate(bone_order)}
        self.children = {}
        for n in bone_order:
            self.children.setdefault(self.parents.get(n), []).append(n)
        self.rest_local = self._local_of(reference_motion, reference_frame)
        self.rest_world = self._world(self.rest_local)

    # ---- local transform per bone for one frame of one motion
    def _local_of(self, motion, frame):
        out = {}
        for name in self.bone_order:
            bm = motion.bones[name]
            pos = motion_translation(bm, frame)
            R = motion_quat_matrix(motion_quat(bm, frame, motion.length))
            m = np.eye(4)
            m[:3, :3] = R
            m[:3, 3] = pos
            out[name] = m
        return out

    def _world(self, local):
        world = {}
        for name in self.bone_order:
            p = self.parents.get(name)
            world[name] = local[name] if p is None else world[p] @ local[name]
        return world

    def world_at(self, motion, frame):
        return self._world(self._local_of(motion, frame))

    def rest_positions(self):
        return {n: self.rest_world[n][:3, 3] for n in self.bone_order}


def rig_from_omf(omf_path, parents, reference_motion_name=None, reference_frame=0):
    data = open(omf_path, 'rb').read()
    chunks = {c.id: c.data for c in xray.read_chunks(data)}
    sm = xray.read_smparams(chunks[xray.C_S_SMPARAMS])
    by_id = {}
    for p in sm['partitions']:
        for bi, bn in zip(p['ids'], p['names']):
            by_id[bi] = bn
    order = [by_id[i] for i in sorted(by_id)]
    motions, declared = xray.read_motions(chunks[xray.C_S_MOTIONS], order)
    names = [m['name'] for m in sm['motions']]
    ref = None
    if reference_motion_name:
        for cand in (reference_motion_name, reference_motion_name.replace('eft_tt33_new_', '')):
            if cand in motions:
                ref = motions[cand]
                break
    if ref is None:
        for cand in ('idle', 'eft_tt33_new_idle'):
            if cand in motions:
                ref = motions[cand]
                break
    if ref is None:
        ref = list(motions.values())[0]
    return SourceRig(order, parents, motions, ref, reference_frame), sm


# ------------------------------------------------------------------ OGF
def ogf_children_mesh(ogf_path):
    """Read all child visuals' geometry (plugin space: file (x,y,z) -> (x,z,y))."""
    data = open(ogf_path, 'rb').read()
    t = {c.id: c.data for c in xray.read_chunks(data)}
    children = []
    for k in xray.read_chunks(t[0x9]):
        sub = {c.id: c.data for c in xray.read_chunks(k.data)}
        hdr = sub[0x1]
        version, vtype = hdr[0], hdr[1]
        tex = None
        if 0x2 in sub:
            pr = xray.PackedReader(sub[0x2])
            tex = (pr.gets(), pr.gets())
        verts, normals, uvs, bones = [], [], [], []
        vd = sub[0x3]
        fmt, count = np.frombuffer(vd, dtype=np.uint32, count=2)
        off = 8
        if fmt == 1:      # FVF_1L : pos nrm tan bnm uv bone(u32)
            arr = np.frombuffer(vd, dtype=np.float64, offset=off, count=count * 15)
            arr = arr.reshape(count, 15)
            for i in range(count):
                verts.append((arr[i, 0], arr[i, 2], arr[i, 1]))
                normals.append((arr[i, 3], arr[i, 5], arr[i, 4]))
                uvs.append((arr[i, 12], 1 - arr[i, 13]))
            bidx = np.frombuffer(vd, dtype=np.uint32, offset=off + 56, count=count * 15)[::15]
            bones = list(bidx)
        elif fmt == 0:    # FVF : pos nrm uv
            arr = np.frombuffer(vd, dtype=np.float64, offset=off, count=count * 8).reshape(count, 8)
            for i in range(count):
                verts.append((arr[i, 0], arr[i, 2], arr[i, 1]))
                normals.append((arr[i, 3], arr[i, 5], arr[i, 4]))
                uvs.append((arr[i, 6], 1 - arr[i, 7]))
            bones = [0] * count
        else:
            raise ValueError('fmt %d in child' % fmt)
        idx = sub[0x4]
        n_idx = np.frombuffer(idx, dtype=np.uint32, count=1)[0]
        ind = np.frombuffer(idx, dtype=np.uint16, offset=4, count=n_idx)
        tris = [(int(ind[i]), int(ind[i + 2]), int(ind[i + 1])) for i in range(0, n_idx, 3)]
        children.append({'texture': tex, 'verts': np.array(verts), 'normals': np.array(normals),
                         'uvs': np.array(uvs), 'bone_idx': np.array(bones), 'tris': tris,
                         'shader': sub.get(0x1) and None})
    return children
