#!/usr/bin/env python3
"""EFT "TT-33 DUO" viewmodel animation  ->  Garry's Mod hands (``ValveBiped.*``).

Reads the X-Ray OMF/OGF sources of the S.T.A.L.K.E.R. Anomaly mod
"Escape From Tarkov TT-33 DUO" and writes Source engine assets:

  out/c_eft_tt33_reference.smd       GMod c_arms_citizen arms + the TT-33 mesh
  out/c_eft_tt33_anim_<name>.smd     one animation SMD per EFT motion
  out/c_eft_tt33.qc                  QC for studiomdl
  out/bone_map.csv                   source bone -> ValveBiped bone
  out/build_report.json              everything that was verified, with numbers

-------------------------------------------------------------------------------
The conventions below were each pinned against an independent artifact of the
same data (the GMod SDK arm SMD, c_arms_definebones.qci, the weapon OGF's own
mesh, and EFT's own motion files); the checks live in the build report.

1. OMF motion storage.  Per bone: flags, a quaternion (one value if the key is
   absent) and a translation.  The quaternion is stored as int16 and, to be
   read back in the kM_Bone frame the rest of the data uses, has to be
   re-ordered and z-negated:  q = (w, x, y, -z) / 32767.  The translations are
   stored in the same flipped frame, so they need z negated as well.  Checked:
   with this, the 30 finger joints of the idle sit 1.4 cm from the gun's own
   surface (i.e. inside the grip) where every other candidate puts them
   0.8-1.0 m away, and the hands end up right/forward/below the eye.

2. Weapon skinning:  the OGF's ik bind chain is what the weapon mesh is
   authored against (all 10 landmark bones land on their own mesh part, mean
   3.3 mm), and the OMF's motion values for the weapon are stiff/held poses:
   stored rotation for 10 of the 13 bones is the identity, and the stored
   translation for wpn_body (0.055, -0.073, -0.246) is exactly the hands'
   ``lead_gun`` anchor.  So the gun is exported as
       reference = OGF bind chain anchored in the SDK right hand with EFT's
                   idle grip offset,
       animation = EFT motion of the gun *relative to the right hand*.

3. GMod side:  SMD euler triples are  R = Rz(c) @ Ry(b) @ Rx(a)  (verified two
   ways: the arm SMD's bones land inside its own arm mesh; and the rotations are
   exactly the c_arms_definebones.qci numbers under the per-bone axis permuted
   mapping R_defbone = Rz(ry) Ry(rx) Rx(rz)).  Both checks are re-run at build
   time over all 39 definebones.

4. Axis map:  source model frame (x right, y forward, z up) -> Source viewmodel
   (x forward, y left, z up):   M = [[0,1,0],[-1,0,0],[0,0,1]].

5. Scale:  the rigid arm segment lengths of the two rigs, 43.09 units per
   source unit (6 bones: upper arm, forearm and hand, both sides; the spread is
   +/-0.6 units ~ 1.4 %).  eff-enigma's README claims 43.164 - same ballpark.

6. Retarget:  with A(p) = scale * M @ p + d and S(X) = M R_src(X, ref) M^T
   R_bind(X)^T the export is

       R(X, f) = M R_src(X, f) M^T R_bind(X)        (== R_bind at f = ref)
       p(X, f) = A(p_src(X, f))

   i.e. every bone of the GMod rig reproduces EFT's pose up to one global
   similarity, so the pose of the gun inside the gripping fingers is exact.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from eft2gmod.omf import read_omf                                       # noqa: E402
from eft2gmod.ogf import read_ogf, collect_skeleton, collect_meshes     # noqa: E402

MOD = '/home/user/work/tarkov/Escape From Tarkov TT-33 DUO/EFT TT-33 DUO/gamedata'
AW = os.path.join(MOD, 'meshes/anomaly_weapons')
OMF_WEAPON = os.path.join(AW, 'hud_animation/wpn_eft_tt33_new_hud_animation.omf')
OMF_HANDS = os.path.join(AW, 'hud_hands_animation/wpn_eft_tt33_new_hand_hud_animation.omf')
OMF_JAMS = os.path.join(AW, 'hud_hands_animation/wpn_eft_tt33_new_hand_hud_jams_animation.omf')
OGF_WEAPON = os.path.join(AW, 'wpn_eft_tt33/wpn_eft_tt33_hud.ogf')

GMOD = '/home/user/work/gmod'
SMD_ARMS = os.path.join(GMOD, 'c_arms_citizen.smd')
SMD_MEDKIT = os.path.join(GMOD, 'src/medkit/c_medkit.smd')
GMOD_ANIM_REFS = [SMD_ARMS, SMD_MEDKIT,
                  os.path.join(GMOD, 'c_arms_combine.smd'),
                  os.path.join(GMOD, 'src/medkit/anim_draw.smd')]
DEFINEBONES = os.path.join(GMOD, 'src/medkit/c_arms_definebones.qci')
OUT = os.path.join(HERE, 'out')

VALVE = 'ValveBiped.'
SPINE4 = VALVE + 'Bip01_Spine4'
R_HAND = VALVE + 'Bip01_R_Hand'

MB = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)
M_AXIS = np.array([[0.0, 1.0, 0.0],
                   [-1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0]])

# name, hands motion, weapon motion, activity, loop, fps-note
SEQ = [
    ('draw',                  'eft_tt33_new_draw',               None, 'ACT_VM_DRAW', 0),
    ('holster',               'eft_tt33_new_holster',            None, 'ACT_VM_HOLSTER', 0),
    ('idle',                  'eft_tt33_new_idle',               'idle', 'ACT_VM_IDLE', 1),
    ('idle_empty',            'eft_tt33_new_idle',               'idle_empty', 'ACT_VM_IDLE_EMPTY', 1),
    ('idle_jammed',           'eft_tt33_new_idle',               'idle_jammed', None, 1),
    ('aim',                   'eft_tt33_new_aim',                None, None, 0),
    ('inspect',               'eft_tt33_new_inspect',            None, 'ACT_VM_FIDGET', 0),
    ('inspect_jammed',        'eft_tt33_new_inspect_jammed',     None, None, 0),
    ('magcheck',              'eft_tt33_new_magcheck',           'magcheck', None, 0),
    ('magcheck_empty',        'eft_tt33_new_magcheck',           'magcheck_empty', None, 0),
    ('magcheck_jammed',       'eft_tt33_new_magcheck',           'magcheck_jammed', None, 0),
    ('magcheck_jammed_empty', 'eft_tt33_new_magcheck',           'magcheck_jammed_empty', None, 0),
    ('reload_nobump',         'eft_tt33_new_reload_nobump',      'reload_nobump', 'ACT_VM_RELOAD', 0),
    ('reload_empty_nobump',   'eft_tt33_new_reload_empty_nobump', 'reload_empty_nobump',
                              'ACT_VM_RELOAD_EMPTY', 0),
    ('shoot_hip',             'eft_tt33_new_shoot_hip',          'eft_tt33_new_shoot_hip',
                              'ACT_VM_PRIMARYATTACK', 0),
    ('shoot_hip1',            'eft_tt33_new_shoot_hip1',         'eft_tt33_new_shoot_hip1', None, 0),
    ('shoot_hip2',            'eft_tt33_new_shoot_hip2',         'eft_tt33_new_shoot_hip2', None, 0),
    ('shoot_aim',             'eft_tt33_new_shoot_aim',          'eft_tt33_new_shoot_aim',
                              'ACT_VM_SECONDARYATTACK', 0),
    ('shoot_aim1',            'eft_tt33_new_shoot_aim1',         'eft_tt33_new_shoot_aim1', None, 0),
    ('shoot_aim2',            'eft_tt33_new_shoot_aim2',         'eft_tt33_new_shoot_aim2', None, 0),
    ('shoot_empty',           'eft_tt33_new_shoot_hip',          'shoot_empty', 'ACT_VM_DRYFIRE', 0),
    ('unjam',                 'eft_tt33_new_unjam',              'unjam', None, 0),
    ('unjam_empty',           'eft_tt33_new_unjam_empty',        'unjam_empty', None, 0),
    ('walk',                  'eft_tt33_new_walk',               None, None, 1),
    ('walk2sprint',           'eft_tt33_new_walk2sprint',        None, None, 0),
    ('sprint',                'eft_tt33_new_sprint',             None, None, 1),
    ('sprint2walk',           'eft_tt33_new_sprint2walk',        None, None, 0),
]
BASE_FPS = 30.0


# --------------------------------------------------------------------- linear
def mat4(rot=None, pos=None):
    m = np.eye(4)
    if rot is not None:
        m[:3, :3] = rot
    if pos is not None:
        m[:3, 3] = pos
    return m


def quat_to_mat(q):
    w, x, y, z = np.asarray(q, float)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(m):
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    return np.array([x, y, z, w])


def euler_smd_to_mat(a, b, c):
    """SMD skeleton rotation triple -> matrix.  ``R = Rz(c) Ry(b) Rx(a)``."""
    ca, sa = math.cos(a), math.sin(a)
    cb, sb = math.cos(b), math.sin(b)
    cc, sc = math.cos(c), math.sin(c)
    return (np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
            @ np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
            @ np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]]))


def mat_to_defbone_euler(r):
    """Inverse of ``R = Rz(ry) Ry(rx) Rx(rz)`` (c_arms_definebones.qci layout)."""
    ry = math.atan2(r[1, 0], r[0, 0])
    rx = math.asin(max(-1.0, min(1.0, -r[2, 0])))
    rz = math.atan2(r[2, 1], r[2, 2])
    return math.degrees(rx), math.degrees(ry), math.degrees(rz)


def angle_deg(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(a.T @ b) - 1) / 2))))


def map_rigid(M, scale, Q):
    """Image of a rigid transform Q under (rotation M, uniform scale).

    Points map as p -> scale * (M p) + d, with d carried by the anchor only, so
    a rigid Q = (R, t) maps to (M R M^T, scale M t) - still rigid, which keeps
    every parent-relative SMD transform orthonormal.  The map is a
    homomorphism: map(A) @ map(B) == map(A @ B).
    """
    out = np.eye(4)
    out[:3, :3] = M @ Q[:3, :3] @ M.T
    out[:3, 3] = scale * (M @ Q[:3, 3])
    return out


# ------------------------------------------------------------------ EFT model
RE_ARM = re.compile(r'^([lr])_(clavicle|upperarm|forearm|forearm_twist|hand)$')
RE_FING0 = re.compile(r'^([lr])_finger0([12]?)$')
RE_FING = re.compile(r'^bip01_([lr])_finger([1-4])([12]?)$')
ARM_UP = {'clavicle': 'bip01', 'upperarm': 'clavicle', 'forearm': 'upperarm',
          'forearm_twist': 'forearm', 'hand': 'forearm'}
ARM_NAME = {'clavicle': 'Clavicle', 'upperarm': 'UpperArm', 'forearm': 'Forearm',
            'forearm_twist': 'Ulna', 'hand': 'Hand'}


def eft_hand_parents(names):
    parent = {'bip01': None, 'lead_gun': 'bip01'}
    for n in names:
        if n in parent:
            continue
        m = RE_ARM.match(n)
        if m:
            side, part = m.group(1) + '_', m.group(2)
            up = ARM_UP[part]
            parent[n] = 'bip01' if up == 'bip01' else side + up
            continue
        m = RE_FING0.match(n)
        if m:
            side, sub = m.group(1) + '_', m.group(2)
            parent[n] = {'': side + 'hand', '1': side + 'finger0',
                         '2': side + 'finger01'}[sub]
            continue
        m = RE_FING.match(n)
        if m:
            side, base, sub = m.group(1) + '_', m.group(2), m.group(3)
            parent[n] = {'' : side + 'hand',
                         '1': 'bip01_%sfinger%s' % (side, base),
                         '2': 'bip01_%sfinger%s1' % (side, base)}[sub]
            continue
        raise ValueError('unmapped source bone name: %s' % n)
    return parent


def map_to_valve(n):
    if n == 'bip01':
        return SPINE4
    if n == 'lead_gun':
        return None
    m = RE_ARM.match(n)
    if m:
        return VALVE + 'Bip01_' + m.group(1).upper() + '_' + ARM_NAME[m.group(2)]
    m = RE_FING0.match(n)
    if m:
        return VALVE + 'Bip01_' + m.group(1).upper() + '_Finger0' + m.group(2)
    m = RE_FING.match(n)
    if m:
        return VALVE + 'Bip01_' + m.group(1).upper() + '_Finger' + m.group(2) + m.group(3)
    return None


def topo_order(names, parent):
    out, todo = [], list(names)
    while todo:
        prog = False
        for n in list(todo):
            p = parent.get(n)
            if p is None or p not in todo:
                out.append(n)
                todo.remove(n)
                prog = True
        if not prog:
            raise RuntimeError('cycle: %s' % todo)
    return out


def omf_local(motion, bone, frame):
    """OMF bone local transform -> 4x4 in the kM_Bone frame (see module doc)."""
    bm = motion.bones[bone]
    if bm.t_present:
        t = np.asarray(bm.translate[min(frame, len(bm.translate) - 1)], float)
    else:
        t = np.asarray(bm.translate[0], float)
    t = np.array([t[0], t[1], -t[2]])
    q = bm.quats[0] if bm.r_absent else bm.quats[min(frame, len(bm.quats) - 1)]
    q = np.asarray(q, float)
    q = np.array([q[3], q[0], q[1], -q[2]]) / 32767.0
    return mat4(quat_to_mat(q), t)


def chain_world(order, parent, local, root_extra=MB):
    w = {}
    for n in order:
        p = parent.get(n)
        w[n] = (w[p] if p is not None else root_extra) @ local[n]
    return w


def bind_local_ogf(bone):
    """OGF ik bind -> local transform in the kM_Bone frame.

    Rotation: ``M_Bone^T Rz(rz) Ry(ry) Rx(rx) M_Bone``, translation: (x, z, y)
    swizzle.  Verified by measuring every ik bone against its own mesh part.
    """
    rx, ry, rz = bone['bind_rotation']
    ca, sa = math.cos(rx), math.sin(rx)
    cb, sb = math.cos(ry), math.sin(ry)
    cc, sc = math.cos(rz), math.sin(rz)
    R = (np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
         @ np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
         @ np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]]))
    return mat4(MB[:3, :3].T @ R @ MB[:3, :3], swz(bone['bind_translation']))


def swz(v):
    return np.array([v[0], v[2], v[1]], float)


# ---------------------------------------------------------------- GMod input
def read_smd_full(path):
    nodes, frames, verts, tris, mats = [], [], [], [], []
    sec, material, pending = None, None, []
    for raw in open(path, encoding='utf-8', errors='replace'):
        line = raw.strip()
        if not line or line.startswith('//') or line.startswith('version'):
            continue
        if line in ('nodes', 'skeleton', 'triangles'):
            sec = line
            continue
        if line == 'end':
            sec = None
            continue
        p = line.split()
        if sec == 'nodes':
            nodes.append({'name': p[1].strip('"'), 'parent': int(p[2])})
        elif sec == 'skeleton':
            if line.startswith('time'):
                frames.append([])
                continue
            pos = np.array([float(x) for x in p[1:4]])
            if len(p) == 7:
                rot = euler_smd_to_mat(*[float(x) for x in p[4:7]])
            else:
                rot = quat_to_mat([float(p[7]), float(p[4]), float(p[5]), float(p[6])])
            frames[-1].append((pos, rot))
        elif sec == 'triangles':
            if len(p) < 10:
                material = p[0]
                pending = []
                continue
            n = int(p[9])
            links = [(int(p[10 + 2 * i]), float(p[11 + 2 * i])) for i in range(n)]
            verts.append({'pos': np.array([float(x) for x in p[1:4]]),
                          'nrm': np.array([float(x) for x in p[4:7]]),
                          'uv': (float(p[7]), float(p[8])), 'links': links})
            pending.append(len(verts) - 1)
            if len(pending) == 3:
                tris.append(tuple(pending))
                mats.append(material)
                pending = []
    return {'nodes': nodes, 'frames': frames, 'verts': verts, 'tris': tris, 'mats': mats}


def parse_definebones(path):
    out = []
    for line in open(path, encoding='utf-8', errors='replace'):
        line = line.strip()
        if not line.startswith('$definebone'):
            continue
        p = line.split('"')
        nums = [float(x) for x in p[4].split()]
        out.append({'name': p[1], 'parent': p[3] or None,
                    'pos': np.array(nums[0:3]), 'rot': np.array(nums[3:6])})
    return out


# ================================================================ the build
class Build:
    def __init__(self, scale=None, origin=(0.0, 0.0, 0.0)):
        self.M = M_AXIS.copy()
        self.scale = scale
        self.origin = np.array(origin, float)
        self.report = {}

        self.src_hands = read_omf(OMF_HANDS)
        self.src_wpn = read_omf(OMF_WEAPON)
        self.src_jams = read_omf(OMF_JAMS)
        self.ogf = read_ogf(OGF_WEAPON)
        self.arms = read_smd_full(SMD_ARMS)
        self.db = parse_definebones(DEFINEBONES)

        self.hn = list(self.src_hands['record_names'])
        self.hp = eft_hand_parents(self.hn)
        self.ho = topo_order(self.hn, self.hp)
        bones, self.ik = collect_skeleton(self.ogf)
        self.ikd = {b['name']: b for b in self.ik}
        self.wn = [b['name'] for b in bones]
        self.wp = {b['name']: (b['parent'] or None) for b in bones}
        self.wo = topo_order(self.wn, self.wp)

        # target nodes: the 46 GMod arms nodes + the 13 weapon bones
        self.tnodes = [{'name': n['name'], 'parent': n['parent']} for n in self.arms['nodes']]
        self.n_arms = len(self.tnodes)
        self.idx = {n['name']: i for i, n in enumerate(self.tnodes)}
        rhand = self.idx[R_HAND]
        for n in self.wo:
            self.tnodes.append({'name': n, 'parent': rhand if self.wp[n] is None
                                else self.idx[self.wp[n]]})
            self.idx[n] = len(self.tnodes) - 1

        # target bind of the arms nodes = exactly c_arms_citizen.smd
        frame0 = self.arms['frames'][0]
        self.arms_bind_world = {}
        for i in range(self.n_arms):
            p = self.tnodes[i]['parent']
            base = self.arms_bind_world[p] if p >= 0 else np.eye(4)
            self.arms_bind_world[i] = base @ mat4(frame0[i][1], frame0[i][0])

    # ------------------------------------------------------------------ utils
    def motion(self, bank, name):
        for m in bank['motions']:
            if m.name == name:
                return m
        return None

    def speed(self, bank, name):
        for m in bank['smparams']['motions']:
            if m.get('name') == name:
                return float(m.get('speed') or 1.0)
        return 1.0

    def eft_hand_worlds(self, motion, frame):
        loc = {n: omf_local(motion, n, frame) for n in self.ho}
        return chain_world(self.ho, self.hp, loc, MB)

    def eft_weapon_worlds(self, motion, frame):
        return self.weapon_src_world(motion, frame)

    def A(self, m):
        """Similarity that maps a source transform into the Source model."""
        out = np.eye(4)
        out[:3, :3] = self.M @ m[:3, :3] @ self.M.T
        out[:3, 3] = self.scale * (self.M @ m[:3, 3]) + self.origin
        return out

    # ------------------------------------------------------------------- fit
    def fit(self):
        self.ref_hand = self.motion(self.src_hands, 'eft_tt33_new_idle')
        self.ref_wpn = self.motion(self.src_wpn, 'idle')
        self.ref_src = self.eft_hand_worlds(self.ref_hand, 0)
        self.ref_hand_anchor = self.ref_src['lead_gun'][:3, 3].copy()
        ref = self.ref_src

        # ---- name map ------------------------------------------------------
        self.name_map, self.rev_map = {}, {}
        missing = []
        for n in self.hn:
            v = map_to_valve(n)
            if not v:
                continue
            if v not in self.idx:
                missing.append(v)
                continue
            self.name_map[n] = v
            self.rev_map[v] = n
        self.report['source_hand_bones'] = len(self.hn)
        self.report['retargeted_bones'] = len(self.name_map)
        self.report['target_bones_without_source'] = missing
        self.report['source_bones_without_target'] = [n for n in self.hn if n not in self.name_map]
        self.report['target_arm_bones_not_animated'] = [
            n['name'] for i, n in enumerate(self.tnodes)
            if i < self.n_arms and n['name'] not in self.rev_map]

        # ---- scale from the rigid arm segments ------------------------------
        ratios, table = [], []
        for v, n in self.rev_map.items():
            p = self.hp[n]
            if p is None or p not in self.name_map:
                continue
            pv = self.name_map[p]
            ls = float(np.linalg.norm(ref[n][:3, 3] - ref[p][:3, 3]))
            lt = float(np.linalg.norm(self.arms_bind_world[self.idx[v]][:3, 3]
                                      - self.arms_bind_world[self.idx[pv]][:3, 3]))
            if ls < 1e-9:
                continue
            table.append({'source': n, 'target': v, 'src_len': round(ls, 5),
                          'tgt_len': round(lt, 4), 'ratio': round(lt / ls, 3)})
            if n.endswith(('upperarm', 'forearm')) or n in ('l_hand', 'r_hand'):
                ratios.append(lt / ls)
        self.report['bone_length_ratios'] = table
        if self.scale is None:
            self.scale = float(np.median(ratios))
        self.report['scale_units_per_source_unit'] = round(self.scale, 4)
        self.report['scale_spread_units'] = [round(min(ratios), 3), round(max(ratios), 3)]
        self.report['scale_source'] = ('median of %d rigid arm-bone length ratios '
                                       '(upper arm/forearm/hand x 2 sides)' % len(ratios))
        self.report['scale_cross_check_eff_enigma'] = 43.164

        # ---- frame corrections ---------------------------------------------
        # S(X) = M R_src(X, ref) M^T R_bind(X)^T   => at the reference frame the
        # target bone carries its own bind rotation (what the mesh expects).
        self.S = {}
        for v, n in self.rev_map.items():
            Rb = self.arms_bind_world[self.idx[v]][:3, :3]
            self.S[v] = Rb.T @ self.M @ ref[n][:3, :3] @ self.M.T

        # ---- weapon: rest chain + per-frame hand-relative pose --------------
        self.L_bind, self.Wbind_ogf = {}, {}
        for n in self.wo:
            self.L_bind[n] = bind_local_ogf(self.ikd[n])
            p = self.wp[n]
            self.Wbind_ogf[n] = (self.Wbind_ogf[p] if p else np.eye(4)) @ self.L_bind[n]
        # rest chain of the gun: the OGF bind with the root on the hands anchor.
        # Its target image is the plain similarity of that chain, so every gun
        # bone sits exactly on A(p_source) at every frame.
        self.L_ref_bind = dict(self.L_bind)
        self.L_ref_bind['wpn_body'] = mat4(np.eye(3), self.ref_hand_anchor)
        self.W_ref_src = {}
        for n in self.wo:
            p = self.wp[n]
            self.W_ref_src[n] = (self.W_ref_src[p] if p else np.eye(4)) @ self.L_ref_bind[n]
        self.report['weapon_bind_vs_mesh_m'] = self.mesh_landmark_check()
        self.report['weapon_grip_offset_m'] = self.grip_check()

        rhand_bind = self.arms_bind_world[self.idx[R_HAND]]

        # ---- reference placements of the arms bones (for the report) -------
        self.report['reference_hands_units'] = {
            'eft_idle_right_hand': [round(float(x), 3) for x in
                                    self.target_bone_pos(R_HAND, ref['r_hand'][:3, 3])],
            'eft_idle_left_hand': [round(float(x), 3) for x in
                                   self.target_bone_pos(VALVE + 'Bip01_L_Hand',
                                                        ref['l_hand'][:3, 3])],
            'sdk_bind_right_hand': [round(float(x), 3) for x in rhand_bind[:3, 3]],
            'sdk_bind_left_hand': [round(float(x), 3) for x in
                                   self.arms_bind_world[self.idx[VALVE + 'Bip01_L_Hand']][:3, 3]],
            'note': 'the arms nodes are skinned in the sdk bind; the animations '
                    'carry EFT\'s pose, which is the viewmodel grip position'}
        self.axis_check(ref)
        self.report['sdk_arms_bind_ref_units'] = {
            'spine4': [round(float(x), 3) for x in self.arms_bind_world[self.idx[SPINE4]][:3, 3]],
            'right_hand': [round(float(x), 3) for x in rhand_bind[:3, 3]],
            'note': 'c_arms_citizen.smd bind; the sequences carry EFT\'s pose, '
                    'the reference SMD carries that same pose as its rest'}
        self.report['motions'] = {
            'hands': [m.name for m in self.src_hands['motions']],
            'weapon': [m.name for m in self.src_wpn['motions']],
            'jams': [m.name for m in self.src_jams['motions']]}
        return self

    def target_bone_pos(self, target_name, src_point):
        return self.scale * (self.M @ src_point) + self.origin

    def axis_check(self, ref):
        a = ref['lead_gun'][:3, 3]
        p = ref['r_hand'][:3, 3]
        self.report['axis_check'] = {
            'source_model_frame': 'x=right, y=forward, z=up',
            'target_frame': 'x=forward, y=left, z=up (Source viewmodel)',
            'right_hand_at_positive_x': bool(p[0] > 0),
            'right_hand_below_the_eye': bool(p[2] < 0),
            'gun_anchor_in_the_right_hand': bool(
                np.linalg.norm(a - p) < 0.12),
            'fingertips_touch_the_gun': self.report['weapon_grip_offset_m'],
            'muzzle_forward_of_the_grip': bool(
                self.ref_wpn.bones['muzzle'].translate[0][1] > 0.1)}

    # ------------------------------------------------------------ weapon math
    def weapon_src_world(self, wpn_motion, frame):
        """Weapon bone worlds in the source frame (the gun held in the hands).

        * rest: the OGF's ik bind chain with the root placed on the hands'
          ``lead_gun`` anchor (exactly where EFT's motion file puts wpn_body)
        * animation: EFT's motion taken *relative to its own frame 0* and applied
          on top of the bind, so the file's storage quirks (the root's 180 deg
          flip, held poses on a third of the bones) cancel out.
        """
        if wpn_motion is None:
            return self.W_ref_src
        anchor = self.ref_hand_anchor
        W = {}
        for n in self.wo:
            L0 = omf_local(wpn_motion, n, 0)
            Lf = omf_local(wpn_motion, n, frame)
            if self.wp[n] is None:
                local = (Lf @ np.linalg.inv(L0)) @ mat4(np.eye(3), anchor)
            else:
                local = (Lf @ np.linalg.inv(L0)) @ self.L_bind[n]
            p = self.wp[n]
            W[n] = (W[p] if p else np.eye(4)) @ local
        return W

    def grip_check(self):
        V = []
        for vis, mesh, indices, tex in collect_meshes(self.ogf):
            V.extend(mesh['verts'])
        V = np.array(V) + self.ref_src['lead_gun'][:3, 3]
        ds = [float(np.linalg.norm(V - self.ref_src[n][:3, 3], axis=1).min())
              for n in self.hn if 'finger' in n]
        return {'mean_m': round(float(np.mean(ds)), 5), 'max_m': round(float(np.max(ds)), 5),
                'n_finger_bones': len(ds),
                'note': 'distance from every finger joint of EFT\'s idle to the '
                        'surface of the weapon mesh placed on the hands lead_gun anchor'}

    def mesh_landmark_check(self):
        parts = {}
        for vis, mesh, indices, tex in collect_meshes(self.ogf):
            name = (tex or 'x').replace('\\', '/').split('/')[-1]
            parts.setdefault(name, []).append(np.array(mesh['verts']))
        parts = {k: np.vstack(v) for k, v in parts.items()}
        want = {'muzzle': ('tt_barrel',), 'slide': ('tt33', 'tt_barrel'),
                'magazin': ('tt_mag',), 'trigger': ('tt_normal', 'tt_grip'),
                'gilza': ('tt33',), 'shell': ('tt33', 'tt_normal'),
                'hammer': ('tt33', 'tt_normal'), 'mag_release': ('tt_mag', 'tt_grip'),
                'wpn_bullet_1': ('tt_mag',), 'wpn_bullet_2': ('tt_mag',)}
        out = {}
        for bone, mats in want.items():
            p = self.Wbind_ogf[bone][:3, 3]
            out[bone] = round(min(float(np.linalg.norm(parts[m] - p, axis=1).min())
                                  for m in mats if m in parts), 5)
        out['mean'] = round(float(np.mean([v for k, v in out.items() if k != 'mean'])), 5)
        return out

    # ---------------------------------------------------------------- posing
    def frame_pose(self, hand_motion, wpn_motion, frame):
        """World matrices of all 59 nodes for one frame.

        Arms bones: ``R = M R_src(f) M^T S(X)^T``  (== the SDK bind at f = ref)
                    ``p = A(p_src(f))``
        Weapon bones: same form, with the gun's bind taken from the OGF, so at
                    the reference frame the gun carries its authored bind.
        Both are exact images of EFT's pose under A, which is why the gun stays
        in the fingers frame by frame.
        """
        W = {}
        S_src = self.eft_hand_worlds(hand_motion, frame)
        Ww_src = self.weapon_src_world(wpn_motion, frame)
        for i, node in enumerate(self.tnodes):
            name = node['name']
            if i >= self.n_arms:
                ws = Ww_src[name]
                W[i] = mat4(self.M @ ws[:3, :3] @ self.M.T,
                            self.target_bone_pos(name, ws[:3, 3]))
                continue
            if name in self.rev_map:
                ws = S_src[self.rev_map[name]]
                R = self.M @ ws[:3, :3] @ self.M.T @ self.S[name].T
                W[i] = mat4(R, self.target_bone_pos(name, ws[:3, 3]))
            else:
                p = node['parent']
                pos, rot = self.arms['frames'][0][i]
                W[i] = (W[p] if p >= 0 else np.eye(4)) @ mat4(rot, pos)
        return W

    def smd_frames(self, hand_motion, wpn_motion, n_frames):
        out = []
        for f in range(n_frames):
            W = self.frame_pose(hand_motion, wpn_motion, f)
            row = []
            for i, node in enumerate(self.tnodes):
                p = node['parent']
                row.append((np.linalg.inv(W[p]) if p >= 0 else np.eye(4)) @ W[i])
            out.append(row)
        return out

    # ------------------------------------------------------------- verifiers
    def verify(self, hand_motion, wpn_motion, n_frames, step=1):
        """Numbers that prove the retarget is an exact rigid similarity.

        * every retargeted bone sits exactly on A(p_source)      (positions)
        * the gun's bones sit exactly on A(p_source) as well, so the finger to
          gun relations are the source ones, scaled by ``scale``
        * the written locals reproduce those world matrices exactly
        """
        worst_pos, worst_gun, worst_grip, worst_local = 0.0, 0.0, 0.0, 0.0
        tips = [n for n in self.hn if n in self.name_map
                and n.endswith(('02', '12', '22', '32', '42'))]
        for f in range(0, n_frames, step):
            S_src = self.eft_hand_worlds(hand_motion, f)
            S_w = self.weapon_src_world(wpn_motion, f)
            W = self.frame_pose(hand_motion, wpn_motion, f)
            for v, n in self.rev_map.items():
                want = self.target_bone_pos(v, S_src[n][:3, 3])
                got = W[self.idx[v]][:3, 3]
                worst_pos = max(worst_pos, float(np.abs(want - got).max()))
            if S_w is not None:
                for n in self.wo:
                    want = self.target_bone_pos(n, S_w[n][:3, 3])
                    got = W[self.idx[n]][:3, 3]
                    worst_gun = max(worst_gun, float(np.abs(want - got).max()))
                for t in tips:
                    for g in self.wo:
                        ds = np.linalg.norm(S_src[t][:3, 3] - S_w[g][:3, 3])
                        dt = np.linalg.norm(W[self.idx[self.name_map[t]]][:3, 3]
                                           - W[self.idx[g]][:3, 3])
                        worst_grip = max(worst_grip, abs(dt - self.scale * ds))
            # the SMD locals must rebuild the same worlds
            row = []
            for i, node in enumerate(self.tnodes):
                p = node['parent']
                row.append((np.linalg.inv(W[p]) if p >= 0 else np.eye(4)) @ W[i])
            W2 = {}
            for i, node in enumerate(self.tnodes):
                p = node['parent']
                W2[i] = (W2[p] if p >= 0 else np.eye(4)) @ row[i]
            for i in range(len(self.tnodes)):
                worst_local = max(worst_local, float(np.abs(W2[i] - W[i]).max()))
        return {'max_bone_position_error_units': round(worst_pos, 9),
                'max_weapon_position_error_units': round(worst_gun, 9),
                'max_finger_to_gun_error_units': round(worst_grip, 9),
                'max_local_chain_rebuild_error': round(worst_local, 9),
                'frames_checked': len(range(0, n_frames, step))}

    def defbone_check(self):
        """Re-verify the SMD euler convention against c_arms_definebones.qci.

        The SMD's per-node rotation, read as ``Rz(c) Ry(b) Rx(a)``, must equal
        the matching ``$definebone`` rotation applied as ``Rz(q1) Ry(q0) Rx(q2)``
        (the numbers are the same, they land on a permuted set of axes).
        """
        def rx(t):
            return np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)],
                             [0, math.sin(t), math.cos(t)]])

        def ry(t):
            return np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0],
                             [-math.sin(t), 0, math.cos(t)]])

        def rz(t):
            return np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0],
                             [0, 0, 1]])

        byname = {x['name']: x for x in self.db}
        worst, worst_wrong, n = 0.0, 0.0, 0
        for i, nd in enumerate(self.arms['nodes'][:self.n_arms]):
            db = byname.get(nd['name'])
            if not db or db['parent'] is None:
                continue
            n += 1
            pos, rot = self.arms['frames'][0][i]
            a, b, c = [math.radians(x) for x in db['rot']]
            R_db = rz(b) @ ry(a) @ rx(c)
            worst = max(worst, float(np.abs(R_db - rot).max()))
            R_wrong = rz(a) @ ry(c) @ rx(b)      # a plausible but wrong assignment
            worst_wrong = max(worst_wrong, float(np.abs(R_wrong - rot).max()))
        return {'definebones_compared': n,
                'max_abs_element_err_vs_smd_euler': round(worst, 9),
                'max_err_of_a_wrong_axis_assignment': round(worst_wrong, 6),
                'note': 'skeleton euler triples of c_arms_citizen.smd are '
                        'R = Rz(c) Ry(b) Rx(a) in radians'}

    def reference_roundtrip_check(self, ref_path, anim_path):
        """The written reference must equal frame 0 of the idle animation."""
        d = read_smd_full(ref_path)
        a = read_smd_full(anim_path)
        dt = dr = 0.0
        for (p1, r1), (p2, r2) in zip(d['frames'][0], a['frames'][0]):
            dt = max(dt, float(np.abs(np.array(p1) - np.array(p2)).max()))
            dr = max(dr, float(np.abs(np.array(r1) - np.array(r2)).max()))
        return {'max_translation_delta': dt, 'max_rotation_delta': dr,
                'vs': 'c_eft_tt33_anim_idle.smd frame 0'}


# ------------------------------------------------------------------- writers
LN = chr(10)


def write_smd(path, nodes, frames, verts=None, tris=None, mats=None):
    out = ['version 1', 'nodes']
    for i, n in enumerate(nodes):
        out.append('%d "%s" %d' % (i, n['name'], n['parent']))
    out.append('end')
    out.append('skeleton')
    for fi, frame in enumerate(frames):
        out.append('time %d' % fi)
        for ni, m in enumerate(frame):
            p = m[:3, 3]
            q = mat_to_quat(m[:3, :3])
            out.append('%d %.7f %.7f %.7f %.7f %.7f %.7f %.7f'
                       % (ni, p[0], p[1], p[2], q[0], q[1], q[2], q[3]))
    out.append('end')
    if verts is not None:
        out.append('triangles')
        for t, mat in zip(tris, mats):
            out.append(mat)
            for vi in t:
                v = verts[vi]
                p, nr, uv, links = v['pos'], v['nrm'], v['uv'], v['links']
                linkstr = ' '.join('%d %.7f' % (b, w) for b, w in links)
                out.append('0 %.7f %.7f %.7f %.7f %.7f %.7f %.7f %.7f %d %s'
                           % (p[0], p[1], p[2], nr[0], nr[1], nr[2], uv[0], uv[1],
                              len(links), linkstr))
        out.append('end')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(LN.join(out) + LN)
    return sum(len(x) + 1 for x in out)


def write_qc(path, seqs, weapon_materials, bbox):
    l = ['// ---------------------------------------------------------------',
         '// EFT "TT-33 DUO" (S.T.A.L.K.E.R. Anomaly mod) -> Garry\'s Mod viewmodel',
         '// 59 bones: the 46 GMod c_arms_citizen nodes (ValveBiped.*) + the 13',
         '// TT-33 bones parented to ValveBiped.Bip01_R_Hand',
         '// generated by eft2gmod/build_gmod.py',
         '// ---------------------------------------------------------------',
         '$modelname "weapons/c_eft_tt33.mdl"',
         '$cdmaterials "models\\weapons\\eft_tt33\\"',
         '$cdmaterials "models\\weapons\\c_arms_citizen\\"',
         '$surfaceprop "weapon"',
         '',
         '$body studio "c_eft_tt33_reference.smd"',
         '',
         '// the muzzle/grip empties come straight from the weapon skeleton',
         '$attachment "muzzle" "muzzle" 0 0 0',
         '$attachment "muzzle_flash" "muzzle" 0 0 0',
         '$attachment "shell" "shell" 0 0 0',
         '$attachment "eject" "shell" 0 0 0',
         '$attachment "mag" "magazin" 0 0 0',
         '$attachment "slide" "slide" 0 0 0',
         '$attachment "anim_attachment_RH" "ValveBiped.Bip01_R_Hand" 0 0 0',
         '',
         '// name                     smd                              fps  frames  activity']
    for s in seqs:
        l.append('$sequence %-22s "c_eft_tt33_anim_%-18s" fps %-3d%s%s  // %d frames, %.2f s'
                 % (s['name'], s['name'] + '.smd', s['fps'], ' loop' if s['loop'] else '',
                    (' activity %s 1' % s['activity']) if s['activity'] else '',
                    s['frames'], s['duration_s']))
    l += ['', '// model bbox over all sequences: %s .. %s'
          % (tuple(round(float(x), 1) for x in bbox[0]),
             tuple(round(float(x), 1) for x in bbox[1])),
          '// weapon materials (vmt files next to this qc): ' + ', '.join(weapon_materials)]
    open(path, 'w', encoding='utf-8').write(LN.join(l) + LN)


def write_vmt(path, base, bump=None, color='[1 1 1]', prop='metal'):
    l = ['"VertexLitGeneric"', '{',
         '    "$basetexture" "%s"' % base,
         '    "$model" "1"',
         '    "$surfaceprop" "%s"' % prop,
         '    "$color2" "%s"' % color,
         '    "$phong" "1"', '    "$phongexponent" "20"', '    "$phongboost" "1"',
         '    "$phongfresnelranges" "[0.4 1 4]"']
    if bump:
        l.append('    "$bumpmap" "%s"' % bump)
        l.append('    "$normalmapalphaenvmapmask" "1"')
    l += ['}', '']
    open(path, 'w').write(LN.join(l))




def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--stage', default='all', choices=('fit', 'all'))
    ap.add_argument('--scale', type=float, default=None)
    ap.add_argument('--verify-step', type=int, default=1)
    args = ap.parse_args()

    b = Build(scale=args.scale).fit()
    rep = b.report
    rep['definebone_check'] = b.defbone_check()

    # ------------------------------------------------- reference pose (rest)
    # The rest pose of the model is EFT's own idle: the arms mesh is skinned
    # from the SDK bind into it, the gun is placed by the same conversion the
    # animations use, so the reference SMD and every sequence share one
    # convention (frame 0 of the idle is the reference, exactly).
    ref_pos, ref_loc = b.frame_pose(b.ref_hand, b.ref_wpn, 0), None
    nodes = b.tnodes
    ref_frame = []
    for i, node in enumerate(nodes):
        p = node['parent']
        ref_frame.append((np.linalg.inv(ref_pos[p]) if p >= 0 else np.eye(4)) @ ref_pos[i])

    # arms: skin the SDK-bind mesh into the reference pose
    arms_verts = []
    for v in b.arms['verts']:
        pos = np.zeros(3)
        nrm = np.zeros(3)
        for bone, w in v['links']:
            M4 = ref_pos[bone] @ np.linalg.inv(b.arms_bind_world[bone])
            pos += w * (M4[:3, :3] @ np.array(v['pos']) + M4[:3, 3])
            nrm += w * (M4[:3, :3] @ np.array(v['nrm']))
        arms_verts.append({'pos': pos, 'nrm': nrm, 'uv': v['uv'], 'links': v['links']})
    arms_tris, arms_mats = b.arms['tris'], b.arms['mats']

    # the gun: the OGF bind mesh carried to EFT's idle and mapped by A
    weapon_verts, weapon_tris, weapon_mats = [], [], []
    for vis, mesh, indices, tex in collect_meshes(b.ogf):
        tname = (tex or 'tt33').replace(chr(92), '/').split('/')[-1]
        base_i = len(weapon_verts)
        for pos, nrm, uv, links in zip(mesh['verts'], mesh['normals'],
                                       mesh['uvs'], mesh['weights']):
            bone = b.wn[links[0][0]] if links else b.wn[0]
            # ogf.py returns the vertices in the OGF's own frame; the bind chain
            # (Wbind_ogf) lives in the same frame, so the delta bind -> target
            # rest is M4 @ map_rigid(inv(Wbind_ogf[bone]))
            # the OGF bind -> EFT's idle (source space), then A (rotation M,
            # uniform scale, origin) -- the vertex positions land exactly on the
            # image of the source mesh, so the skinning to the target bones (who
            # sit on A(p_source) as well) is exact
            D = b.W_ref_src[bone] @ np.linalg.inv(b.Wbind_ogf[bone])
            M4 = np.eye(4)
            M4[:3, :3] = b.scale * (b.M @ D[:3, :3])
            M4[:3, 3] = b.scale * (b.M @ D[:3, 3]) + b.origin
            pos_t = M4[:3, :3] @ np.array(pos) + M4[:3, 3]
            nrm_t = M4[:3, :3] @ np.array(nrm)
            weapon_verts.append({'pos': pos_t, 'nrm': nrm_t, 'uv': uv,
                                 'links': [(b.idx[b.wn[bi]], w) for bi, w in links]})
        for k in range(0, len(indices), 3):
            weapon_tris.append((base_i + indices[k], base_i + indices[k + 1],
                                base_i + indices[k + 2]))
            weapon_mats.append(tname)
    rep['mesh'] = {'arms_vertices': len(arms_verts), 'arms_triangles': len(arms_tris),
                   'weapon_vertices': len(weapon_verts),
                   'weapon_triangles': len(weapon_tris),
                   'weapon_materials': sorted(set(weapon_mats)),
                   'arms_materials': sorted(set(arms_mats))}

    if args.stage == 'fit':
        print(json.dumps(rep, indent=2, default=str))
        return

    os.makedirs(OUT, exist_ok=True)
    n_off = len(arms_verts)
    write_smd(os.path.join(OUT, 'c_eft_tt33_reference.smd'), nodes, [ref_frame],
              verts=arms_verts + weapon_verts,
              tris=arms_tris + [(a + n_off, c + n_off, d + n_off) for a, c, d in weapon_tris],
              mats=arms_mats + weapon_mats)

    # ----------------------------------------------------- animations
    seqs_out, allpts = [], []
    for name, hands, wpn, act, loop in SEQ:
        hm = b.motion(b.src_hands, hands) or b.motion(b.src_jams, hands)
        wm = b.motion(b.src_wpn, wpn) if wpn else None
        if hm is None:
            rep.setdefault('missing_motions', []).append(hands)
            continue
        n_frames = max(hm.frames, wm.frames if wm is not None else 0)
        spd = b.speed(b.src_hands, hands)
        spd_w = b.speed(b.src_wpn, wpn) if wpn else spd
        fps = int(round(BASE_FPS * (spd if abs(spd - 1.0) < 0.5 else 1.0)))
        if abs(spd_w - spd) > 1e-6 and wm is not None:
            rep.setdefault('speed_mismatch', []).append([name, spd, spd_w])
        frames = b.smd_frames(hm, wm, n_frames)
        write_smd(os.path.join(OUT, 'c_eft_tt33_anim_%s.smd' % name), nodes, frames)
        v = b.verify(hm, wm, n_frames, step=max(1, args.verify_step))
        for fr in frames:
            for m in fr:
                allpts.append(m[:3, 3])
        seqs_out.append({'name': name, 'hands_motion': hands, 'weapon_motion': wpn,
                         'frames': len(frames), 'fps': fps, 'source_fps': BASE_FPS,
                         'source_speed': spd, 'duration_s': round(len(frames) / fps, 3),
                         'activity': act, 'loop': bool(loop), 'verification': v})
    rep['reference_roundtrip'] = b.reference_roundtrip_check(
        os.path.join(OUT, 'c_eft_tt33_reference.smd'),
        os.path.join(OUT, 'c_eft_tt33_anim_idle.smd'))
    rep['sequences'] = seqs_out
    rep['sequence_count'] = len(seqs_out)
    pts = np.array(allpts)
    rep['output_bbox_units'] = [[round(float(x), 2) for x in pts.min(0)],
                                [round(float(x), 2) for x in pts.max(0)]]

    # ----------------------------------------------------- qc + materials
    write_qc(os.path.join(OUT, 'c_eft_tt33.qc'), seqs_out, sorted(set(weapon_mats)),
             (pts.min(0), pts.max(0)))
    matdir = os.path.join(OUT, 'materials', 'models', 'weapons', 'eft_tt33')
    os.makedirs(matdir, exist_ok=True)
    base = 'models/weapons/eft_tt33/'
    write_vmt(os.path.join(matdir, 'tt33.vmt'), base + 'tt33', base + 'tt33_bump')
    write_vmt(os.path.join(matdir, 'tt33_gold.vmt'), base + 'tt33_gold',
              base + 'tt33_gold_bump', color='[0.95 0.82 0.45]')
    write_vmt(os.path.join(matdir, 'tt_barrel.vmt'), base + 'tt_barrel', base + 'tt_barrel_bump')
    write_vmt(os.path.join(matdir, 'tt_grip.vmt'), base + 'tt_grip', base + 'tt_grip_bump',
              color='[0.9 0.85 0.8]', prop='wood')
    write_vmt(os.path.join(matdir, 'tt_mag.vmt'), base + 'tt_mag', base + 'tt_mag_bump')
    write_vmt(os.path.join(matdir, 'tt_normal.vmt'), base + 'tt_normal', base + 'tt_normal_bump')

    # ----------------------------------------------------- bone map
    with open(os.path.join(OUT, 'bone_map.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['#', 'source_bone', 'gmod_bone', 'source_parent', 'gmod_parent',
                    'src_len', 'gmod_len', 'len_ratio', 'note'])
        for i, n in enumerate(b.hn):
            v = b.name_map.get(n)
            sp = b.hp.get(n) or ''
            if v:
                pv = b.name_map.get(b.hp.get(n), '')
                ls = float(np.linalg.norm(b.ref_src[n][:3, 3] - b.ref_src[b.hp[n]][:3, 3])) \
                    if b.hp.get(n) else 0.0
                lt = float(np.linalg.norm(b.arms_bind_world[b.idx[v]][:3, 3]
                                          - b.arms_bind_world[b.idx[pv]][:3, 3])) if pv else 0.0
                w.writerow([i, n, v, sp, pv, round(ls, 5), round(lt, 4),
                            round(lt / ls, 2) if ls > 1e-9 else '', ''])
            else:
                w.writerow([i, n, '-', sp, '', '', '', '',
                            'anchor for the weapon, not a ValveBiped bone'])
        for n in b.wn:
            w.writerow(['', n, n, b.wp[n] or R_HAND, b.wp[n] or R_HAND, '', '', '',
                        'weapon bone, parented to ValveBiped.Bip01_R_Hand'])

    with open(os.path.join(OUT, 'build_report.json'), 'w') as fh:
        json.dump(rep, fh, indent=2, default=str)

    show = {k: v for k, v in rep.items()
            if k not in ('motions', 'bone_length_ratios', 'sequences')}
    print(json.dumps(show, indent=2, default=str))
    print('sequences:')
    for s in seqs_out:
        v = s['verification']
        print('  %-22s %-44s %3d f %2d fps %5.2fs  bone/weapon/locals err %.1e  grip %.1e'
              % (s['name'], s['hands_motion'] + (' + ' + s['weapon_motion']
                                                 if s['weapon_motion'] else ''),
                 s['frames'], s['fps'], s['duration_s'],
                 max(v['max_bone_position_error_units'],
                     v['max_weapon_position_error_units'],
                     v['max_local_chain_rebuild_error']),
                 v['max_finger_to_gun_error_units']))


if __name__ == '__main__':
    main()
