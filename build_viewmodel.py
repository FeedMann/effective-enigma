"""Build a GMod/Source viewmodel SMD from the X-Ray EFT TT-33 hands + weapon rips.

  source hands rig (42 bones, OMF)  --retarget-->  GMod arms SMD skeleton (ValveBiped.*)
  source weapon rig (13 bones, OMF) + weapon OGF mesh  --attach-->  right hand

All math is done in the X-Ray "source" space and mapped to the SMD space with a
similarity C (scale / rotation / translation).  Because linear blend skinning
commutes with an affine map, a single formula covers everything:

    W_target(bone, f) = C o W_source(bone, f) o C^-1      (bone frame transform)
    v_target(f)       = C ( v_source(f) )                 (skinned geometry)

The SMD skeleton (= bind pose) is the source reference pose (idle frame 0),
which is a proper two-handed hold, and every clip's frames are written as local
transforms of the same bind, exactly like a normal studiomdl sequence.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'tools'))

import ogf as ogfmod
import rig
import smd
import xray
from viewmodel import BONE_MAP, EXTRA_MAP, HANDS_HP, Placement, frame_matrix

MOD = os.path.join(HERE, 'mod', 'Escape From Tarkov TT-33 DUO', 'EFT TT-33 DUO', 'gamedata', 'meshes')
HANDS_OMF = os.path.join(MOD, 'anomaly_weapons', 'hud_hands_animation', 'wpn_eft_tt33_new_hand_hud_animation.omf')
HANDS_JAM_OMF = os.path.join(MOD, 'anomaly_weapons', 'hud_hands_animation', 'wpn_eft_tt33_new_hand_hud_jams_animation.omf')
GUN_OMF = os.path.join(MOD, 'anomaly_weapons', 'hud_animation', 'wpn_eft_tt33_new_hud_animation.omf')
GUN_OGF = os.path.join(MOD, 'anomaly_weapons', 'wpn_eft_tt33', 'wpn_eft_tt33_hud.ogf')
ARMS_SMD = os.path.join(HERE, 'ee', '8d7a3f28ba34eb5.smd')
OUT = os.path.join(HERE, 'out')

# X-Ray visual model space -> animation space (verified from geometry:
# barrel +Y_model -> forward +Z_anim ; up +Z_model -> -Y_anim)
RX90 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

REF_MOTION_HANDS = 'eft_tt33_new_idle'
REF_MOTION_GUN = 'idle'
REF_FRAME = 0

# clip -> (hands bank motion, weapon bank motion, out name)
CLIPS = [
    ('eft_tt33_new_idle', 'idle', 'idle'),
    ('eft_tt33_new_aim', 'idle', 'aim'),
    ('eft_tt33_new_reload_nobump', 'reload_nobump', 'reload'),
    ('eft_tt33_new_reload_empty_nobump', 'reload_empty_nobump', 'reload_empty'),
    ('eft_tt33_new_draw', 'idle', 'draw'),
    ('eft_tt33_new_holster', 'idle', 'holster'),
    ('eft_tt33_new_fire', 'eft_tt33_new_shoot_hip', 'fire'),
]
SHOT_CLIPS = ['eft_tt33_new_shoot_hip', 'eft_tt33_new_shoot_hip1', 'eft_tt33_new_shoot_hip2',
              'eft_tt33_new_shoot_aim', 'eft_tt33_new_shoot_aim1', 'eft_tt33_new_shoot_aim2']
EXTRA_CLIPS = [
    ('eft_tt33_new_walk', 'idle', 'walk'),
    ('eft_tt33_new_sprint', 'idle', 'sprint'),
    ('eft_tt33_new_magcheck', 'magcheck', 'magcheck'),
    ('eft_tt33_new_inspect', 'idle', 'inspect'),
]


def mat(rot=None, pos=None):
    m = np.eye(4)
    if rot is not None:
        m[:3, :3] = rot
    if pos is not None:
        m[:3, 3] = pos
    return m


def chain(order, parents, loc):
    world = {}
    for n in order:
        p = parents.get(n)
        world[n] = loc[n] if p is None else world[p] @ loc[n]
    return world


def apply_point(C, p):
    return C.scale * (C.R @ np.asarray(p).reshape(-1, 3).T).T + C.t


def apply_bone(C, M):
    """transform a bone frame (rigid world matrix) into the target space"""
    out = np.eye(4)
    out[:3, :3] = C.R @ M[:3, :3] @ C.R.T
    out[:3, 3] = C.scale * (C.R @ M[:3, 3]) + C.t
    return out


class Builder:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.report = {}
        self.load()
        self.fit()
        self.make_bind()

    # ------------------------------------------------------------------ load
    def load(self):
        self.arms = smd.parse_smd(ARMS_SMD)
        self.nodes = [self.arms['nodes'][i] for i in sorted(self.arms['nodes'])]
        self.idx = {n[0]: i for i, n in enumerate(self.nodes)}
        self.parents = {i: self.nodes[i][1] for i in range(len(self.nodes))}
        self.rest_local = {}
        for i in sorted(self.arms['nodes']):
            px, py, pz, rx, ry, rz = self.arms['rest'][i]
            self.rest_local[i] = mat(smd.euler_matrix_zyx(rx, ry, rz), (px, py, pz))
        self.rest_world = chain(list(range(len(self.nodes))), self.parents, self.rest_local)

        self.hands, self.hands_sm = rig.rig_from_omf(HANDS_OMF, HANDS_HP, REF_MOTION_HANDS, REF_FRAME)
        self.hands_jam = None
        if os.path.exists(HANDS_JAM_OMF):
            self.hands_jam, _ = rig.rig_from_omf(HANDS_JAM_OMF, HANDS_HP, REF_MOTION_HANDS, 0)

        gt = {c.id: c.data for c in xray.read_chunks(open(GUN_OGF, 'rb').read())}
        gn = ogfmod.read_bone_names(gt[0xD])
        self.gun_parents = {b['name']: (b['parent'] or None) for b in gn}
        self.gun, self.gun_sm = rig.rig_from_omf(GUN_OMF, self.gun_parents, REF_MOTION_GUN, REF_FRAME)
        self.gun_parts = rig.ogf_children_mesh(GUN_OGF)

        self.mapped = {s_: self.idx[t_] for s_, t_ in BONE_MAP.items()}
        self.extra = {}
        for s_, targets in EXTRA_MAP.items():
            for t_ in targets:
                self.extra[t_] = s_
        self.unmapped_idx = [i for i, (nm, _) in enumerate(self.nodes)
                             if nm not in self.mapped and nm not in [] and all(nm not in v for v in EXTRA_MAP.values())]

    # ---------------------------------------------------------------- fit / similarity
    def fit(self):
        src = self.hands.rest_world
        f_src = self.gun.rest_world['muzzle'][:3, 3] - self.gun.rest_world['wpn_body'][:3, 3]
        f_src /= np.linalg.norm(f_src)
        l_src = src['l_hand'][:3, 3] - src['r_hand'][:3, 3]
        l_src += -np.dot(l_src, f_src) * f_src
        l_src /= np.linalg.norm(l_src)

        t_hand = self.rest_world[self.idx['ValveBiped.Bip01_R_Hand']][:3, 3]
        t_hand_l = self.rest_world[self.idx['ValveBiped.Bip01_L_Hand']][:3, 3]
        l_tgt = t_hand_l - t_hand
        l_tgt /= np.linalg.norm(l_tgt)
        f_tgt = np.array([0.0, -1.0, 0.0])
        f_tgt = f_tgt - np.dot(f_tgt, l_tgt) * l_tgt
        f_tgt /= np.linalg.norm(f_tgt)

        segs = [('l_upperarm', 'l_forearm', 'ValveBiped.Bip01_L_UpperArm', 'ValveBiped.Bip01_L_Forearm'),
                ('l_forearm', 'l_hand', 'ValveBiped.Bip01_L_Forearm', 'ValveBiped.Bip01_L_Hand'),
                ('r_upperarm', 'r_forearm', 'ValveBiped.Bip01_R_UpperArm', 'ValveBiped.Bip01_R_Forearm'),
                ('r_forearm', 'r_hand', 'ValveBiped.Bip01_R_Forearm', 'ValveBiped.Bip01_R_Hand')]
        num = den = 0.0
        seg_rep = []
        for a_s, b_s, a_t, b_t in segs:
            ls = float(np.linalg.norm(src[b_s][:3, 3] - src[a_s][:3, 3]))
            lt = float(np.linalg.norm(self.rest_world[self.idx[b_t]][:3, 3] - self.rest_world[self.idx[a_t]][:3, 3]))
            num += ls * lt
            den += ls * ls
            seg_rep.append([a_s + '->' + b_s, round(ls, 4), round(lt, 3)])
        s = num / den
        src_mid = 0.5 * (src['l_hand'][:3, 3] + src['r_hand'][:3, 3])
        tgt_mid = 0.5 * (t_hand + t_hand_l)
        R_sim = frame_matrix(f_tgt, np.cross(l_tgt, f_tgt)) @ frame_matrix(f_src, np.cross(l_src, f_src)).T
        C = Placement(f_src, l_src, f_tgt, l_tgt, s, tgt_mid - s * (R_sim @ src_mid))
        self.C = C
        self.report['fit'] = {
            'scale_units_per_metre': round(s, 3),
            'source_forward': [round(x, 4) for x in f_src],
            'source_left': [round(x, 4) for x in l_src],
            'target_forward': [round(x, 4) for x in f_tgt],
            'target_left': [round(x, 4) for x in l_tgt],
            'matched_segments': seg_rep,
        }

    # ------------------------------------------------------------------ bind
    def make_bind(self):
        # bind skeleton in the target space = C o source reference skeleton
        self.bind_world = {}
        for s_, i in self.mapped.items():
            self.bind_world[i] = apply_bone(self.C, self.hands.rest_world[s_])
        for t_, s_ in self.extra.items():
            self.bind_world[self.idx[t_]] = self.bind_world[self.mapped[s_]].copy()
        for i in self.unmapped_idx:
            p = self.parents[i]
            local = np.linalg.inv(self.rest_world[p]) @ self.rest_world[i] if p >= 0 else self.rest_local[i]
            self.bind_world[i] = self.bind_world[p] @ local if p >= 0 else local

        self.gun_names = self.gun.bone_order
        self.gun_off = len(self.nodes)
        self.gun_idx = {n: self.gun_off + k for k, n in enumerate(self.gun_names)}
        self.total_bones = len(self.nodes) + len(self.gun_names)
        for n in self.gun_names:
            self.bind_world[self.gun_idx[n]] = apply_bone(self.C, self.gun.rest_world[n])

        # parents of the output skeleton: gun root hangs off the right hand
        self.out_parent = dict(self.parents)
        for n in self.gun_names:
            gp = self.gun_parents[n]
            self.out_parent[self.gun_idx[n]] = self.gun_idx[gp] if gp else self.idx['ValveBiped.Bip01_R_Hand']

        self.Q = self.gun.rest_world['wpn_body'] @ RX90          # model space -> anim space (reference)
        self.bind_local = {}
        for i in range(self.total_bones):
            p = self.out_parent[i]
            self.bind_local[i] = self.bind_world[i] if p < 0 else np.linalg.inv(self.bind_world[p]) @ self.bind_world[i]

        # ---------------- bind geometry
        v0 = self.arms['verts']
        n0 = self.arms['normals']
        vb = np.zeros_like(v0)
        nb = np.zeros_like(n0)
        for vi in range(len(v0)):
            acc = np.zeros(3)
            accn = np.zeros(3)
            tot = 0.0
            for b, w in self.arms['weights'][vi]:
                M = self.bind_world[b] @ np.linalg.inv(self.rest_world[b])
                acc += w * (M[:3, :3] @ v0[vi] + M[:3, 3])
                accn += w * (M[:3, :3] @ n0[vi])
                tot += w
            vb[vi] = acc / tot
            nb[vi] = accn / tot
        self.arm_bind_verts = vb
        self.arm_bind_normals = nb / np.linalg.norm(nb, axis=1, keepdims=True)

        blocks = []
        for part in self.gun_parts:
            b = int(part['bone_idx'][0])
            name = self.gun_names[b]
            v = apply_point(self.C, (self.Q[:3, :3] @ part['verts'].T).T + self.Q[:3, 3])
            n = (self.C.R @ (self.Q[:3, :3] @ part['normals'].T)).T
            blocks.append({'bone': self.gun_idx[name], 'verts': v, 'normals': n,
                           'uvs': part['uvs'], 'tris': part['tris'],
                           'material': 'tt33_' + (part['texture'][0].split('\\')[-1] if part['texture'] else 'body')})
        self.gun_blocks = blocks

    # ---------------------------------------------------------------- frame evaluation
    def frames(self, hands_motion, gun_motion, stride=1):
        n = hands_motion.length
        S0 = self.hands.rest_local
        out = []
        for f in range(0, n + 1, stride):
            S = self.hands._local_of(hands_motion, f)
            # accumulation of parent-relative deltas (rigid, no telescoping)
            A = {}
            for name in self.hands.bone_order:
                p = self.hands.parents.get(name)
                A[name] = (S[name] @ A[p] @ np.linalg.inv(S0[name])) if p else (S[name] @ np.linalg.inv(S0[name]))
            W = {}
            for s_, i in self.mapped.items():
                W[i] = A[s_] @ self.bind_world[i]
            for t_, s_ in self.extra.items():
                W[self.idx[t_]] = A[s_] @ self.bind_world[self.idx[t_]]
            for i in self.unmapped_idx:
                p = self.parents[i]
                W[i] = W[p] @ (np.linalg.inv(self.bind_world[p]) @ self.bind_world[i])
            gf = min(f, gun_motion.length)
            gunworld = self.gun.world_at(gun_motion, gf)
            for name in self.gun_names:
                W[self.gun_idx[name]] = apply_bone(self.C, gunworld[name])
            loc = {}
            for i in range(self.total_bones):
                p = self.out_parent[i]
                loc[i] = W[i] if p < 0 else np.linalg.inv(W[p]) @ W[i]
            out.append({'f': f, 'W': W, 'local': loc})
        return out

    def skin_arms(self, W):
        v = np.zeros_like(self.arm_bind_verts)
        n = np.zeros_like(self.arm_bind_normals)
        for vi in range(len(v)):
            acc = np.zeros(3)
            accn = np.zeros(3)
            tot = 0.0
            for b, w in self.arms['weights'][vi]:
                M = W[b] @ np.linalg.inv(self.bind_world[b])
                acc += w * (M[:3, :3] @ self.arm_bind_verts[vi] + M[:3, 3])
                accn += w * (M[:3, :3] @ self.arm_bind_normals[vi])
                tot += w
            v[vi] = acc / tot
            n[vi] = accn
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        np.divide(n, ln, out=n, where=ln > 1e-12)
        return v, n

    def skin_gun(self, W):
        out = []
        for blk in self.gun_blocks:
            M = W[blk['bone']] @ np.linalg.inv(self.bind_world[blk['bone']])
            v = (M[:3, :3] @ blk['verts'].T).T + M[:3, 3]
            n = (M[:3, :3] @ blk['normals'].T).T
            out.append({'verts': v, 'normals': n})
        return out

    # ---------------------------------------------------------------- SMD writing
    def write_smd(self, path, frames, note=''):
        """frames: list of {local: {bone: 4x4}}, mesh = bind geometry (materials/textures)"""
        L = []
        L.append('// %s' % note if note else '// X-Ray EFT TT-33 DUO -> GMod viewmodel')
        L.append('version 1')
        L.append('nodes')
        for i in range(len(self.nodes)):
            L.append('%d "%s" %d' % (i, self.nodes[i][0], self.parents[i]))
        for n in self.gun_names:
            L.append('%d "%s" %d' % (self.gun_idx[n], n, self.out_parent[self.gun_idx[n]]))
        L.append('end')
        L.append('skeleton')
        L.append('time 0')
        for fi, fr in enumerate(frames):
            blocks = [fr] if fi == 0 else []
            if fi == 0:
                # reference (bind) pose, then first animation frame
                for i in range(self.total_bones):
                    m = self.bind_local[i]
                    rx, ry, rz = smd.from_matrix_zyx_angles(m[:3, :3])
                    p = m[:3, 3]
                    L.append('%d %.6f %.6f %.6f %.6f %.6f %.6f' % (i, p[0], p[1], p[2], rx, ry, rz))
        L.append('end')
        for k, fr in enumerate(frames):
            L.append('time %d' % k)
            for i in range(self.total_bones):
                m = fr[i]
                rx, ry, rz = smd.from_matrix_zyx_angles(m[:3, :3])
                p = m[:3, 3]
                L.append('%d %.6f %.6f %.6f %.6f %.6f %.6f' % (i, p[0], p[1], p[2], rx, ry, rz))
        L.append('triangles')
        # arms (two materials, provided by the source SMD)
        for matname in self.arms['materials']:
            fsel = [k for k, m in enumerate(self.arms['vert_material']) if m == matname]
            if not fsel:
                continue
            L.append(matname)
            emitted = set()
            for face_i, tri in enumerate(self.arms['faces']):
                if self.arms['mat_per_face'][face_i] != matname:
                    continue
                for vi in tri:
                    nv = self.arm_bind_normals[vi]
                    u, vv = self.arms['uvs'][vi]
                    ln = '%d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f' % (
                        self.arms['parent_bones'][vi], self.arm_bind_verts[vi][0], self.arm_bind_verts[vi][1],
                        self.arm_bind_verts[vi][2], nv[0], nv[1], nv[2], u, vv)
                    links = self.arms['weights'][vi]
                    ln += ' %d' % len(links)
                    for b, w in links:
                        ln += ' %d %.6f' % (b, w)
                    L.append(ln)
        # weapon
        for blk in self.gun_blocks:
            L.append(blk['material'])
            v, n, uv = blk['verts'], blk['normals'], blk['uvs']
            for tri in blk['tris']:
                for vi in tri:
                    ln = '%d %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f' % (
                        blk['bone'], v[vi][0], v[vi][1], v[vi][2], n[vi][0], n[vi][1], n[vi][2], uv[vi][0], uv[vi][1])
                    ln += ' 1 %d 1.000000' % blk['bone']
                    L.append(ln)
        L.append('end')
        open(path, 'w').write('\n'.join(L) + '\n')
        return len(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--motions', nargs='*')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    b = Builder()
    print(json.dumps(b.report['fit'], indent=1))

    clips = list(CLIPS) + (list(EXTRA_CLIPS) if args.all else []) + \
            [(c, 'eft_tt33_new_shoot_hip', c.replace('eft_tt33_new_', '')) for c in SHOT_CLIPS if args.all]
    if args.motions:
        want = set(args.motions)
        clips = [(h, g, o) for (h, g, o) in clips if o in want]
    for hname, gname, oname in clips:
        if hname not in b.hands.motions:
            print('skip (no hands motion) %s' % hname)
            continue
        hands_m = b.hands.motions[hname]
        if gname not in b.gun.motions:
            gname = 'idle'
        gun_m = b.gun.motions[gname]
        frames = b.frames(hands_m, gun_m, args.stride)
        path = os.path.join(OUT, 'tt33_viewmodel_%s.smd' % oname)
        lines = b.write_smd(path, [fr['local'] for fr in frames],
                            note='EFT TT-33 DUO viewmodel, hands clip %s, weapon clip %s, %d frames' %
                                 (hname, gname, len(frames)))
        print('wrote %-52s frames=%d lines=%d' % (os.path.basename(path), len(frames), lines))
        if args.check:
            fr = frames[min(3, len(frames) - 1)]
            v, n = b.skin_arms(fr['W'])
            g = b.skin_gun(fr['W'])
            print('   check: arms verts finite=%s  gun blocks=%d' % (np.isfinite(v).all(), len(g)))


if __name__ == '__main__':
    main()
