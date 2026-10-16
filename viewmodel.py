"""Build a GMod/Source viewmodel from X-Ray (STALKER/Anomaly) EFT TT-33 hands+weapon data.

Pipeline
--------
1. source hands rig   <- hands OMF (42 bones, corrected quaternion convention)
2. source weapon rig  <- weapon OMF (13 bones) + weapon OGF mesh (model space)
3. similarity C       <- source rig space -> target (GMod arms SMD) space
4. re-bind            <- GMod arms mesh is posed into the source's hold pose and becomes the new bind
5. retarget           <- per-bone local deltas, conjugated into the target bone frames
6. export             <- compilable SMD (bind + animated frames) with ValveBiped bone names
"""
import numpy as np
import smd
import rig
import xray
import ogf as ogfmod

HANDS_HP = {
    'bip01': None,
    'l_clavicle': 'bip01', 'l_upperarm': 'l_clavicle', 'l_forearm': 'l_upperarm',
    'l_forearm_twist': 'l_forearm', 'l_hand': 'l_forearm', 'lead_gun': 'bip01',
    'r_clavicle': 'bip01', 'r_upperarm': 'r_clavicle', 'r_forearm': 'r_upperarm',
    'r_forearm_twist': 'r_forearm', 'r_hand': 'r_forearm',
}
for _side in ('l', 'r'):
    for _f in '1234':
        HANDS_HP['bip01_%s_finger%s' % (_side, _f)] = '%s_hand' % _side
        HANDS_HP['bip01_%s_finger%s1' % (_side, _f)] = 'bip01_%s_finger%s' % (_side, _f)
        HANDS_HP['bip01_%s_finger%s2' % (_side, _f)] = 'bip01_%s_finger%s1' % (_side, _f)
    HANDS_HP['%s_finger0' % _side] = '%s_hand' % _side
    HANDS_HP['%s_finger01' % _side] = '%s_finger0' % _side
    HANDS_HP['%s_finger02' % _side] = '%s_finger01' % _side

# X-Ray hands bone -> ValveBiped bone name in the GMod arms SMD
BONE_MAP = {
    'bip01': 'ValveBiped.Bip01_Spine4',
    'l_clavicle': 'ValveBiped.Bip01_L_Clavicle',
    'l_upperarm': 'ValveBiped.Bip01_L_UpperArm',
    'l_forearm': 'ValveBiped.Bip01_L_Forearm',
    'l_forearm_twist': 'ValveBiped.Bip01_L_Wrist',
    'l_hand': 'ValveBiped.Bip01_L_Hand',
    'l_finger0': 'ValveBiped.Bip01_L_Finger0',
    'l_finger01': 'ValveBiped.Bip01_L_Finger01',
    'l_finger02': 'ValveBiped.Bip01_L_Finger02',
    'r_clavicle': 'ValveBiped.Bip01_R_Clavicle',
    'r_upperarm': 'ValveBiped.Bip01_R_UpperArm',
    'r_forearm': 'ValveBiped.Bip01_R_Forearm',
    'r_forearm_twist': 'ValveBiped.Bip01_R_Wrist',
    'r_hand': 'ValveBiped.Bip01_R_Hand',
    'r_finger0': 'ValveBiped.Bip01_R_Finger0',
    'r_finger01': 'ValveBiped.Bip01_R_Finger01',
    'r_finger02': 'ValveBiped.Bip01_R_Finger02',
}
# the X-Ray forearm-twist bone also drives the Source ulna helper
EXTRA_MAP = {'l_forearm_twist': ['ValveBiped.Bip01_L_Ulna'],
             'r_forearm_twist': ['ValveBiped.Bip01_R_Ulna']}
for _side in ('l', 'r'):
    for _f in '1234':
        BONE_MAP['bip01_%s_finger%s' % (_side, _f)] = 'ValveBiped.Bip01_%s_Finger%s' % (_side.upper(), _f)
        BONE_MAP['bip01_%s_finger%s1' % (_side, _f)] = 'ValveBiped.Bip01_%s_Finger%s1' % (_side.upper(), _f)
        BONE_MAP['bip01_%s_finger%s2' % (_side, _f)] = 'ValveBiped.Bip01_%s_Finger%s2' % (_side.upper(), _f)


def rot_mat(axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def frame_matrix(forward, up):
    """Rotation whose +X is 'forward' and +Z is 'up' (Source/Blender style basis)."""
    f = np.asarray(forward, dtype=np.float64)
    f = f / np.linalg.norm(f)
    u = np.asarray(up, dtype=np.float64)
    left = np.cross(u, f)
    left /= np.linalg.norm(left)
    u2 = np.cross(f, left)
    return np.stack([f, left, u2], axis=1)   # columns = basis vectors


class Placement:
    """similarity: source X-Ray rig space -> target SMD space"""

    def __init__(self, src_fwd, src_left, tgt_fwd, tgt_left, scale, translation):
        Bs = frame_matrix(src_fwd, np.cross(src_left, src_fwd))
        Bt = frame_matrix(tgt_fwd, np.cross(tgt_left, tgt_fwd))
        self.R = Bt @ Bs.T
        self.scale = scale
        self.t = np.asarray(translation, dtype=np.float64)

    def apply(self, pts):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        return (self.scale * (self.R @ pts.T)).T + self.t

    def apply_mat(self, M):
        out = np.eye(4)
        out[:3, :3] = self.scale * (self.R @ M[:3, :3])
        out[:3, 3] = self.scale * (self.R @ M[:3, 3]) + self.t
        return out

    def apply_vec(self, v):
        return self.scale * (self.R @ np.asarray(v, dtype=np.float64))
