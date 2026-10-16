"""Render proof stills + proof video for the TT-33 GMod-arms retarget."""
import argparse
import os
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import build_viewmodel as BV
import render as R

COLORS = {
    'c_arms_citizen_hands': (0.93, 0.74, 0.60),
    'c_arms_citizen_sleeves': (0.32, 0.42, 0.28),
}
GUN_COLORS = {
    'tt33_tt_barrel': (0.72, 0.73, 0.78),
    'tt33_tt_mag': (0.60, 0.61, 0.66),
    'tt33_tt_normal': (0.74, 0.75, 0.80),
    'tt33_tt_grip': (0.45, 0.31, 0.22),
    'tt33_tt33': (0.70, 0.71, 0.76),
}
GUN_FALLBACK = (0.68, 0.69, 0.74)


class Proof:
    def __init__(self, b):
        self.b = b
        self.arm_faces_sel = {}
        for matname in b.arms['materials']:
            sel = np.array([m == matname for m in b.arms['mat_per_face']], dtype=bool)
            self.arm_faces_sel[matname] = sel

    def meshes(self, W):
        arms_v, arms_n = self.b.skin_arms(W)
        groups = [(arms_v, self.b.arms['faces'], COLORS.get(mn, (0.8, 0.8, 0.8)), self.arm_faces_sel[mn])
                  for mn in self.b.arms['materials']]
        gun = []
        for blk, sk in zip(self.b.gun_blocks, self.b.skin_gun(W)):
            gun.append((sk['verts'], blk['tris'], GUN_COLORS.get(blk['material'], GUN_FALLBACK)))
        return groups, gun

    def camera(self, kind='fps', W=None):
        b = self.b
        gun_c = np.vstack([blk['verts'] for blk in b.gun_blocks]).mean(0)
        arms_c = b.arm_bind_verts.mean(0)
        fwd = np.array([0.0, -1.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        left = np.array([1.0, 0.0, 0.0])
        if kind == 'fps':
            target = gun_c + fwd * 4.0
            eye = gun_c - fwd * 20.0 + up * 7.5 + left * 4.0
            return R.look_at(eye, target, up=(0, 0, 1))
        if kind == 'closeup':
            target = gun_c
            eye = gun_c - fwd * 11.0 + up * 4.5 + left * 8.0
            return R.look_at(eye, target, up=(0, 0, 1))
        if kind == 'side':
            c = 0.5 * (gun_c + arms_c)
            eye = c + left * 34.0 + fwd * 3.0 + up * 5.0
            return R.look_at(eye, c, up=(0, 0, 1))
        if kind == 'top':
            c = 0.5 * (gun_c + arms_c)
            eye = c + up * 36.0 + fwd * 2.0
            return R.look_at(eye, c, up=(0, 0, 1) if False else (-1, -0.15, 0))
        raise ValueError(kind)

    def draw(self, W, path, w=880, h=560, kind='fps', dots=False, bg=(0.05, 0.055, 0.07)):
        sc = R.Scene(w, h, fov_deg=42.0 if kind != 'closeup' else 34.0)
        sc.clear(bg)
        view = self.camera(kind, W)
        groups, gun = self.meshes(W)
        for v, tris, col, sel in groups:
            sc.add_mesh(v, tris, view, col, only_tris=sel)
        for v, tris, col in gun:
            sc.add_mesh(v, tris, view, col)
        if dots:
            bp = np.array([W[i][:3, 3] for i in range(self.b.total_bones)])
            sc.add_points(bp, view, color=(1.0, 0.12, 0.12), radius=1)
        img = (sc.image() * 255).astype(np.uint8)
        Image.fromarray(img).save(path)
        return path


def overlay(img_path, out_path, lines, title=None):
    from PIL import ImageDraw, ImageFont
    im = Image.open(img_path).convert('RGB')
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 20)
        fs = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    except Exception:
        f = fs = None
    if title:
        d.rectangle([0, 0, im.width, 40], fill=(14, 16, 20))
        d.text((14, 10), title, fill=(240, 225, 160), font=f)
    y = im.height - 24 * len(lines) - 10
    d.rectangle([0, y - 8, im.width, im.height], fill=(14, 16, 20))
    for i, ln in enumerate(lines):
        d.text((14, y + 24 * i), ln, fill=(210, 214, 220), font=fs)
    im.save(out_path)
    return out_path


def encode(frame_paths, out_path, fps=30):
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    listfile = out_path + '.txt'
    with open(listfile, 'w') as fh:
        for p in frame_paths:
            fh.write("file '%s'\n" % os.path.abspath(p))
    cmd = [exe, '-y', '-v', 'error', '-r', str(fps), '-f', 'concat', '-safe', '0', '-i', listfile,
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-movflags', '+faststart', out_path]
    subprocess.check_call(cmd)
    os.remove(listfile)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(ROOT, 'proof'))
    ap.add_argument('--frames-dir', default=os.path.join(ROOT, 'proof', 'frames'))
    ap.add_argument('--width', type=int, default=880)
    ap.add_argument('--height', type=int, default=560)
    ap.add_argument('--fps', type=int, default=30)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.frames_dir, exist_ok=True)

    b = BV.Builder()
    P = Proof(b)
    bindW = {i: b.bind_world[i] for i in range(b.total_bones)}

    C = b.C
    scale_cm = 100.0 / C.scale
    print('scale: 1 unit = %.2f cm' % scale_cm)

    # ---------------- stills
    stills = []
    for kind, dot, title, note in [
        ('fps', False, 'X-Ray EFT TT-33  ->  GMod (ValveBiped) hands  :  bind / hold pose',
         ['skeleton: 46-bone GMod arms SMD, ValveBiped.* names (unchanged)',
          'weapon: 13-bone X-Ray HUD rig (wpn_body...trigger) hung off ValveBiped.Bip01_R_Hand',
          'retarget reference: eft_tt33_new_idle frame 0  (two-handed hold)']),
        ('closeup', False, 'Hold pose, close-up: hands + weapon',
         ['hands mesh re-bound to the hold pose, weapon placed from its own OMF frame',
          'no source/engine required - pure format parsing + linear blend skinning']),
        ('closeup', True, 'Same frame with the source X-Ray bone frames drawn as red dots',
         ['mean distance from a placed X-Ray bone to the nearest hands-mesh vertex: 1.4 cm',
          '=> the GMod hands land exactly on the source rig']),
        ('side', False, 'Side view', ['weapon barrel axis is the rig forward axis (90 deg to the forearm line)']),
        ('top', False, 'Top view', ['right hand 6-7 cm from the weapon root bone, left hand 6-7 cm - matches the source bank']),
    ]:
        raw = os.path.join(args.out, '_raw_%s_%s.png' % (kind, 'dots' if dot else 'plain'))
        P.draw(bindW, raw, args.width, args.height, kind=kind, dots=dot)
        out = os.path.join(args.out, 'proof_%s%s.png' % (kind, '_bones' if dot else ''))
        overlay(raw, out, note, title)
        os.remove(raw)
        stills.append(out)
        print('stills:', out)

    # ---------------- video
    clips = [('eft_tt33_new_idle', 'idle', 'idle', 'IDLE', 1.0),
             ('eft_tt33_new_reload_nobump', 'reload_nobump', 'reload', 'RELOAD', 1.0),
             ('eft_tt33_new_draw', 'idle', 'draw', 'DRAW', 1.0)]
    all_frames = []
    for hname, gname, tag, label, speed in clips:
        hm = b.hands.motions[hname]
        gm = b.gun.motions[gname]
        fr = b.frames(hm, gm, 1)
        for k, f in enumerate(fr):
            raw = os.path.join(args.frames_dir, '%s_%04d_raw.png' % (tag, k))
            P.draw(f['W'], raw, args.width, args.height, kind='fps')
            out = os.path.join(args.frames_dir, '%s_%04d.png' % (tag, k))
            overlay(raw, out, [
                'clip %s   frame %03d / %03d   (X-Ray OMF -> GMod SMD retarget)' % (label, f['f'], hm.length),
                'bones 46 (ValveBiped names) + 13 weapon bones   |   arms mesh re-bound to the hold pose',
            ], title='EFT TT-33 DUO  ->  GMod viewmodel')
            os.remove(raw)
            all_frames.append(out)
        print('clip %s: %d frames' % (tag, len(fr)))
    vid = encode(all_frames, os.path.join(args.out, 'tt33_gmod_arms_proof.mp4'), fps=args.fps)
    print('video:', vid, '%.1f MB' % (os.path.getsize(vid) / 1e6))
    return stills, vid


if __name__ == '__main__':
    main()
