"""Render a short structural proof of the EFT TT-33 -> GMod retarget.

This is a deterministic software render of the parsed OGF/OMF/SMD data, not a
capture from Garry's Mod or a StudioMDL compile.  It is useful to inspect the
retargeted hand pose and motion before compiling the SMDs in Source.
"""
import argparse
import os
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import build_viewmodel as BV
import render as R

ARM_COLORS = {
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
    def __init__(self, builder):
        self.b = builder
        self.arm_faces_sel = {
            material: np.array([m == material for m in builder.arms['mat_per_face']], dtype=bool)
            for material in builder.arms['materials']
        }
        all_gun = np.vstack([block['verts'] for block in builder.gun_blocks])
        self.gun_center = all_gun.mean(axis=0)
        self.arms_center = builder.arm_bind_verts.mean(axis=0)
        # A stable 3/4 side view keeps the pistol profile visible and remains
        # framed while the hands move through draw/reload.
        self.target = 0.72 * self.gun_center + 0.28 * self.arms_center

    def meshes(self, world):
        arms_v, _ = self.b.skin_arms(world)
        arm_groups = [
            (arms_v, self.b.arms['faces'], ARM_COLORS.get(name, (0.8, 0.8, 0.8)),
             self.arm_faces_sel[name])
            for name in self.b.arms['materials']
        ]
        gun_groups = []
        for block, skinned in zip(self.b.gun_blocks, self.b.skin_gun(world)):
            gun_groups.append((skinned['verts'], block['tris'],
                               GUN_COLORS.get(block['material'], GUN_FALLBACK)))
        return arm_groups, gun_groups

    def camera(self, kind='proof'):
        if kind in ('proof', 'side', 'closeup'):
            if kind == 'closeup':
                offset = np.array([26.0, 4.0, 11.0])
            elif kind == 'side':
                offset = np.array([34.0, 5.0, 10.0])
            else:
                offset = np.array([30.0, 4.0, 12.0])
            eye = self.target + offset
            return R.look_at(eye, self.target, up=(0.0, 0.0, 1.0))
        if kind == 'top':
            eye = self.target + np.array([0.0, 0.0, 42.0])
            return R.look_at(eye, self.target, up=(-1.0, -0.15, 0.0))
        raise ValueError(kind)

    def draw(self, world, path, width=720, height=450, kind='proof', dots=False):
        scene = R.Scene(width, height, fov_deg=38.0 if kind != 'closeup' else 32.0)
        scene.clear((0.035, 0.043, 0.055))
        view = self.camera(kind)
        arm_groups, gun_groups = self.meshes(world)
        for verts, tris, color, select in arm_groups:
            scene.add_mesh(verts, tris, view, color, only_tris=select)
        for verts, tris, color in gun_groups:
            scene.add_mesh(verts, tris, view, color)
        if dots:
            origins = np.array([world[i][:3, 3] for i in range(self.b.total_bones)])
            scene.add_points(origins, view, color=(1.0, 0.12, 0.12), radius=2)
        Image.fromarray((scene.image() * 255).astype(np.uint8)).save(path)
        return path


def _fonts():
    try:
        bold = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 18)
        regular = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
        small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
        return bold, regular, small
    except Exception:
        default = ImageFont.load_default()
        return default, default, default


def overlay(path, lines, title='TT-33 DUO  |  X-Ray OMF to GMod ValveBiped'):
    image = Image.open(path).convert('RGB')
    draw = ImageDraw.Draw(image)
    title_font, text_font, small_font = _fonts()
    draw.rectangle((0, 0, image.width, 36), fill=(12, 16, 22))
    draw.text((12, 8), title, fill=(246, 223, 154), font=title_font)
    line_height = 18
    y = image.height - (line_height * len(lines) + 10)
    draw.rectangle((0, y - 6, image.width, image.height), fill=(12, 16, 22))
    for i, line in enumerate(lines):
        draw.text((12, y + line_height * i), line, fill=(219, 225, 232),
                  font=small_font if len(line) > 95 else text_font)
    image.save(path)
    return path


def encode_sequence(frame_dir, count, out_path, fps):
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    pattern = os.path.join(frame_dir, 'frame_%06d.png')
    cmd = [ffmpeg, '-y', '-v', 'error', '-framerate', str(fps), '-i', pattern,
           '-frames:v', str(count), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
           '-crf', '20', '-movflags', '+faststart', out_path]
    import subprocess
    subprocess.check_call(cmd)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=os.path.join(ROOT, 'proof'))
    parser.add_argument('--frames-dir', default=None)
    parser.add_argument('--width', type=int, default=720)
    parser.add_argument('--height', type=int, default=450)
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--stride', type=int, default=2,
                        help='sample every N source animation frames; keep output timing at --fps')
    args = parser.parse_args()
    if args.fps <= 0 or args.stride <= 0:
        parser.error('--fps and --stride must be positive')

    os.makedirs(args.out, exist_ok=True)
    frames_dir = args.frames_dir or os.path.join(args.out, '_frames')
    os.makedirs(frames_dir, exist_ok=True)
    builder = BV.Builder()
    proof = Proof(builder)
    bind_world = {i: builder.bind_world[i] for i in range(builder.total_bones)}

    static = os.path.join(args.out, 'tt33_gmod_retarget_hold.png')
    raw = os.path.join(args.out, '_hold_raw.png')
    proof.draw(bind_world, raw, args.width, args.height, kind='proof')
    overlay(raw, [
        'Reference pose: EFT TT-33 idle frame 0; weapon-only reference SMD, standard GMod hands remain separate.',
        'Rig: 46 c_arms nodes (ValveBiped names) + 13 weapon bones; source mesh is used only in this preview.',
        'Software render from parsed SMD/OGF/OMF data — not a Garry\'s Mod game capture.'
    ])
    os.replace(raw, static)

    bones = os.path.join(args.out, 'tt33_gmod_retarget_bones.png')
    raw = os.path.join(args.out, '_bones_raw.png')
    proof.draw(bind_world, raw, args.width, args.height, kind='proof', dots=True)
    overlay(raw, [
        'Red dots mark the 59 exported bone origins; names and hierarchy are in the SMD files.',
        'GMod arms use ValveBiped.* names; the 13 original X-Ray weapon bones are parented under the right hand.',
        'Software render — not an in-game capture.'
    ], title='TT-33 DUO  |  exported GMod bone layout')
    os.replace(raw, bones)

    clips = [
        ('eft_tt33_new_idle', 'idle', 'IDLE'),
        ('eft_tt33_new_reload_nobump', 'reload_nobump', 'RELOAD'),
        ('eft_tt33_new_draw', 'idle', 'DRAW'),
    ]
    frame_paths = []
    frame_index = 0
    for hands_name, gun_name, label in clips:
        hands_motion = builder.hands.motions[hands_name]
        gun_motion = builder.gun.motions[gun_name]
        clip_frames = builder.frames(hands_motion, gun_motion, args.stride)
        for frame in clip_frames:
            path = os.path.join(frames_dir, 'frame_%06d.png' % frame_index)
            proof.draw(frame['W'], path, args.width, args.height, kind='proof')
            overlay(path, [
                'Clip: %s   |   source frame %d / %d   |   hands: %s' %
                (label, frame['f'], hands_motion.length - 1, hands_name),
                'Weapon: %s   |   %d ValveBiped hand nodes + %d weapon bones   |   software preview, not GMod capture' %
                (gun_name, len(builder.nodes), len(builder.gun_names))
            ])
            frame_paths.append(path)
            frame_index += 1
        print('%s: %d rendered frames (source length %d)' %
              (label, len(clip_frames), hands_motion.length))

    video = os.path.join(args.out, 'tt33_gmod_arms_proof.mp4')
    encode_sequence(frames_dir, frame_index, video, args.fps)
    print('hold still:', static)
    print('bone still:', bones)
    print('video:', video, 'frames=%d fps=%d duration=%.2fs' %
          (frame_index, args.fps, frame_index / float(args.fps)))
    # Keep only the requested deliverables, not hundreds of intermediate PNGs.
    shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
