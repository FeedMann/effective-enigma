# EFT "TT-33 DUO" → Garry's Mod viewmodel hands

Retarget of a S.T.A.L.K.E.R. Anomaly weapon mod's **HUD animations** onto GMod's
`ValveBiped.*` hand bones, and export as Source engine assets (SMD + QC + VMT)
that can be compiled with studiomdl / Crowbar.

Source mod: `Escape From Tarkov TT-33 DUO` (Anomaly) — EFT-ported weapon with
its own hands/HUD motion bank.
Target: `c_arms_citizen.smd` (GMod SDK arms, 46 `ValveBiped.*` nodes) + the
TT-33's 13 bones parented to `ValveBiped.Bip01_R_Hand` — 59 nodes total.

---

## 1. Result: is this drag-and-drop or manual?

**For mods authored the same way (EFT-ported weapons for Anomaly with a HUD
motion bank + `wpn_hand`-style hands): drag-and-drop.** You change the 5 file
paths at the top of `build_gmod.py`, run two commands, and you get the whole
asset set. Nothing is hand-keyed, no bone is assigned by hand:

```
python3 build_gmod.py            # reference SMD + 27 animation SMDs + QC + VMT + report
blender -b -P blender/render_proof.py -- --camera pov --out out/proof_pov.mp4
```

What is derived **automatically** from the files (no manual input):

| what | how |
|---|---|
| bone mapping | naming rules on the Anomaly HUD-hand bones (`l_/r_clavicle…hand`, `bip01_*_finger*`, `lead_gun`) → `ValveBiped.Bip01_*` |
| unit scale | measured from the rigid arm segments of both rigs (43.09 units / source unit, spread ±0.6) |
| axis / frame convention | OMF quaternion+translation storage, OGF bind chain, SMD euler — each pinned against an independent artifact and re-checked in the build report |
| weapon rest pose | OGF bind chain, root placed on the hands' `lead_gun` anchor (exactly where EFT's motion file puts `wpn_body`) |
| every frame of every motion | converted, verified, written to its own SMD (27 sequences) |
| QC, materials, bone map, report | generated |

What is **not** automatic (would need work / is honest limitation):

* a weapon whose rig is a **different family** (vanilla SoC/CS/CoP weapons,
  `wpn_*` models without the Anomaly HUD hands naming) — the bone-mapping
  regexes and the `lead_gun`/`bip01` anchors assume the EFT-ported layout;
* motions that are **not** HUD/viewmodel motions (world/3rd-person anims have a
  different skeleton);
* bones that do not exist in GMod's arm rig: the EFT thumb-base roll, forearm
  twist (mapped onto `Ulna`), and the `lead_gun` anchor. The GMod nodes
  `Wrist`, `Neck1`, `Head1` and the mesh node are static (no source data).
* the **hands model itself** is GMod's own citizen arms — only the *animation*
  and the weapon come from the mod. (Swapping in the mod's own hands mesh is a
  separate job: those are Anomaly's `wpn_hand_*.ogf`, a different skeleton.)

So: this specific mod = automatic. A random STALKER weapon = automatic **if**
it is the same authoring family, otherwise expect to adapt the mapping table
(an afternoon, not weeks), because everything geometric is measured, not typed.

---

## 2. Verification (build_report.json has all numbers)

Every check is re-run at build time on the actual data:

| check | result |
|---|---|
| retargeted bone positions vs the source pose (all frames, all 41 bones) | **0.0 units** error |
| weapon bone positions vs the source pose (13 bones, all frames) | **0.0 units** |
| finger ↔ gun distances preserved (this couples both chains) | **0.0 units** |
| written SMD locals rebuilt into world matrices | **0.0 units** |
| reference SMD vs frame 0 of the idle sequence | **0.0** translation/rotation delta |
| SMD euler convention vs `c_arms_definebones.qci` (38 bones) | max element err **0.0016** (wrong axis assignment: 1.96) |
| OGF ik bind vs its own weapon mesh (10 landmark bones) | mean **3.3 mm** |
| EFT idle fingertip → gun surface | mean **1.4 cm** (i.e. inside the grip) |
| unit scale | **43.09** (eff-enigma's README claims 43.164 — same ballpark) |

Meaning: the exported pose is the source pose under one global similarity
(rotation + uniform scale), so the gun keeps its exact relation to the gripping
fingers frame by frame. That is the "without error" part.

**Not verified:** in-engine. There is no GMod / studiomdl / Crowbar in this
sandbox, so `$sequence` playback, `$includemodel`, `$proceduralbones` and the
actual skinning in-engine were **not** tested. The proof video is rendered in
Blender from the exported SMD data (same hours, same numbers).

---

## 3. Files

```
eft2gmod/                  `eft2gmod/` python package: lzhuf, binrw, ogf, omf readers
  build_gmod.py            the converter (all paths/config at the top)
  blender/render_proof.py  Blender proof renderer (reads the SMDs it exports)
  out/
    c_eft_tt33_reference.smd      59-bone, 46 782-vert reference model
    c_eft_tt33_anim_*.smd         27 animation sequences
    c_eft_tt33.qc                 QC with $body, $attachment, 27 $sequence
    materials/models/weapons/eft_tt33/*.vmt
    bone_map.csv                  source bone → ValveBiped bone (with lengths)
    build_report.json             every number above
    proof_pov.mp4                 proof reel, GMod viewmodel POV
    proof_side.mp4                proof reel, 3/4 view
    stills/*.png                  one frame per montage clip
  analysis/                 the format/convention probes used to pin the math
```

## 4. Compiling for GMod

1. put `c_eft_tt33_reference.smd`, the 27 `c_eft_tt33_anim_*.smd` and
   `c_eft_tt33.qc` in one folder, with `c_arms_definebones.qci` (GMod SDK)
   next to them;
2. `studiomdl c_eft_tt33.qc` (or drag the QC into Crowbar);
3. copy `materials/models/weapons/eft_tt33/*.vmt` + the mod's DDS textures into
   `garrysmod/materials/models/weapons/eft_tt33/`;
4. the model already includes GMod's `c_arms_animations.mdl` sequences? No —
   add `$includemodel weapons/c_arms_animations.mdl` if you want the shared
   arm layers; the sequences below are self-contained.
5. hook up in a SWEP with `self.ViewModelFOV = 54`, and the sequences as
   `ACT_VM_*` (already set in the QC for the ones that exist).

## 5. Sequences

draw, holster, idle, idle_empty, idle_jammed, aim, inspect, inspect_jammed,
magcheck(+empty/jammed/jammed_empty), reload_nobump, reload_empty_nobump,
shoot_hip(1,2), shoot_aim(1,2), shoot_empty, unjam, unjam_empty, walk,
walk2sprint, sprint, sprint2walk — 27 total, 30/60 fps, ~1900 frames.

Known imperfections (not hidden): the retargeted arms are the *citizen* arms, so
proportions differ slightly from the mod's hands; the wrist / ulna / thumb-base
degrees of freedom have no ValveBiped equivalent, so the wrist is held; no
in-engine pass was possible here.
