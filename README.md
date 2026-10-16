# EFT TT-33 DUO → GMod viewmodel retarget

## Result

This package contains a weapon-only GMod/Source reference SMD plus 17 skeleton-only animation SMDs. The animation skeleton has **59 bones**: the 46 nodes from the supplied GMod `c_arms_citizen.smd` (including `ValveBiped.*` names) and 13 original TT-33 weapon bones. The weapon root is parented to `ValveBiped.Bip01_R_Hand`; the left/right hand and finger tracks are retargeted to the matching ValveBiped names.

The reference SMD intentionally contains **only the weapon mesh**. GMod's player hands should remain separate and be drawn with `SWEP.UseHands = true`, as recommended by the GMod viewmodel-hands documentation. The software proof render includes the source arms mesh only to visualize the retarget.

## Main files

- `tt33_viewmodel_reference.smd` — 59-node reference skeleton + 11 weapon mesh parts (9,136 source vertices, 10,372 triangles).
- `tt33_anim_*.smd` — idle, aim, draw, holster, reload, empty reload, fire, walk, sprint, magcheck, inspect, and six shot variants.
- `c_eft_tt33.qc` — starter QC with standard GMod hands bone definitions and sequence declarations.
- `c_arms_definebones.qci` — bone-preservation include from the official GMod medkit example.
- `SWEP_viewmodel_snippet.lua` — minimal `SWEP.ViewModel` / `SWEP.UseHands` setup.
- `materials/models/weapons/c_eft_tt33/*.vmt` — temporary debug-white material placeholders, not the original EFT textures.
- `bone_map.csv`, `build_report.json` — mapping and parser/export report.
- `tt33_gmod_arms_proof.mp4` and PNG stills — software-rendered visual proof; **not** a GMod game capture.
- `source_reproducer/` — corrected retarget scripts and the minimum input files needed to rerun the conversion.

## Validation performed here

- Parsed the supplied archive's actual X-Ray formats: hand OMF bank (19 motions), jam OMF bank (2 motions), weapon OMF bank (18 motions), and HUD OGF (11 mesh parts).
- Verified the retarget fit against all four upper/lower arm segments; estimated scale is 43.164 Source units per source-rig unit.
- Generated reference and animation SMDs, then parsed them back: bone names and parents resolve, weight references are in range, frame indices are present, and geometry/animation values are finite.
- At idle frame 0, the exported pose matches the reference bind matrices to numerical precision. The build's skinning checks report finite vertices for every exported clip.

**Not verified:** This environment has no Blender, Garry's Mod, Crowbar, or `studiomdl`, so I could not compile the QC or test it in-game. Treat the QC and `$origin 0 0 66` as a good starting point, not an in-game certification. Replace the debug-white VMTs with converted Source VTF materials from the archive's DDS textures before release.

## Compile/use

1. Keep the QC, QCI, all SMDs, and the `materials/` folder in the corresponding compile project paths.
2. Use Crowbar or Garry's Mod `studiomdl` to compile `c_eft_tt33.qc`.
3. Put the compiled model at `models/weapons/c_eft_tt33.mdl` and use the Lua snippet. Adjust `$origin 0 0 66` after viewing it in-game if needed.
4. Convert the original `.dds` textures to `.vtf` and replace the temporary VMT basetextures.

## Format/tool note

The archives contain **`.omf` animation banks and `.ogf` models**, not `.ogz` files. The old `PavelBlend/blender_stalker_tools_addon` repository is archived and its README recommends the newer `PavelBlend/blender-xray` add-on; that add-on lists `.ogf` and `.omf` import support. This conversion was tested with a local parser so it did not require Blender.

## What was incomplete in the supplied working repo

The downloaded `effective-enigma` tree was not runnable as-is: it omitted `smd.py`, pointed to a missing `ee/` subfolder, misread the OGF vertex records as float64, mishandled version-4 OMF motion marks, and wrote malformed SMD section/material records. The corrected scripts are in `source_reproducer/`.

## References

- [GMod viewmodel hands](https://wiki.facepunch.com/gmod/Using_Viewmodel_Hands)
- [PavelBlend/blender-xray](https://github.com/PavelBlend/blender-xray)
- [Original conversion scripts](https://github.com/FeedMann/effective-enigma)
- [Original TT-33 DUO archive](https://github.com/FeedMann/glowing-octo-memory)
