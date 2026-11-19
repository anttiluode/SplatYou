# SplatYou — the wave decoder meets the physical world

**SplatWorld's real 7.2 MB ONNX face decoder + an immutable Sihti-style residue anchor + a measured optical inverse + a camera/monitor feedback loop.** Runs locally on Windows with CPU `onnxruntime` and OpenCV; no local server, no PyTorch, no cloud GPU.

This is a **new integration / experimental instrument**, not a newly trained model. All three modes use the *same unchanged* `splat_decoder.onnx`. Our numerical tests and synthetic inverse pass; **actual S20/diffuser validation is still required**. A nice simulated face is not evidence that scattering has been inverted.

## Why this version, rather than SplatWorld 5 by name?

- [SplatWorld](https://github.com/anttiluode/SplatWorld) supplied the 128-D ONNX decoder (96px RGB Gabor field; the original CelebA training run).
- [SplatWorld2](https://github.com/anttiluode/SplatWorld2) supplied *identity-anchored transport*: warp detail from an immutable anchor, never feed reconstructed detail back into itself.
- [SplatWorld3](https://github.com/anttiluode/SplatWorld3) explored an address-conditioned operator plate and showed why the outer 'fire' can also just be out-of-chart extrapolation.
- [SplatWorld4](https://github.com/anttiluode/SplatWorld4) Gate 7 supplied the important **weak-direction active sensing** lesson. SplatYou asks which *screen mask*, rather than camera position, is predicted to expose weak directions of the face model.
- [Sihti](https://github.com/anttiluode/Sihti) showed that `image = core + sum(residues)` is an exact telescoping identity. Here we use that decomposition only for a frozen photographic anchor. **We have not imported or retrained SihtiImageModel2's P3 generator.** Its own experiments found that crisp stochastic texture does not simply emerge from least-squares patch memory.
- [LightScan](https://github.com/anttiluode/LightScan) supplied practical monitor/phone rig lessons: manual exposure, stable scene geometry, and a real camera-forward calibration gate.

## Install and run

```bat
cd /d E:\DocsHouse\SplatYou
git pull --ff-only
python3.13 -m pip install -r requirements.txt
python3.13 splatyou.py simulate
python3.13 -m unittest discover -s tests -v
```

Get the **authentic** decoder from your existing local `SplatWorld/splat_decoder.onnx`, or let the script download the pinned original 7.2 MB model one time:

```bat
python3.13 splatyou.py studio --download-model
```

It also checks adjacent `../SplatWorld`, `../SplatWorld2`, and `../SplatWorld3` folders, as well as `$SPLAT_MODEL` or `--model PATH`. **There is no invisible mock substitution:** if the real model is missing, physical and studio modes stop with an explicit error.

## Mode 1 — the missing second half: RESIDUE STUDIO

```bat
python3.13 splatyou.py studio
python3.13 splatyou.py studio --anchor E:\photos\your_high_resolution_face.jpg
```

The **left** panel is the unchanged 96x96 wave decoder, upscaled. The **middle** panel optionally transports fine frequency bands of a **user-supplied high-resolution anchor** using dense optical flow from the current decoded guide back to a frozen guide. The **right** panel shows the model's Sihti-style fine-band energy. Drag in the live panel to change the local latent; move the `radius` and `detail` sliders. SPACE or N chooses a new identity, A toggles anchor detail, S saves. After changing identity the anchor is disabled until you explicitly re-enable it, because pairing a photograph with an unrelated identity is unreliable.

**Critical:** if you have no high-resolution anchor, no new high-frequency facts can be conjured from a 96px face. The middle panel then deliberately equals the left one. Anchor transport is an effect, **not a learned super-resolution network, identity lock, or true face reenactment.**

## Mode 2 — face pulled through a *real* diffuser

Keep the Samsung and screen fixed, diffuser in place, manual ISO/shutter/WB/focus, room dim. Do not open an IP Webcam browser preview on the same monitor: it creates a separate optical feedback loop.

```bat
python3.13 splatyou.py inverse --camera http://192.168.1.157:8080 ^
  --zone 420 --patterns 120 --settle 0.8 --samples 2 --views 3 \
  --output runs\physical01
```

(Windows Command Prompt needs the whole command on one line, or `^` continuations rather than `\`.) The full-screen projection tells you when to press SPACE. The script:

1. Measures black and white camera references through the installed diffuser. Rejects weak contrast, finds the physical illuminated region.
2. Displays **120 known screen patterns**, measures their real S20 JPEG camera responses, and fits a linear ridge forward map of 8x8 screen intensities to 14x14 camera features. Holds out 16% of the patterns *before* fitting. Reports forward RMSE against a mean-image baseline. **If it loses the held-out gate, don't trust the inverse.**
3. Displays a new latent-generated face behind the diffuser, but passes **only camera readings** to the inverse routine. It searches authentic ONNX candidates and makes damped locally measured corrections in a small tangent chart. No target latent or target pixels enter the solver.
4. It actively chooses which screen **mask** to query next using the currently weak SVD directions, *without looking at the unseen measurements*. It receives three physical observations with different known masks.
5. Saves `report.json` with optical errors and an honest image-space comparison against a model-free regularized 8x8 reconstruction. `comparison.png` shows **target (evaluation only)** / face-prior reconstruction / pixel-only baseline. `last_camera.jpg` captures the last raw camera frame.

A successful fit of *camera measurements* does not prove a correctly recovered face: multiple face latents can look similar through strongly scattering material. The image-space comparison is the check. The objective is a **2D, prior-constrained inverse**, not 3D CT or x-ray vision.

**Attack the prior** with a non-face scene:

```bat
python3.13 splatyou.py inverse --target-image E:\photos\unknown_scene.jpg ^
  --calibration runs\physical01\calibration\forward.npz --output runs\out_of_family
```

If the fit claims a confident face for this unrelated picture, that's a failure of model-mismatch detection. One source of model mismatch is a non-linear/position-dependent diffuser or phone image processing; this version reports failures instead of hiding them.

## Mode 3 — semantic optical resonator

```bat
python3.13 splatyou.py feedback ^
  --calibration runs\physical01\calibration\forward.npz \
  --settle 0.15 --samples 1 --gain 1.0 --output runs\resonator
```

The decoder draws a face **onto the same monitor** the S20 sees. The program predicts the camera image from its **measured forward optical transfer** and compares it with the *actual* camera observation. That **innovation / residual**, not just screen brightness, nudges two local face dimensions and moves the artistic radius toward the fire under strong disturbance. As the discrepancy disappears the radius returns to the coherent core. Your hand or a glass in the path can perturb the loop. R resets the latent deviation and S saves the camera frame if the OpenCV window receives the key; Q or ESC exits.

The loop runs only as fast as the model, camera snapshots, Wi-Fi and `--settle` allow. **It does not promise 30 FPS** and may be unstable when the measured transfer changes. The radius is a **visual metaphor** for unsettled sensor residuals, not a physical phase transition or calibrated uncertainty. The camera-view model is linear and may underfit complex scattering.

## Measurements and gates

| Gate | Success | Refutation |
|---|---|---|
| Exact residue conservation | core + bands equals input numerically | bands leak/lose information |
| Forward calibration | held-out camera RMSE beats constant prediction | unstable rig or uninvertible feature family |
| Latent inverse | new face improves on pixel-only grid baseline in *image-space* error | solver fits glow but hallucinates a face |
| Active masks | weak-direction selection improves across several blind targets versus matched random masks | masks are decorative rather than informative |
| Out-of-family | mismatch identified for unrelated pictures/occluders | prior insists every scene is a face |
| Feedback | controlled obstruction produces reproducible residual-driven latent change | screen recursion / stale snapshots make only noise |

Only the numerical tests and fake-optics simulation are run and passed here. **Physical gates remain open until you run the Samsung experiment.** In particular the active-mask advantage over matched random masks has NOT been proven.

## Reproducibility

```bat
python3.13 splatyou.py simulate --output runs\simulation
python3.13 -m unittest discover -s tests -v
```

`simulate` uses a **synthetic toy decoder and synthetic linear Gaussian camera**, not real SplatWorld weights or any physical scattering measurement. It tests algorithms and dataflow; never quote its numbers as physical evidence. The original decoder is trained on facial photographs and is *not* capable of representing arbitrary external objects.

The optical calibration only applies while the camera, screen, diffuser, ISO, focus, orientation, JPEG settings and screen brightness remain unchanged. When one changes, recalibrate. Captured photos stay on your PC. Never use this tool for medical imaging.

## Files

- `splatyou.py`: executable studio / calibration / inverse / feedback / simulation UI.
- `splatyou/model.py`: authentic ONNX loader, latent charts and optional pinned-download.
- `splatyou/residue.py`: exact telescope + immutable-anchor frequency transport.
- `splatyou/optics.py`: real camera, projector, measured forward calibration, serialization.
- `splatyou/inverse.py`: camera-only latent fit, direct pixel baseline, active weak-mode mask selection.
- `tests/test_engine.py`: synthetic unit and no-leak tests (no camera/model required).

Research idea: Antti Luode. Integration and implementation: Sol / ChatGPT. November 19, 2026.
