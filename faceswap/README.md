# Portrait transfer

The entry point is `run_3d_faceswap.py`, using INSwapper by default after the local benchmark. Models are reused when calling a pipeline instance repeatedly. Inputs are never modified. Outputs must be PNG; each output also has a `_mask.png` and a JSON record containing input hashes, selected face indices, configuration, execution providers, elapsed time, and diagnostic scores.

```powershell
.\venv\Scripts\python.exe -X utf8 run_3d_faceswap.py -s img35.jpg -t img37.jpg -o outputs/portrait.png
```

Relative paths are resolved against the project directory. `--source-index` and `--target-index` select faces ordered by descending bounding-box area. Detection failures, invalid geometry, insufficient skin samples and attempts to overwrite an input raise errors. A failed quality gate is reported for review rather than silently labelled a successful quality result.

## Backends

- `--backend liveportrait`: 21 learned 3D keypoints, absolute target pose/expression, source scale, synthesis, optional lip aperture correction measured on the synthesized face, residual five-point alignment, then semantic composition. `--no-retarget-lips` disables lip correction. `--stitching` is experimental and degraded pose in the development tests; it is disabled by default.
- `--backend inswapper`: target-conditioned identity transfer using the source's normalized buffalo_l embedding. The generator's native face resolution is 128x128. It reuses the same semantic masks and bounded skin color transfer. It does not use LivePortrait's 3D renderer or lip retargeting.

The mask-v2 compositor preserves an estimated boundary band for B's glasses frames, while replacing transparent lens interiors with the generated face. Likely opaque lenses retain full protection. Frame boundaries are estimated from the parser's glasses region; they are not a precise material segmentation and can miss a bridge, reflections or a thin rim. Lens reflections are not independently reconstructed. Hair, ears and accessories remain protected. Pixels outside the actual alpha and inside the estimated protected frame are preserved exactly.

Color correction now applies to segmented skin rather than iris, teeth and lips. INSwapper pixels are sampled directly from the native model crop into the output, avoiding the extra 128-to-512-to-output image resampling. The 512 crop remains useful for masks and debug images; this change does not increase the generator's native resolution.

To inspect an individual test, add `--debug`:

```powershell
.\venv\Scripts\python.exe -X utf8 run_3d_faceswap.py -s img35.jpg -t img31.png -o outputs/glasses_test.png --debug
```

The `glasses_test_debug` directory contains `model_input_native.png`, `model_native.png`, `generated_crop.png`, `target_crop.png`, `matched_crop.png`, `support.png`, `alpha_crop.png`, `frame_protection.png`, `lens_interior.png`, `result_crop.png`, label visualizations and coordinate transforms. Use these to distinguish generator defects from compositing defects. New JSON records include implementation hashes and `pipeline_revision: sampling-v3`.

## Sampling and native-256 experiments (24 October 2026)

Defaults remain `--backend inswapper --input-sampling native` without bridge refinement. Anti-aliasing reduced grain in the reported case, but all three tested strengths regressed one mouth gate in the 16-pair frozen-mask regression set. HyperSwap passed 13/16 rather than the native128 baseline's 16/16. These are project proxy gates, not human approval. See `experiments/round3_20261024/REPORT.md` for the separate public-pipeline checks and remaining failures.

```powershell
# Compare these against the same inputs with --input-sampling native.
.\venv\Scripts\python.exe -X utf8 run_3d_faceswap.py -s img37.jpg -t img36.png -o outputs/round3_area.png --input-sampling area --debug
.\venv\Scripts\python.exe -X utf8 run_3d_faceswap.py -s img37.jpg -t img36.png -o outputs/round3_256.png --backend hyperswap --input-sampling area --debug
```

`area` warps at 4x then downsamples with INTER_AREA and compensates pixel centers. `balanced` blends 50% area with native; `light` blends 25%. Upscaling retains the native warp. These options change model inputs, not native generator resolution. Native mode reproduces the previous inference.

HyperSwap 1a is an experimental native256 backend using the same semantic compositor, deliberately ignoring its learned output mask for controlled comparisons. Install via `experiments/round3_20261024/download_candidate.py` if absent. Release source, CRC and SHA256 are recorded beside the model; weights use ResearchRAIL-MS, independently of repository code licensing. Model prerequisites and usage restrictions still apply.

`--protect-bridge` adds a narrow edge estimate between two detected lens components. Sparse references drawn from two target photos show improved bridge coverage, but rim coverage remains incomplete and reflections are not separated as an optical layer. This option remains experimental. `--protection-mask path.png` accepts an original-target-sized grayscale image; pixels above 127 preserve B exactly. Use it for human-corrected opaque rims, not the entire transparent lens. JSON includes the custom mask hash and exact-preservation diagnostic.

Every run separates `quality.project_gate_pass` from `visual_acceptance.decision`, initially `pending_human_review`. Inspect source resemblance, B expression, eyes/mouth, skin, seams, actual frame rims/bridge and reflections. Neither an estimated-mask invariant nor an embedding score is visual acceptance.

For a controlled comparison with the previous compositor, keep the same inputs and run:

```powershell
.\venv\Scripts\python.exe -X utf8 run_3d_faceswap.py -s img35.jpg -t img31.png -o outputs/glasses_legacy.png --glasses-policy full --legacy-color-transfer --two-step-sampling --debug
```

PyTorch uses CUDA when available; `--device cpu` selects CPU. ONNX sessions currently use the available CPU provider with four threads. Requesting `--device cuda` does not claim that ONNX inference runs on CUDA. No ONNX installation is replaced automatically.

## Local model prerequisites

Existing LivePortrait checkpoints are under `pretrained_weights/liveportrait`. The existing face parser uses `res/cp/79999_iter.pth`, the repository's `model.py`/`resnet.py`, and the cached ResNet18 backbone. Detection, 106 landmarks, 68 landmarks/pose and recognition use the existing buffalo_l package in the user's `.insightface/models` directory.

The additional identity model is installed under `pretrained_weights/inswapper/inswapper_128.onnx`. Its source and verified SHA-256 are recorded beside it. To fetch it again explicitly:

```powershell
.\venv\Scripts\python.exe experiments/faceswap/download_swapper.py
```

The downloader uses the [official InsightFace release](https://github.com/deepinsight/insightface/releases/tag/model-zoo) and verifies the published digest before installation. See the [official model usage](https://github.com/deepinsight/insightface/tree/master/examples/in_swapper) and [model licensing](https://github.com/deepinsight/insightface#license); model licensing differs from the code's MIT license.

## Reproduce evaluation

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements_portrait_eval.txt
.\venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
.\venv\Scripts\python.exe -X utf8 experiments/faceswap/benchmark.py --phase tune --variant identity --single --run-name identity_tune_repeat
.\venv\Scripts\python.exe -X utf8 experiments/faceswap/benchmark.py --phase holdout --variant identity --run-name identity_holdout_repeat
.\venv\Scripts\python.exe -X utf8 experiments/faceswap/benchmark.py --phase final --variant identity --run-name identity_final_repeat
.\venv\Scripts\python.exe -X utf8 experiments/faceswap/benchmark.py --phase self --variant identity --run-name identity_self_repeat
.\venv\Scripts\python.exe -X utf8 experiments/faceswap/edge_cases.py
.\venv\Scripts\python.exe -X utf8 experiments/mask_review/ablation.py --phase development --run-name repeat_mask_development
.\venv\Scripts\python.exe -X utf8 experiments/mask_review/ablation.py --phase validation --variant direct --run-name repeat_mask_validation
```

Use `--baseline` to also run the archived original algorithm. Historical variant `identity` freezes the previous compositor; `identity_current` uses the mask-v2 defaults. The benchmark isolates legacy inputs so its template-writing behavior cannot overwrite the original images' `.pkl` files. The old script is retained at `experiments/faceswap/baseline_original.py` for comparison, not as the production entry point.

## Reading the scores

ArcFace cosine is a similarity, not an accuracy percentage or probability. Pose error comes from an independent InsightFace estimator. Eye and mouth scores compare normalized 68-landmark aperture ratios, not complete facial expression. Background MAE uses the same target-derived exclusion region for every algorithm; exact outside-alpha preservation is a separate invariant. `visual_quality_status` remains unverified even when automated gates pass. The glasses-region pixel difference may increase when its transparent interior is correctly replaced; use `estimated_frame_max_error` for frame preservation, not the whole-glasses-region difference.

Project gates are declared in `benchmark.py`: identity cosine >= 0.35, identity margin over target >= 0.05, pose MAE <= 5 degrees, mouth ratio error <= 0.08, mean eye ratio error <= 0.10, and background MAE <= 1 intensity level. These are provisional engineering gates, not published acceptance standards. Identity margin is inapplicable to identical inputs and can be misleading for two photos of the same person. Early self-test reports that predate this correction retain their original gate fields for audit; use their reconstruction metrics, not those margin flags.

PSNR, SSIM and LPIPS are reported for self-reconstruction, where an actual reference exists, including a face bounding-box ROI so unchanged background cannot dominate. Cross-identity swaps have no exact pixel ground truth. FID is not used on this small collection of portraits; it requires a suitable real-image reference distribution and substantially more samples for a meaningful distribution-level comparison. See [TorchMetrics FID](https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html).

The image pairs are limited to manually reviewed, clothed portraits in this workspace. This evaluation does not establish broad demographic coverage, extreme-pose performance, commercial deployment readiness, or human-rated realism. Inspect the comparison images as well as the metrics. No face restoration model is applied because invented detail can change identity.
