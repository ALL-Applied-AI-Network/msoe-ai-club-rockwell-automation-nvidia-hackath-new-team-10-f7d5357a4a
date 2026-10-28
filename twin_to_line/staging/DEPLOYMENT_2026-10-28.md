# Staging5 USB deployment status — 2026-10-28

Status: the ARM64 USB demonstration package is complete and verified for
**shadow/evaluation use on fixture G only**. It is not approved for automatic
product release, PLC/robot actuation, or fixture S.

## Launch artifact

- Local kit directory: `data/ttl_kit_staging5_2026-10-28/`
- Self-extracting launcher: `ttl_kit_staging5_2026-10-28.run`
- Reproducible package builder: [`gb10/build_staging5_kit.py`](../gb10/build_staging5_kit.py)
- Launcher SHA-256: `8921f1fb1acddf614eb4f37b03e3d55d30db4480ae773119d9aa193e849ef276`
- Launcher size: 175,373,020 bytes

The launcher contains the application, authenticated model and provenance,
ARM64 CPython 3.11.17, pinned ARM64 NumPy/OpenCV wheels, launch scripts, and
validation-only golden preflight data. It installs without network access.
Generated kit bytes and collected/training data remain local and are excluded
from Git; distribute the authenticated launcher separately by USB or as a
release artifact.

Verify and launch from the copied USB directory:

```bash
sha256sum -c ttl_kit_staging5_2026-10-28.run.sha256
bash ttl_kit_staging5_2026-10-28.run --install "$HOME/ttl_staging5_2026-10-28"
bash "$HOME/ttl_staging5_2026-10-28/run_sample.sh"
bash "$HOME/ttl_staging5_2026-10-28/live_demo.sh" folder:/path/to/images
```

Live audit output includes the source image, fixture and position, expected
occupancy and its context source, safety-gated visual class, raw five-class
model argmax, score and threshold, deterministic decision, reason code,
timestamp when available, and placement event identifier. A raw unsupported
argmax is evidence only and is never reported as a defect detection.

## Frozen model and export

- Training job: `3c875c39-3bb5-4f17-bb07-46e30679c520`
- Frozen `model.pt` SHA-256:
  `e4ea012992dcb18c1e776f6360dc111ba35f80cb83061c82e87b3f3d439b4f3e`
- Deployment `model.onnx` SHA-256:
  `3d79fbeca0a81c9a7eeb1f7d6edcc58ce7300207bafe14a4cef3b86d5170bf94`
- Validation-selected defect threshold: `0.9292464256286621`
- Exact class order: `pass`, `missing_cap`, `crooked_cap`, `damaged_vial`,
  `no_vial`

The ONNX comparison covered all 510 development crops (360 train and 150
validation). It loaded zero protected-test rows and produced zero global-argmax
or decision mismatches. Maximum absolute logit error was
`3.814697265625e-05`; maximum probability error was
`3.516674041748047e-06`.

## Package verification

The final launcher passed its outer checksum, embedded archive checksum, all 43
inner checksum records, extraction, and the model-aware golden preflight in a
deployment-version-matched local environment. The preflight used two
validation-only frames (60 positions),
matched all expected rows and scores with zero drift, and produced:

- Decisions: 44 `PASS`, 3 `DEFECT`, 1 `MISSED`, 12 `WAITING`
- Visual results: 44 `pass`, 3 `crooked_cap`, 13 `no_vial`
- Protected-test images or manifests embedded: none

The final focused release suite passed 78 tests with 10 optional-dependency
skips. The broader `twin_to_line` suite passed 288 tests with 54 skips and two
known environment-only failures: an absent archived `rc_1024/inputs/staged_3`
fixture and the optional USD `pxr` module not being installed. Neither is used
by the offline staging5 live-demo path.

YAAD ARM64 deployment smoke job
`36ddd73d-7d2a-4944-a177-30887a656cf0` succeeded on `spark-e82b`. It verified
the frozen model hash, ARM64 standalone Python archive, offline installation of
the pinned ARM64 wheels, OpenCV loading of an ONNX export, and a finite
`[30, 5]` inference result. Recorded environment and timing:

- Architecture: `aarch64`
- Python: `3.11.17`
- NumPy: `2.1.0`
- OpenCV: `4.10.0`
- Median 30-crop CPU inference: `0.08664284693077207` seconds
- Protected test used by this smoke job: false

The YAAD run rehearsed the exact frozen model hash, pinned standalone Python
archive, pinned ARM64 wheel versions, and OpenCV-DNN inference components. It
did not receive and execute the final 175 MB self-extracting archive itself.
Therefore the first compute-module installation must be treated as the final
archive rehearsal; the installer fails closed unless its embedded checksums,
offline bootstrap, and golden preflight all pass.

## Protected evaluation and release limitations

Protected evaluation job `14160052-ea75-4861-bf4d-46fd90b82b4b` was run once
after model and threshold freezing. It must not be used for tuning. Results were
147/150 (98.0%) exact visual and process agreement, but exact supported-defect
recall was only 5/8 (62.5%). Conservative non-PASS intervention covered 7/8
supported defects; one crooked-cap defect at G29 escaped as `PASS`.

These results are a one-visit event holdout under shared camera, lighting, and
fixture conditions. The protected labels are staging protocol references, not
independently adjudicated image labels, so they are not independent deployment
proof.

There are no `damaged_vial` development examples and no fixture-S evidence.
Therefore `five_class_model_ready` remains false. A `damaged_vial` global
argmax is forced to `UNKNOWN`, and deterministic rules resolve it to
`UNSURE`/manual review. The demo must remain shadow-only and must never be used
as authority to pass or release product.
