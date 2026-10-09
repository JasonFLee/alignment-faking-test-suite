# Figures

Result figures referenced from the top-level `README.md` live here. None are committed
yet.

Expected filenames, matching the checklist in the top-level README:

- `attention-compliant-vs-faking.png` — attention maps for the same question, compliant
  model against non-compliant model
- `attention-with-and-without-noise.png` — attention map with and without Gaussian input
  noise
- `noise-scale-calibration.png` — how the output degrades as `noise_std` rises
- `noise-detection-sweep.png` — refusal rate across the noise levels the sweep covers
- `final_safety_accuracy.png` — written directly by `testGaussianNoiseGraph.py`
- `fine-tuning-run.png` — training curve, step count and wall-clock time
- `intermediate-probability-distribution.png` — top tokens per layer for one prompt
- `mechanistic-data-panel.png` — the GUI panel showing the per-step trace

`testGaussianNoiseGraph.py` writes `final_safety_accuracy.png` into the working directory,
so it needs moving here. `noiseCheck.py` opens a matplotlib window rather than saving; add a
`plt.savefig("docs/figures/attention-with-and-without-noise.png")` call before `plt.show()`
to capture it.
