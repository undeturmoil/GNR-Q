# Experimental Results

`raw/` contains the original machine-readable JSON outputs produced on the
NVIDIA DGX Spark during the experiments reported in the GNR-Q paper.

These files are preserved without modification.

## Packed W4

- `packed_w4_gnrq_pure.json`
  - Primary GNR-Q result.
  - Pure representation reconstruction.
  - CE_WEIGHT = 0.0.
  - Reported CE-gap closure: 27.52%.

- `packed_w4_linear_control.json`
  - Parameter-matched low-rank projection control.
  - Reported CE-gap closure: 26.51%.

- `packed_w4_gnrq_final.json`
  - Historical filename from the original experiment workspace.
  - This is the task-aware auxiliary run with CE_WEIGHT = 0.25.
  - It is not the primary quantization-reconstruction result.

## Simulated quantization

- `w4_seed{1,7,42}.json`
  - W4 seed-reproducibility experiment.

- `w3_seed{1,7,42}.json`
  - W3 seed-reproducibility experiment.

- `w4_layer{9,18,27}.json`
  - Reconstruction-placement ablation.

- `w4_rank{32,64,128,256}.json`
  - Reconstruction-capacity ablation.

- `qwen3_4b_w4_gnrq_full.json`
- `qwen3_4b_w3_gnrq_full.json`
  - Earlier full-condition runs retained for provenance.

- `smoke_w4.json`
  - Initial smoke-test result.

## Reproduced results

The scripts under `experiments/` write newly reproduced outputs to:

`results/reproduced/`

That directory is intentionally excluded from Git until the outputs are
explicitly reviewed.
