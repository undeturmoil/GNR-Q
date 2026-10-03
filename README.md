# GNR-Q

## Inference-Time State-Conditioned Reconstruction for Memory-Efficient Quantized Language Models

**GNR-Q (Guided Neural Reconstruction for Quantization)** investigates whether part of the representation quality lost through low-bit LLM quantization can be reconstructed at inference time from the quantized model's current hidden state.

The full-precision teacher is used only during offline reconstruction training. Deployment uses a frozen quantized backbone together with a compact state-conditioned reconstructor.

## Core formulation

For a full-precision hidden state \(h^{FP}\) and its quantized counterpart \(h^Q\), define

\[
e = h^{FP} - h^Q.
\]

GNR-Q learns

\[
\hat e = R_\theta(h^Q)
\]

and applies

\[
\hat h = h^Q + \hat e.
\]

The central hypothesis is deliberately narrow: a useful component of quantization-induced representation error is statistically predictable from the surviving quantized state.

## Primary packed-W4 result

Model: `Qwen/Qwen3-4B-Base`

| Configuration | Model allocation | Sidecar | Loss | PPL | CE-gap closure |
|---|---:|---:|---:|---:|---:|
| BF16 | 7.545 GiB | - | 2.56338 | 12.980 | - |
| Packed W4 | 2.792 GiB | - | 2.70808 | 15.000 | 0% |
| Packed W4 + GNR-Q | ~2.794 GiB | 1.880 MiB | 2.66825 | 14.415 | 27.52% |
| Packed W4 + low-rank control | ~2.794 GiB | 1.880 MiB | 2.66972 | 14.436 | 26.51% |

The primary packed-W4 GNR-Q sidecar is trained with **representation reconstruction only** (`CE_WEIGHT = 0.0`). No next-token cross-entropy supervision is used.

Final hidden-state MSE decreases from `0.53760` to `0.39009`, a reduction of approximately **27.44%**.

The combined packed-W4 model plus BF16 sidecar retains approximately **62.97% lower CUDA-allocated model memory** than the BF16 model allocation.

## Final validation and controls

A separate matched packed-W4 run reproduces the reconstruction effect and adds state-dependence controls, cross-corpus transfer, spectral analysis, and runtime measurement.

### State-dependence controls

| Condition | Loss | PPL | Hidden MSE | CE-gap closure |
|---|---:|---:|---:|---:|
| BF16 | 2.56338 | 12.980 | - | - |
| Packed W4 | 2.70808 | 15.000 | 0.53760 | 0% |
| Mean residual | 2.68484 | 14.656 | 0.47255 | 16.06% |
| Shuffled residual | 2.68678 | 14.684 | 0.47240 | 14.72% |
| Correctly paired GNR-Q | 2.66768 | 14.407 | 0.38863 | 27.92% |

The mean-residual control applies one global correction vector to every state. The shuffled-residual control retains the GNR-Q architecture but breaks the intended state-residual correspondence during training.

Correct pairing therefore adds **11.86 percentage points** of CE-gap recovery over the global-mean control and **13.20 percentage points** over the shuffled-residual control.

### Cross-corpus transfer

The sidecar is trained only on WikiText-2 and evaluated on 8,192 C4 validation tokens **without retraining**.

| C4 condition | Loss | PPL | Hidden MSE | CE-gap closure |
|---|---:|---:|---:|---:|
| BF16 | 3.15675 | 23.494 | - | - |
| Packed W4 | 3.24364 | 25.627 | 0.29456 | 0% |
| W4 + WikiText-trained GNR-Q | 3.21265 | 24.845 | 0.16702 | 35.67% |

Hidden-state MSE is reduced by **43.30%** on the C4 evaluation blocks.

### Residual spectrum

Spectral statistics are computed over 8,192 WikiText-2 validation token states.

| Matrix | Top-8 energy | Top-32 | Top-128 | Entropy effective rank | Participation ratio |
|---|---:|---:|---:|---:|---:|
| Target residual | 28.41% | 38.64% | 52.79% | 453.39 | 45.57 |
| Predicted correction | 87.76% | 99.50% | ~100% | 7.22 | 3.59 |
| Unexplained residual | 17.81% | 27.43% | 43.17% | 777.81 | 128.98 |

The predicted correction is produced through a rank-128 bottleneck, so its absolute rank is architecture constrained. The useful observation is that the captured correction is highly concentrated within that subspace, while the residual remaining after correction is substantially more diffuse.

### Runtime microbenchmark

Batch size 1 on NVIDIA GB10:

| Path | Packed W4 | W4 + GNR-Q |
|---|---:|---:|
| 128-token prefill | 861.46 tok/s | 861.47 tok/s |
| Cached decode | 47.39 tok/s | 47.44 tok/s |
| Decode latency | 21.102 ms/token | 21.077 ms/token |

The nominal decode difference in the idle rerun is `-0.12%`. This is treated as measurement noise rather than a speedup: no resolvable GNR-Q throughput penalty was observed in this specific batch-1 microbenchmark.

## Diagnostic simulated experiments

### Seed reproducibility

Simulated W4 CE-gap closure:

- seed 1: 18.03%
- seed 7: 18.57%
- seed 42: 17.83%

Simulated W3 CE-gap closure:

- seed 1: 14.56%
- seed 7: 14.22%
- seed 42: 15.06%

### Reconstruction placement

Simulated W4, rank 128:

- layer 9: 25.50%
- layer 18: 23.46%
- layer 27: 17.83%

### Reconstruction capacity

Simulated W4, layer 27:

- rank 32: 12.28%
- rank 64: 14.42%
- rank 128: 17.83%
- rank 256: 18.19%

## Repository structure

```text
GNR-Q/
├── Dockerfile
├── requirements.txt
├── CITATION.cff
├── run_gnrq_v5_final.sh
├── src/
│   ├── gnrq_poc_spark.py
│   ├── gnrq_packed_pure.py
│   ├── gnrq_linear_control.py
│   ├── gnrq_packed_task_aware.py
│   └── gnrq_v5_final_validation.py
├── experiments/
│   ├── run_w4_seeds.sh
│   ├── run_w3_seeds.sh
│   ├── run_layer_ablation.sh
│   ├── run_rank_ablation.sh
│   ├── run_packed_pure.sh
│   ├── run_lowrank_control.sh
│   └── run_task_aware.sh
├── results/
│   └── raw/
│       └── v5_final_validation_final.json
└── paper/
    ├── main.tex
    ├── GNR-Q_preprint.pdf
    └── README.md
```

## Environment

Primary experiments were run on an NVIDIA DGX Spark with NVIDIA GB10.

- Python 3.12.3
- CUDA 13.0
- PyTorch 2.13.0+cu130
- Transformers 5.17.0
- Datasets 5.0.1
- TorchAO 0.18.0

The supplied Dockerfile is based on:

```text
vllm/vllm-openai:v0.30.0
```

## Build

From the repository root:

```bash
docker build -t gnrq:paper .
```

Run with GPU access and a persistent Hugging Face cache:

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  -v "$PWD:/workspace/GNR-Q" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  gnrq:paper
```

Inside the container:

```bash
bash experiments/run_packed_pure.sh
bash experiments/run_lowrank_control.sh
bash experiments/run_w4_seeds.sh
bash experiments/run_w3_seeds.sh
bash experiments/run_layer_ablation.sh
bash experiments/run_rank_ablation.sh
```

The final matched validation can be launched directly from the repository root on the DGX Spark host:

```bash
./run_gnrq_v5_final.sh
```

The runner starts the `gnrq:paper` container and executes `src/gnrq_v5_final_validation.py`.

## Model and data

The repository does not redistribute model weights, WikiText-2, or C4.

The scripts obtain:

- Model: `Qwen/Qwen3-4B-Base`
- Primary dataset: `Salesforce/wikitext`, `wikitext-2-raw-v1`
- Cross-corpus evaluation: English C4 validation stream
- Sequence length: 128
- Training blocks: 256
- WikiText evaluation blocks: 64
- C4 evaluation blocks: 64

## Raw results

Machine-readable JSON outputs used to prepare the paper are preserved under `results/raw/`.

Key files:

- `packed_w4_gnrq_pure.json` — primary pure-reconstruction packed-W4 result.
- `packed_w4_linear_control.json` — parameter-matched low-rank projection control.
- `v5_final_validation_final.json` — matched state-dependence controls, C4 transfer, residual spectrum, and batch-1 runtime.
- `packed_w4_gnrq_final.json` — historical filename for the **task-aware auxiliary experiment** with `CE_WEIGHT = 0.25`; it is not the primary reconstruction-only result.

## Paper

**GNR-Q: Inference-Time State-Conditioned Reconstruction for Memory-Efficient Quantized Language Models**

Authors:

- SeungGeun Baeck — Vhexlab Co., Ltd.
- Dumi Pyo — Department of Psychology, Ajou University
- HaeJung Suk — Department of Digital Media, Ajou University (corresponding author)

### Preprint

Zenodo v2.0:

- DOI: https://doi.org/10.5281/zenodo.23113127

### Repository

- https://github.com/undeturmoil/GNR-Q

## License

Source code in this repository is released under the **Apache License 2.0**.

The accompanying manuscript is distributed under **CC BY 4.0** via Zenodo.
