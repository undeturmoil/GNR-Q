# GNR-Q

## Inference-Time State-Conditioned Reconstruction for Memory-Efficient Quantized Language Models

**GNR-Q (Guided Neural Reconstruction for Quantization)** investigates whether part of the representation quality lost through low-bit LLM quantization can be reconstructed at inference time from the quantized model's current hidden state.

The full-precision teacher is used only during offline reconstruction training. Deployment uses a frozen quantized backbone together with a compact state-conditioned reconstructor.

## Core formulation

For a full-precision hidden state $h^{FP}$ and its quantized counterpart $h^Q$, define the representation residual as

$$
e = h^{FP} - h^Q.
$$

GNR-Q learns a compact reconstructor

$$
\hat e = R_\theta(h^Q)
$$

and applies

$$
\hat h = h^Q + \hat e.
$$

The working hypothesis is that a useful component of quantization-induced representation error is statistically predictable from the surviving quantized state.

## Main packed-W4 result

Model: `Qwen/Qwen3-4B-Base`

| Configuration | Model allocation | Sidecar | Loss | PPL | CE-gap closure |
|---|---:|---:|---:|---:|---:|
| BF16 | 7.545 GiB | - | 2.56338 | 12.980 | - |
| Packed W4 | 2.792 GiB | - | 2.70808 | 15.000 | 0% |
| Packed W4 + GNR-Q | ~2.794 GiB | 1.880 MiB | 2.66825 | 14.415 | 27.52% |
| Packed W4 + low-rank control | ~2.794 GiB | 1.880 MiB | 2.66972 | 14.436 | 26.51% |

The primary packed-W4 GNR-Q sidecar is trained with **representation reconstruction only** (`CE_WEIGHT = 0.0`). No next-token cross-entropy supervision is used in this primary result.

Final hidden-state MSE decreases from `0.53760` to `0.39009`, a reduction of approximately **27.44%**.

The combined packed-W4 model plus BF16 sidecar retains approximately **62.97% lower CUDA-allocated model memory** than the BF16 model allocation.

## Diagnostic experiments

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
├── src/
│   ├── gnrq_poc_spark.py
│   ├── gnrq_packed_pure.py
│   ├── gnrq_linear_control.py
│   └── gnrq_packed_task_aware.py
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
└── paper/
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

The simulated experiments explicitly use 3 local reconstruction epochs, followed by 1 frozen-suffix end-to-end refinement epoch. With the reported base learning rate of `3e-4`, the end-to-end refinement code uses `0.3 × lr = 9e-5`.

Training batch size is 4; evaluation batch size is 2 in the reported configuration.

## Model and data

The repository does not redistribute model weights or WikiText-2.

The scripts obtain:

- Model: `Qwen/Qwen3-4B-Base`
- Dataset: `Salesforce/wikitext`
- Dataset configuration: `wikitext-2-raw-v1`
- Sequence length: 128
- Training blocks: 256
- Evaluation blocks: 64

## Raw results

Original machine-readable JSON outputs used to prepare the paper are preserved under `results/raw/`.

`packed_w4_gnrq_pure.json` is the primary pure-reconstruction packed-W4 result.

`packed_w4_linear_control.json` is the parameter-matched low-rank projection control.

`packed_w4_gnrq_final.json` is retained under its historical filename for provenance; it corresponds to the **task-aware auxiliary experiment** with `CE_WEIGHT = 0.25`, not the primary reconstruction-only result.

## Paper

**GNR-Q: Inference-Time State-Conditioned Reconstruction for Memory-Efficient Quantized Language Models**

Authors:

- SeungGeun Baeck — Vhexlab Co., Ltd.
- Dumi Pyo — Department of Psychology, Ajou University
- HaeJung Suk — Department of Digital Media, Ajou University (corresponding author)

Planned arXiv classification:

- Primary: `cs.LG`
- Cross-list: `cs.CL`

Repository: https://github.com/undeturmoil/GNR-Q

## License

Source code in this repository is released under the **Apache License 2.0**.

The accompanying manuscript is planned for distribution under **CC BY 4.0**.
