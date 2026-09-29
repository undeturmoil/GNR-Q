#!/usr/bin/env python3
import argparse
import math
import random
import copy
import json
import time
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import datasets


def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def fake_groupwise_symmetric_quantize_(model, bits=4, group_size=128):
    """Fake W{bits}A16 RTN quantization.

    Weights are rounded to low-bit values and dequantized back to BF16.
    This intentionally isolates quantization numerics from kernel/runtime effects.
    Embeddings, RMSNorms and LM head are left untouched.
    """
    qmin = -(2 ** (bits - 1))
    qmax = (2 ** (bits - 1)) - 1
    n_linear = 0
    n_params = 0

    for _, module in model.model.layers.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        w = module.weight.data
        orig_dtype = w.dtype
        out_features, in_features = w.shape
        pad = (-in_features) % group_size

        wf = w.float()
        if pad:
            wf = F.pad(wf, (0, pad))
        wg = wf.view(out_features, -1, group_size)

        # symmetric RTN; avoid zero scales
        scale = wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
        q = torch.round(wg / scale).clamp(qmin, qmax)
        dq = (q * scale).view(out_features, -1)[:, :in_features]
        module.weight.data.copy_(dq.to(orig_dtype))

        n_linear += 1
        n_params += w.numel()

    return n_linear, n_params


def build_blocks(tokenizer, split: str, seq_len: int, n_blocks: int):
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    texts = [x["text"] for x in ds if x["text"].strip()]
    joined = "\n\n".join(texts)
    ids = tokenizer(joined, add_special_tokens=False)["input_ids"]
    needed = seq_len * n_blocks
    if len(ids) < needed:
        raise RuntimeError(f"Not enough tokens in {split}: {len(ids)} < {needed}")
    ids = torch.tensor(ids[:needed], dtype=torch.long).view(n_blocks, seq_len)
    return ids


class NeuralResidualReconstructor(nn.Module):
    """Tiny context-conditioned residual predictor: h_q -> delta_hat."""
    def __init__(self, hidden_size: int, rank: int = 128):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.down = nn.Linear(hidden_size, rank * 2, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-3)

    def forward(self, h):
        x = self.norm(h)
        a, b = self.down(x).chunk(2, dim=-1)
        x = F.silu(a) * b
        return self.up(x)


def get_layer_hidden(model, input_ids, layer_idx: int):
    out = model.model(
        input_ids=input_ids,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    # hidden_states[0] = embedding output; +1 maps decoder layer index -> its output
    return out.hidden_states[layer_idx + 1]


def correction_hook(reconstructor, enable_grad: bool):
    def _hook(module, inputs, output):
        if isinstance(output, tuple):
            h = output[0]
            ctx = nullcontext() if enable_grad else torch.no_grad()
            with ctx:
                delta = reconstructor(h.float()).to(h.dtype)
            return (h + delta,) + output[1:]
        h = output
        ctx = nullcontext() if enable_grad else torch.no_grad()
        with ctx:
            delta = reconstructor(h.float()).to(h.dtype)
        return h + delta
    return _hook


@torch.no_grad()
def local_metrics(teacher, student, recon, loader, layer_idx, device):
    mse_q = 0.0
    mse_c = 0.0
    cos_q = 0.0
    cos_c = 0.0
    n = 0
    for (ids,) in loader:
        ids = ids.to(device, non_blocking=True)
        ht = get_layer_hidden(teacher, ids, layer_idx).float()
        hq = get_layer_hidden(student, ids, layer_idx).float()
        hc = hq + recon(hq)

        mse_q += F.mse_loss(hq, ht).item()
        mse_c += F.mse_loss(hc, ht).item()
        cos_q += F.cosine_similarity(hq, ht, dim=-1).mean().item()
        cos_c += F.cosine_similarity(hc, ht, dim=-1).mean().item()
        n += 1
    return {
        "mse_q": mse_q / n,
        "mse_c": mse_c / n,
        "cos_q": cos_q / n,
        "cos_c": cos_c / n,
    }


@torch.no_grad()
def final_hidden_metrics(teacher, student, recon, loader, layer_idx, device):
    mse_q = 0.0
    mse_c = 0.0
    cos_q = 0.0
    cos_c = 0.0
    n = 0
    for (ids,) in loader:
        ids = ids.to(device, non_blocking=True)
        ht = teacher.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state.float()
        hq = student.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state.float()

        handle = student.model.layers[layer_idx].register_forward_hook(
            correction_hook(recon, enable_grad=False)
        )
        hc = student.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state.float()
        handle.remove()

        mse_q += F.mse_loss(hq, ht).item()
        mse_c += F.mse_loss(hc, ht).item()
        cos_q += F.cosine_similarity(hq, ht, dim=-1).mean().item()
        cos_c += F.cosine_similarity(hc, ht, dim=-1).mean().item()
        n += 1
    return {
        "mse_q": mse_q / n,
        "mse_c": mse_c / n,
        "cos_q": cos_q / n,
        "cos_c": cos_c / n,
    }


@torch.no_grad()
def perplexity(model, loader, device, recon=None, layer_idx=None):
    losses = []
    handle = None
    if recon is not None:
        handle = model.model.layers[layer_idx].register_forward_hook(
            correction_hook(recon, enable_grad=False)
        )
    try:
        for (ids,) in loader:
            ids = ids.to(device, non_blocking=True)
            out = model(input_ids=ids, labels=ids, use_cache=False, return_dict=True)
            losses.append(out.loss.float().item())
    finally:
        if handle is not None:
            handle.remove()
    mean_loss = sum(losses) / len(losses)
    return math.exp(mean_loss), mean_loss


def train_local_reconstructor(teacher, student, recon, loader, layer_idx, device, epochs, lr):
    recon.train()
    opt = torch.optim.AdamW(recon.parameters(), lr=lr, weight_decay=0.01)

    for epoch in range(1, epochs + 1):
        run = 0.0
        steps = 0
        for (ids,) in loader:
            ids = ids.to(device, non_blocking=True)
            with torch.no_grad():
                ht = get_layer_hidden(teacher, ids, layer_idx).float()
                hq = get_layer_hidden(student, ids, layer_idx).float()
                base = F.mse_loss(hq, ht).clamp_min(1e-10)

            hc = hq + recon(hq)
            # normalized objective: 1.0 = uncorrected quantized baseline
            loss = F.mse_loss(hc, ht) / base

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(recon.parameters(), 1.0)
            opt.step()

            run += loss.item()
            steps += 1
        print(f"[local] epoch {epoch:02d} normalized_mse={run/steps:.5f}")


def train_end_to_end(teacher, student, recon, loader, layer_idx, device, epochs, lr):
    """Refine reconstructor through the frozen quantized suffix.

    Only reconstructor weights are trainable. This makes the correction optimize
    the final representation, not merely local hidden-state MSE.
    """
    recon.train()
    opt = torch.optim.AdamW(recon.parameters(), lr=lr, weight_decay=0.01)
    handle = student.model.layers[layer_idx].register_forward_hook(
        correction_hook(recon, enable_grad=True)
    )
    try:
        for epoch in range(1, epochs + 1):
            run = 0.0
            steps = 0
            for (ids,) in loader:
                ids = ids.to(device, non_blocking=True)
                with torch.no_grad():
                    ht_final = teacher.model(
                        input_ids=ids, use_cache=False, return_dict=True
                    ).last_hidden_state.float()

                hs_final = student.model(
                    input_ids=ids, use_cache=False, return_dict=True
                ).last_hidden_state.float()

                # Normalize by teacher energy so scale is comparable across batches.
                denom = ht_final.pow(2).mean().detach().clamp_min(1e-8)
                loss = F.mse_loss(hs_final, ht_final) / denom

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(recon.parameters(), 1.0)
                opt.step()

                run += loss.item()
                steps += 1
            print(f"[e2e]   epoch {epoch:02d} final_hidden_nmse={run/steps:.6f}")
    finally:
        handle.remove()


def freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 8])
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--layer", type=int, default=27,
                    help="0-based decoder layer where correction is injected")
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--train-blocks", type=int, default=256)
    ap.add_argument("--eval-blocks", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--e2e-epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", default="gnrq_qwen3_4b.pt")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    device = torch.device("cuda")
    seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    print("=== GNR-Q PoC / DGX Spark ===")
    print(vars(args))
    print("Torch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("Transformers:", transformers.__version__)
    print("Datasets:", datasets.__version__)
    print("GPU:", torch.cuda.get_device_name(0))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("Loading BF16 teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)
    freeze(teacher)

    print("Loading second BF16 copy for quantized student...")
    student = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)
    freeze(student)

    n_layers = len(student.model.layers)
    if not (0 <= args.layer < n_layers):
        raise ValueError(f"--layer must be in [0, {n_layers-1}]")

    print(f"Applying fake W{args.bits}A16 RTN quantization, group={args.group_size}...")
    n_linear, n_params = fake_groupwise_symmetric_quantize_(
        student, bits=args.bits, group_size=args.group_size
    )
    print(f"Quantized {n_linear} Linear modules / {n_params/1e9:.3f}B weights")

    hidden = teacher.config.hidden_size
    recon = NeuralResidualReconstructor(hidden, args.rank).to(device=device, dtype=torch.float32)
    n_recon = sum(p.numel() for p in recon.parameters())
    print(f"Reconstructor: {n_recon/1e6:.3f}M params; BF16 storage ~{n_recon*2/1024**2:.2f} MiB")

    print("Preparing WikiText-2 blocks...")
    train_ids = build_blocks(tok, "train", args.seq_len, args.train_blocks)
    eval_ids = build_blocks(tok, "validation", args.seq_len, args.eval_blocks)
    train_loader = DataLoader(
        TensorDataset(train_ids), batch_size=args.batch_size, shuffle=True,
        pin_memory=True, drop_last=False
    )
    eval_loader = DataLoader(
        TensorDataset(eval_ids), batch_size=max(1, args.batch_size // 2), shuffle=False,
        pin_memory=True, drop_last=False
    )

    print("\n[0] Baseline local hidden mismatch")
    m0 = local_metrics(teacher, student, recon, eval_loader, args.layer, device)
    print(m0)

    print("\n[1] Train dynamic neural residual predictor")
    train_local_reconstructor(
        teacher, student, recon, train_loader, args.layer, device,
        epochs=args.local_epochs, lr=args.lr
    )
    recon.eval()
    m1 = local_metrics(teacher, student, recon, eval_loader, args.layer, device)
    print("after local training:", m1)

    if args.e2e_epochs > 0:
        print("\n[2] End-to-end refinement through frozen quantized suffix")
        train_end_to_end(
            teacher, student, recon, train_loader, args.layer, device,
            epochs=args.e2e_epochs, lr=args.lr * 0.3
        )
        recon.eval()

    print("\n[3] Final hidden-state recovery")
    fm = final_hidden_metrics(teacher, student, recon, eval_loader, args.layer, device)
    print(fm)

    print("\n[4] WikiText-2 validation perplexity")
    ppl_t, loss_t = perplexity(teacher, eval_loader, device)
    ppl_q, loss_q = perplexity(student, eval_loader, device)
    ppl_c, loss_c = perplexity(student, eval_loader, device, recon, args.layer)
    gap = max(loss_q - loss_t, 1e-12)
    closure = (loss_q - loss_c) / gap
    print(f"teacher BF16       : loss={loss_t:.5f} ppl={ppl_t:.3f}")
    print(f"quantized W{args.bits}A16 : loss={loss_q:.5f} ppl={ppl_q:.3f}")
    print(f"quant + GNR-Q      : loss={loss_c:.5f} ppl={ppl_c:.3f}")
    print(f"cross-entropy gap closure: {closure*100:.2f}%")

    payload = {
        "model": args.model,
        "bits": args.bits,
        "group_size": args.group_size,
        "layer": args.layer,
        "rank": args.rank,
        "state_dict": recon.state_dict(),
        "local_metrics": m1,
        "final_metrics": fm,
        "loss_teacher": loss_t,
        "loss_quant": loss_q,
        "loss_corrected": loss_c,
        "gap_closure": closure,
    }
    payload["torch_version"] = torch.__version__
    payload["cuda_version"] = torch.version.cuda
    payload["transformers_version"] = transformers.__version__
    payload["datasets_version"] = datasets.__version__
    payload["gpu"] = torch.cuda.get_device_name(0)
    payload["reconstructor_params"] = n_recon
    payload["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    payload["elapsed_sec"] = float(time.perf_counter() - t0)
    payload["model_commit"] = getattr(teacher.config, "_commit_hash", None)

    torch.save(payload, args.save)
    json_path = str(Path(args.save).with_suffix(".json"))
    json_payload = {k: v for k, v in payload.items() if k != "state_dict"}
    Path(json_path).write_text(json.dumps(json_payload, indent=2, ensure_ascii=False))
    print(f"Peak CUDA allocated: {payload['peak_cuda_allocated_bytes']/1024**3:.3f} GiB")
    print(f"Elapsed: {payload['elapsed_sec']:.1f} sec")
    print(f"Model commit: {payload['model_commit']}")
    print(f"Saved: {args.save}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
