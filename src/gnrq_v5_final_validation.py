#!/usr/bin/env python3
import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import quantize_, Int4WeightOnlyConfig
from torchao.quantization import Int4TilePackedTo4dTensor

from gnrq_poc_spark import NeuralResidualReconstructor, build_blocks, seed_all


def freeze(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def recon_forward(recon, h):
    dtype = next(recon.parameters()).dtype
    return h + recon(h.to(dtype)).to(h.dtype)


def make_loader(ids, batch_size, shuffle=False):
    return DataLoader(TensorDataset(ids), batch_size=batch_size, shuffle=shuffle)


def build_c4_blocks(tokenizer, seq_len, n_blocks):
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    token_ids = []
    need = seq_len * n_blocks
    for row in ds:
        text = row.get("text", "")
        if not text or not text.strip():
            continue
        token_ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        if len(token_ids) >= need:
            break
    if len(token_ids) < need:
        raise RuntimeError(f"Not enough C4 tokens: {len(token_ids)} < {need}")
    return torch.tensor(token_ids[:need], dtype=torch.long).view(n_blocks, seq_len)


@torch.no_grad()
def eval_ce(model, loader, correction=None):
    losses = []
    for (ids,) in loader:
        ids = ids.cuda(non_blocking=True)
        h = model.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
        if correction is not None:
            h = correction(h.float()).to(model.lm_head.weight.dtype)
        logits = model.lm_head(h)
        loss = F.cross_entropy(
            logits[:, :-1, :].float().reshape(-1, logits.size(-1)),
            ids[:, 1:].reshape(-1),
        )
        losses.append(loss.item())
    loss = sum(losses) / len(losses)
    return loss, math.exp(loss)


@torch.no_grad()
def eval_hidden(teacher, student, loader, correction=None):
    mse_q = mse_c = cos_q = cos_c = 0.0
    n = 0
    for (ids,) in loader:
        ids = ids.cuda(non_blocking=True)
        ht = teacher.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state.float()
        hq = student.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state.float()
        hc = hq if correction is None else correction(hq)
        mse_q += F.mse_loss(hq, ht).item()
        mse_c += F.mse_loss(hc, ht).item()
        cos_q += F.cosine_similarity(hq, ht, dim=-1).mean().item()
        cos_c += F.cosine_similarity(hc, ht, dim=-1).mean().item()
        n += 1
    return {
        "mse_q": mse_q / n,
        "mse_c": mse_c / n,
        "mse_reduction_pct": (1.0 - (mse_c / mse_q)) * 100.0,
        "cos_q": cos_q / n,
        "cos_c": cos_c / n,
    }


def train_normal(teacher, student, recon, loader, epochs, lr):
    opt = torch.optim.AdamW(recon.parameters(), lr=lr, weight_decay=0.01)
    for epoch in range(1, epochs + 1):
        recon.train()
        run = 0.0
        steps = 0
        for (ids,) in loader:
            ids = ids.cuda(non_blocking=True)
            with torch.no_grad():
                ht = teacher.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state.float()
                hq = student.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state.float()
                base_mse = F.mse_loss(hq, ht).detach().clamp_min(1e-8)
            hc = hq + recon(hq)
            loss = F.mse_loss(hc, ht) / base_mse
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(recon.parameters(), 1.0)
            opt.step()
            run += loss.item()
            steps += 1
        print(f"[normal] epoch {epoch:02d} nmse={run/steps:.6f}")
    recon.eval()


def train_shuffled(teacher, student, recon, loader, epochs, lr):
    opt = torch.optim.AdamW(recon.parameters(), lr=lr, weight_decay=0.01)
    for epoch in range(1, epochs + 1):
        recon.train()
        run = 0.0
        steps = 0
        for (ids,) in loader:
            ids = ids.cuda(non_blocking=True)
            with torch.no_grad():
                ht = teacher.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state.float()
                hq = student.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state.float()
                e = ht - hq
                flat = e.reshape(-1, e.size(-1))
                perm = torch.randperm(flat.size(0), device=flat.device)
                e_shuf = flat[perm].view_as(e)
                denom = e_shuf.pow(2).mean().detach().clamp_min(1e-8)
            pred = recon(hq.to(next(recon.parameters()).dtype)).float()
            loss = F.mse_loss(pred, e_shuf) / denom
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(recon.parameters(), 1.0)
            opt.step()
            run += loss.item()
            steps += 1
        print(f"[shuffle] epoch {epoch:02d} nmse={run/steps:.6f}")
    recon.eval()


@torch.no_grad()
def estimate_mean_residual(teacher, student, loader):
    total = None
    n = 0
    for (ids,) in loader:
        ids = ids.cuda(non_blocking=True)
        ht = teacher.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state.float()
        hq = student.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state.float()
        e = (ht - hq).reshape(-1, ht.size(-1))
        s = e.sum(dim=0)
        total = s if total is None else total + s
        n += e.size(0)
    return total / n


@torch.no_grad()
def residual_spectrum(teacher, student, recon, loader, max_tokens=8192):
    xs, ps, rs = [], [], []
    count = 0
    for (ids,) in loader:
        ids = ids.cuda(non_blocking=True)
        ht = teacher.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state.float()
        hq = student.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state.float()
        pred = recon(hq.to(next(recon.parameters()).dtype)).float()
        e = (ht - hq).reshape(-1, ht.size(-1))
        p = pred.reshape(-1, pred.size(-1))
        r = e - p
        take = min(e.size(0), max_tokens - count)
        if take <= 0:
            break
        xs.append(e[:take].cpu())
        ps.append(p[:take].cpu())
        rs.append(r[:take].cpu())
        count += take
        if count >= max_tokens:
            break

    def spectrum(mats):
        mat = torch.cat(mats, dim=0).float()
        mat = mat - mat.mean(dim=0, keepdim=True)
        cov = (mat.T @ mat) / max(mat.size(0) - 1, 1)
        eig = torch.linalg.eigvalsh(cov).clamp_min(0).flip(0)
        total = eig.sum().clamp_min(1e-20)
        frac = eig / total
        cum = torch.cumsum(frac, dim=0)
        energy = {}
        for k in [8, 16, 32, 64, 128, 256, 512, 1024]:
            if k <= eig.numel():
                energy[str(k)] = float(cum[k - 1].item())
        p = frac.clamp_min(1e-20)
        return {
            "n_tokens": int(mat.size(0)),
            "dimension": int(mat.size(1)),
            "topk_cumulative_energy": energy,
            "entropy_effective_rank": float(torch.exp(-(p * torch.log(p)).sum()).item()),
            "participation_ratio": float((total * total / eig.square().sum().clamp_min(1e-20)).item()),
            "top_20_eigenvalue_fractions": [float(v) for v in frac[:20].tolist()],
        }

    return {
        "target_residual": spectrum(xs),
        "predicted_correction": spectrum(ps),
        "unexplained_residual": spectrum(rs),
    }


def sync():
    torch.cuda.synchronize()


@torch.no_grad()
def benchmark_prefill(student, recon, seq_len=128, batch=1, warmup=10, iters=50):
    vocab = student.config.vocab_size
    ids = torch.randint(0, vocab, (batch, seq_len), device="cuda")

    def base_once():
        h = student.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
        student.lm_head(h)

    def recon_once():
        h = student.model(input_ids=ids, use_cache=False, return_dict=True).last_hidden_state
        h = recon_forward(recon, h.float()).to(student.lm_head.weight.dtype)
        student.lm_head(h)

    def measure(fn):
        for _ in range(warmup):
            fn()
        sync()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        sync()
        sec = time.perf_counter() - t0
        return {"mean_ms": sec / iters * 1000.0, "tokens_per_sec": batch * seq_len * iters / sec}

    return {"packed_w4": measure(base_once), "w4_plus_gnrq": measure(recon_once)}


@torch.no_grad()
def benchmark_decode(student, recon, prompt_len=128, batch=1, steps=128, warmup_runs=2, runs=5):
    vocab = student.config.vocab_size

    def one_run(use_recon):
        prompt = torch.randint(0, vocab, (batch, prompt_len), device="cuda")
        out = student.model(input_ids=prompt, use_cache=True, return_dict=True)
        past = out.past_key_values
        h = out.last_hidden_state[:, -1:, :]
        if use_recon:
            h = recon_forward(recon, h.float()).to(student.lm_head.weight.dtype)
        logits = student.lm_head(h)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        sync()
        t0 = time.perf_counter()
        for _ in range(steps):
            out = student.model(input_ids=next_id, past_key_values=past, use_cache=True, return_dict=True)
            past = out.past_key_values
            h = out.last_hidden_state
            if use_recon:
                h = recon_forward(recon, h.float()).to(student.lm_head.weight.dtype)
            logits = student.lm_head(h)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        sync()
        return time.perf_counter() - t0

    for _ in range(warmup_runs):
        one_run(False)
        one_run(True)

    base_times = [one_run(False) for _ in range(runs)]
    recon_times = [one_run(True) for _ in range(runs)]
    base_mean = sum(base_times) / len(base_times)
    recon_mean = sum(recon_times) / len(recon_times)
    return {
        "packed_w4": {"ms_per_token": base_mean / steps * 1000.0, "tokens_per_sec": batch * steps / base_mean},
        "w4_plus_gnrq": {"ms_per_token": recon_mean / steps * 1000.0, "tokens_per_sec": batch * steps / recon_mean},
        "overhead_pct": (recon_mean / base_mean - 1.0) * 100.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Base")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--rank", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--train-blocks", type=int, default=256)
    ap.add_argument("--eval-blocks", type=int, default=64)
    ap.add_argument("--c4-blocks", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", default="results/raw/v5_final_validation.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")

    print("=== GNR-Q V5 FINAL VALIDATION ===")
    print(vars(args))
    print("GPU:", torch.cuda.get_device_name(0))
    print("Torch:", torch.__version__)
    print("CUDA:", torch.version.cuda)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("Preparing WikiText-2...")
    train_ids = build_blocks(tok, "train", args.seq_len, args.train_blocks)
    eval_ids = build_blocks(tok, "validation", args.seq_len, args.eval_blocks)
    train_loader = make_loader(train_ids, args.batch_size, shuffle=True)
    train_ordered = make_loader(train_ids, args.batch_size, shuffle=False)
    eval_loader = make_loader(eval_ids, 2, shuffle=False)

    print("Preparing C4 validation...")
    c4_ids = build_c4_blocks(tok, args.seq_len, args.c4_blocks)
    c4_loader = make_loader(c4_ids, 2, shuffle=False)

    print("Loading BF16 teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="sdpa"
    ).to(device)
    freeze(teacher)

    print("Loading student...")
    student = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True, attn_implementation="sdpa"
    ).to(device)
    freeze(student)

    print("Packing student to INT4/HQQ...")
    cfg = Int4WeightOnlyConfig(
        group_size=args.group_size,
        int4_packing_format="tile_packed_to_4d",
        int4_choose_qparams_algorithm="hqq",
    )
    quantize_(student.model.layers, cfg)
    gc.collect()
    torch.cuda.empty_cache()
    sync()

    packed_count = sum(
        1 for m in student.model.layers.modules()
        if isinstance(m, nn.Linear) and isinstance(m.weight, Int4TilePackedTo4dTensor)
    )
    print("Packed Linear modules:", packed_count)

    hidden = teacher.config.hidden_size

    print("\n[1] Train normal GNR-Q")
    recon = NeuralResidualReconstructor(hidden, args.rank).to(device=device, dtype=torch.float32)
    train_normal(teacher, student, recon, train_loader, args.epochs, args.lr)
    recon = recon.to(torch.bfloat16)
    normal_corr = lambda h: recon_forward(recon, h)

    print("\n[2] WikiText evaluation")
    wt_loss_t, wt_ppl_t = eval_ce(teacher, eval_loader)
    wt_loss_q, wt_ppl_q = eval_ce(student, eval_loader)
    wt_loss_r, wt_ppl_r = eval_ce(student, eval_loader, normal_corr)
    wt_gap = wt_loss_q - wt_loss_t
    wt_closure = (wt_loss_q - wt_loss_r) / wt_gap
    wt_hidden = eval_hidden(teacher, student, eval_loader, normal_corr)

    print("\n[3] Mean-residual control")
    mean_e = estimate_mean_residual(teacher, student, train_ordered)
    mean_corr = lambda h: h + mean_e.to(h.device, h.dtype)
    mean_loss, mean_ppl = eval_ce(student, eval_loader, mean_corr)
    mean_closure = (wt_loss_q - mean_loss) / wt_gap
    mean_hidden = eval_hidden(teacher, student, eval_loader, mean_corr)

    print("\n[4] Shuffled-residual control")
    shuffled = NeuralResidualReconstructor(hidden, args.rank).to(device=device, dtype=torch.float32)
    train_shuffled(teacher, student, shuffled, train_loader, args.epochs, args.lr)
    shuffled = shuffled.to(torch.bfloat16)
    shuf_corr = lambda h: recon_forward(shuffled, h)
    shuf_loss, shuf_ppl = eval_ce(student, eval_loader, shuf_corr)
    shuf_closure = (wt_loss_q - shuf_loss) / wt_gap
    shuf_hidden = eval_hidden(teacher, student, eval_loader, shuf_corr)

    print("\n[5] C4 transfer evaluation")
    c4_loss_t, c4_ppl_t = eval_ce(teacher, c4_loader)
    c4_loss_q, c4_ppl_q = eval_ce(student, c4_loader)
    c4_loss_r, c4_ppl_r = eval_ce(student, c4_loader, normal_corr)
    c4_gap = c4_loss_q - c4_loss_t
    c4_closure = (c4_loss_q - c4_loss_r) / c4_gap if abs(c4_gap) > 1e-12 else float("nan")
    c4_hidden = eval_hidden(teacher, student, c4_loader, normal_corr)

    print("\n[6] Residual spectrum")
    spectrum = residual_spectrum(teacher, student, recon, eval_loader)

    print("\n[7] Runtime benchmark")
    prefill = benchmark_prefill(student, recon, seq_len=args.seq_len, batch=1)
    decode = benchmark_decode(student, recon, prompt_len=args.seq_len, batch=1)

    payload = {
        "model": args.model,
        "seed": args.seed,
        "packed_linear_modules": packed_count,
        "training": {
            "dataset": "WikiText-2 train",
            "train_blocks": args.train_blocks,
            "seq_len": args.seq_len,
            "epochs": args.epochs,
            "rank": args.rank,
            "objective": "normalized hidden-state MSE only",
        },
        "wikitext_validation": {
            "bf16": {"loss": wt_loss_t, "ppl": wt_ppl_t},
            "packed_w4": {"loss": wt_loss_q, "ppl": wt_ppl_q},
            "gnrq": {"loss": wt_loss_r, "ppl": wt_ppl_r, "gap_closure": wt_closure, "hidden": wt_hidden},
            "mean_residual_control": {"loss": mean_loss, "ppl": mean_ppl, "gap_closure": mean_closure, "hidden": mean_hidden},
            "shuffled_residual_control": {"loss": shuf_loss, "ppl": shuf_ppl, "gap_closure": shuf_closure, "hidden": shuf_hidden},
        },
        "c4_transfer": {
            "bf16": {"loss": c4_loss_t, "ppl": c4_ppl_t},
            "packed_w4": {"loss": c4_loss_q, "ppl": c4_ppl_q},
            "gnrq": {"loss": c4_loss_r, "ppl": c4_ppl_r, "gap_closure": c4_closure, "hidden": c4_hidden},
        },
        "spectrum": spectrum,
        "runtime": {"prefill_batch1": prefill, "decode_batch1": decode},
    }

    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    print(f"WikiText GNR-Q CE gap closure: {wt_closure*100:.2f}%")
    print(f"Mean control closure:          {mean_closure*100:.2f}%")
    print(f"Shuffle control closure:       {shuf_closure*100:.2f}%")
    print(f"C4 transfer closure:           {c4_closure*100:.2f}%")
    print(f"Decode overhead:               {decode['overhead_pct']:.2f}%")
    print("Saved:", out)


if __name__ == "__main__":
    main()
