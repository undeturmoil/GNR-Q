import gc
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchao.quantization import quantize_, Int4WeightOnlyConfig
from torchao.quantization import Int4TilePackedTo4dTensor

from gnrq_poc_spark import (
    NeuralResidualReconstructor,
    build_blocks,
    seed_all,
)

MODEL = "Qwen/Qwen3-4B-Base"
GROUP = 128
RANK = 128
TRAIN_BLOCKS = 256
EVAL_BLOCKS = 64
SEQ = 128
BATCH = 4
EPOCHS = 4
LR = 3e-4
CE_WEIGHT = 0.25
SEED = 42

device = torch.device("cuda")
seed_all(SEED)

def mem():
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated()

def mib(x):
    return x / 1024**2

def gib(x):
    return x / 1024**3

def freeze(m):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)

def recon_forward(recon, h):
    dtype = next(recon.parameters()).dtype
    return h + recon(h.to(dtype)).to(h.dtype)

@torch.no_grad()
def eval_hidden(teacher, student, recon, loader):
    mq = mc = cq = cc = 0.0
    n = 0

    for (ids,) in loader:
        ids = ids.to(device)

        ht = teacher.model(
            input_ids=ids,
            use_cache=False,
            return_dict=True
        ).last_hidden_state.float()

        hq = student.model(
            input_ids=ids,
            use_cache=False,
            return_dict=True
        ).last_hidden_state.float()

        hc = recon_forward(recon, hq)

        mq += F.mse_loss(hq, ht).item()
        mc += F.mse_loss(hc, ht).item()
        cq += F.cosine_similarity(hq, ht, dim=-1).mean().item()
        cc += F.cosine_similarity(hc, ht, dim=-1).mean().item()
        n += 1

    return {
        "mse_q": mq/n,
        "mse_c": mc/n,
        "cos_q": cq/n,
        "cos_c": cc/n,
    }

@torch.no_grad()
def eval_ce(model, loader, recon=None):
    losses = []

    for (ids,) in loader:
        ids = ids.to(device)

        h = model.model(
            input_ids=ids,
            use_cache=False,
            return_dict=True
        ).last_hidden_state

        if recon is not None:
            h = recon_forward(recon, h.float()).to(
                model.lm_head.weight.dtype
            )

        logits = model.lm_head(h)

        shift_logits = logits[:, :-1, :].float().reshape(
            -1, logits.size(-1)
        )
        shift_labels = ids[:, 1:].reshape(-1)

        loss = F.cross_entropy(
            shift_logits,
            shift_labels
        )
        losses.append(loss.item())

    loss = sum(losses) / len(losses)
    return loss, math.exp(loss)

print("=== PACKED W4 + GNR-Q FINAL RECONSTRUCTION ===")
print("GPU:", torch.cuda.get_device_name(0))
print("Torch:", torch.__version__)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

print("Preparing WikiText...")
train_ids = build_blocks(tok, "train", SEQ, TRAIN_BLOCKS)
eval_ids = build_blocks(tok, "validation", SEQ, EVAL_BLOCKS)

train_loader = DataLoader(
    TensorDataset(train_ids),
    batch_size=BATCH,
    shuffle=True
)

eval_loader = DataLoader(
    TensorDataset(eval_ids),
    batch_size=2,
    shuffle=False
)

gc.collect()
torch.cuda.empty_cache()
base = mem()

print("\nLoading BF16 teacher...")
teacher = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
).to(device)

freeze(teacher)

teacher_mem = mem() - base
print(f"BF16 teacher resident : {gib(teacher_mem):.3f} GiB")

print("\nLoading BF16 student...")
before_student = mem()

student = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
).to(device)

freeze(student)

student_bf16 = mem() - before_student
print(f"Student BF16 resident : {gib(student_bf16):.3f} GiB")

print("\nPacking student to real INT4/HQQ...")

cfg = Int4WeightOnlyConfig(
    group_size=GROUP,
    int4_packing_format="tile_packed_to_4d",
    int4_choose_qparams_algorithm="hqq",
)

quantize_(student.model.layers, cfg)

gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

student_w4 = mem() - before_student

packed_count = 0
for module in student.model.layers.modules():
    if isinstance(module, nn.Linear):
        if isinstance(module.weight, Int4TilePackedTo4dTensor):
            packed_count += 1

print(f"Packed Linear modules : {packed_count}")
print(f"Packed W4 resident    : {gib(student_w4):.3f} GiB")
print(
    f"Resident reduction    : "
    f"{(1-student_w4/student_bf16)*100:.2f}%"
)
print(
    f"Compression ratio     : "
    f"{student_bf16/student_w4:.2f}x"
)

hidden = teacher.config.hidden_size

recon = NeuralResidualReconstructor(
    hidden,
    RANK
).to(device=device, dtype=torch.float32)

recon_params = sum(p.numel() for p in recon.parameters())

print(
    f"GNR-Q params          : "
    f"{recon_params/1e6:.3f} M"
)
print(
    f"GNR-Q FP32 params     : "
    f"{recon_params*4/1024**2:.3f} MiB"
)

print("\n[0] Baseline packed W4 quality")

loss_t, ppl_t = eval_ce(teacher, eval_loader)
loss_q, ppl_q = eval_ce(student, eval_loader)

print(f"BF16        loss={loss_t:.5f} ppl={ppl_t:.3f}")
print(f"Packed W4   loss={loss_q:.5f} ppl={ppl_q:.3f}")

print("\n[1] Train GNR-Q on DETACHED packed hidden states")

opt = torch.optim.AdamW(
    recon.parameters(),
    lr=LR,
    weight_decay=0.01
)

for epoch in range(1, EPOCHS+1):
    recon.train()

    total = 0.0
    total_nmse = 0.0
    total_ce = 0.0
    steps = 0

    for (ids,) in train_loader:
        ids = ids.to(device)

        # packed model and teacher are inference-only
        with torch.no_grad():
            ht = teacher.model(
                input_ids=ids,
                use_cache=False,
                return_dict=True
            ).last_hidden_state.float()

            hq = student.model(
                input_ids=ids,
                use_cache=False,
                return_dict=True
            ).last_hidden_state.float()

        base_mse = F.mse_loss(
            hq, ht
        ).detach().clamp_min(1e-8)

        hc = hq + recon(hq)

        nmse = F.mse_loss(hc, ht) / base_mse

        logits = student.lm_head(
            hc.to(student.lm_head.weight.dtype)
        )

        ce = F.cross_entropy(
            logits[:, :-1, :].float().reshape(
                -1, logits.size(-1)
            ),
            ids[:, 1:].reshape(-1)
        )

        loss = nmse + CE_WEIGHT * ce

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            recon.parameters(), 1.0
        )
        opt.step()

        total += loss.item()
        total_nmse += nmse.item()
        total_ce += ce.item()
        steps += 1

    print(
        f"epoch {epoch:02d} "
        f"loss={total/steps:.5f} "
        f"nmse={total_nmse/steps:.5f} "
        f"ce={total_ce/steps:.5f}"
    )

recon.eval()

print("\n[2] Packed W4 + FP32 GNR-Q")

hm = eval_hidden(
    teacher,
    student,
    recon,
    eval_loader
)
print(hm)

loss_c, ppl_c = eval_ce(
    student,
    eval_loader,
    recon
)

gap = loss_q - loss_t
closure = (loss_q - loss_c) / gap

print(f"BF16            loss={loss_t:.5f} ppl={ppl_t:.3f}")
print(f"Packed W4       loss={loss_q:.5f} ppl={ppl_q:.3f}")
print(f"W4 + GNR-Q      loss={loss_c:.5f} ppl={ppl_c:.3f}")
print(f"CE gap closure  : {closure*100:.2f}%")

print("\n[3] Convert sidecar itself to BF16 for deployment")

del opt
gc.collect()
torch.cuda.empty_cache()

recon = recon.to(torch.bfloat16)

deploy_bytes = sum(
    p.numel() * p.element_size()
    for p in recon.parameters()
)

loss_b, ppl_b = eval_ce(
    student,
    eval_loader,
    recon
)

closure_b = (loss_q-loss_b)/gap

print(
    f"GNR-Q BF16 storage : "
    f"{deploy_bytes/1024**2:.3f} MiB"
)
print(
    f"W4 + BF16 GNR-Q    : "
    f"loss={loss_b:.5f} ppl={ppl_b:.3f}"
)
print(
    f"BF16 sidecar closure: "
    f"{closure_b*100:.2f}%"
)

payload = {
    "bf16_resident_gib": gib(student_bf16),
    "packed_w4_resident_gib": gib(student_w4),
    "memory_reduction_pct":
        (1-student_w4/student_bf16)*100,
    "compression_ratio":
        student_bf16/student_w4,
    "gnrq_params": recon_params,
    "gnrq_bf16_mib":
        deploy_bytes/1024**2,
    "loss_teacher": loss_t,
    "loss_w4": loss_q,
    "loss_w4_gnrq": loss_b,
    "ppl_teacher": ppl_t,
    "ppl_w4": ppl_q,
    "ppl_w4_gnrq": ppl_b,
    "gap_closure":
        closure_b,
    "hidden_metrics": hm,
}

torch.save(
    {
        "result": payload,
        "state_dict": recon.state_dict()
    },
    "packed_w4_gnrq_final.pt"
)

import json
with open(
    "packed_w4_gnrq_final.json",
    "w"
) as f:
    json.dump(payload, f, indent=2)

print("\nSaved packed_w4_gnrq_final.pt")
print("Saved packed_w4_gnrq_final.json")
print("=== DONE ===")
