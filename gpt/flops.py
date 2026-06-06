#!/usr/bin/env python3
"""
Compute FLOPs and parameter count for a HuggingFace transformer model
directly from its config.json. Zero third-party dependencies.

Usage:
    python flops.py <config_path_or_hf_repo_id> [options]

Options:
    --seq-len N      Sequence length used for FLOPs accounting (default 2048).
    --batch N        Batch size (default 1).
    --tokens N       If given, also report training FLOPs C = 6 N D.
    --json           Emit raw JSON only.
    --plot           Sweep --seq-len range and print an ASCII log-log chart.
    --svg PATH       Also write an SVG plot to PATH. Implies --plot.
    --plot-min N     Min seq_len for the sweep (default 1).
    --plot-max N     Max seq_len for the sweep (default 1048576).
    --plot-points N  Number of log-spaced points (default 21).

Examples:
    python flops.py meta-llama/Llama-2-7b-hf
    python flops.py ./config.json --seq-len 4096
    python flops.py Qwen/Qwen2.5-7B --tokens 15e12        # training cost for 15T tokens
    python flops.py ./config.json --plot --svg out.svg    # prefill vs decode vs seq_len
"""

import argparse
import json
import math
import os
import sys
import urllib.request


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

# Model families whose MLP uses a gated-linear-unit (3 matmuls: gate, up, down)
# instead of the classic 2-matmul (up, down) MLP. Matched against HF model_type.
GLU_FAMILIES = {
    "llama", "mistral", "mixtral", "qwen2", "qwen2_moe", "qwen3", "qwen3_moe",
    "gemma", "gemma2", "gemma3", "phi3", "phi4", "olmo", "olmo2", "olmoe",
    "deepseek", "deepseek_v2", "deepseek_v3", "deepseek_r1", "starcoder2",
    "internlm2", "yi", "baichuan", "minicpm", "cohere", "command_r",
    "granite", "granitemoe", "stablelm", "persimmon",
}


def load_config(src: str) -> dict:
    """Load config.json from a local file/dir or a HuggingFace repo id."""
    if os.path.isfile(src):
        with open(src) as f:
            return json.load(f)
    if os.path.isdir(src):
        with open(os.path.join(src, "config.json")) as f:
            return json.load(f)
    # Treat as HF repo id: "org/name" or "org/name:revision"
    repo, _, rev = src.partition(":")
    url = f"https://huggingface.co/{repo}/resolve/{rev or 'main'}/config.json"
    req = urllib.request.Request(url, headers={"User-Agent": "flops.py/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def _get(cfg: dict, *keys, default=None):
    """Return the first non-null value among the given keys."""
    for k in keys:
        if cfg.get(k) is not None:
            return cfg[k]
    return default


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _attn_block(cfg: dict, d: int, h: int, h_kv: int, head_dim: int, seq_len: int):
    """Return (params_per_layer, fwd_flops_per_token_per_layer, attn_kind, dims).

    `dims` is a human-readable summary of the relevant dimensions. Picks one of
    three flavors based on the config:
      - "MLA"        : Multi-head Latent Attention (DeepSeek V2/V3/V4 etc.)
      - "GQA" / "MHA": standard path (GQA/MQA when h_kv < h, else MHA)
    """
    if "q_lora_rank" in cfg:
        # ---- MLA (low-rank Q, latent KV) ----
        q_lora  = cfg["q_lora_rank"]
        qk_rope = _get(cfg, "qk_rope_head_dim", default=0)
        qk_nope = _get(cfg, "qk_nope_head_dim", default=max(1, head_dim - qk_rope))
        v_head  = _get(cfg, "v_head_dim", default=qk_nope)
        # V3 names the KV latent explicitly; V4 instead uses (h_kv * head_dim).
        kv_lora = _get(cfg, "kv_lora_rank", default=h_kv * head_dim)
        o_lora  = _get(cfg, "o_lora_rank")
        q_dim   = qk_nope + qk_rope                       # full per-head Q width
        # parameter counts (per layer)
        q_p  = d * q_lora + q_lora * h * q_dim
        kv_p = d * (kv_lora + qk_rope) + kv_lora * h * (qk_nope + v_head)
        o_p  = (h * v_head) * o_lora + o_lora * d if o_lora else (h * v_head) * d
        params = q_p + kv_p + o_p
        # FLOPs: 2× linear projections + attention scores (QK^T + Attn·V)
        flops = 2 * params + 2 * seq_len * h * (q_dim + v_head)
        dims = (f"heads={h} q_lora={q_lora} kv_lora={kv_lora} "
                f"qk={qk_nope}+{qk_rope} v={v_head}"
                + (f" o_lora={o_lora}" if o_lora else ""))
        return params, flops, "MLA", dims
    # ---- standard MHA / GQA / MQA ----
    qkv_p = d * (h * head_dim + 2 * h_kv * head_dim)
    o_p   = (h * head_dim) * d
    params = qkv_p + o_p
    flops  = 2 * params + 4 * seq_len * h * head_dim
    kind = "GQA" if h_kv < h else "MHA"
    dims = f"heads(Q/KV)={h}/{h_kv} head_dim={head_dim}"
    return params, flops, kind, dims


def _unmodelled(cfg: dict) -> list:
    """Names of architectural features the script knows about but doesn't count."""
    notes = []
    if cfg.get("index_n_heads"):
        notes.append(f"index/sparse attention ({cfg['index_n_heads']} heads, top-{cfg.get('index_topk','?')})")
    if cfg.get("num_nextn_predict_layers", 0):
        notes.append(f"{cfg['num_nextn_predict_layers']} multi-token-prediction layer(s)")
    if cfg.get("num_hash_layers", 0):
        notes.append(f"{cfg['num_hash_layers']} hash-attention layer(s)")
    if cfg.get("sliding_window") and cfg.get("sliding_window") != cfg.get("max_position_embeddings"):
        notes.append(f"sliding-window attention (window={cfg['sliding_window']})")
    return notes


def analyze(cfg: dict, seq_len: int) -> dict:
    """Compute parameter count and forward FLOPs (per token and per sequence)."""
    # Unwrap nested configs from multimodal / VLM wrappers.
    if "hidden_size" not in cfg:
        for key in ("text_config", "llm_config"):
            if key in cfg:
                cfg = cfg[key]
                break

    d = _get(cfg, "hidden_size", "n_embd", "d_model")
    L = _get(cfg, "num_hidden_layers", "n_layer", "num_layers")
    h = _get(cfg, "num_attention_heads", "n_head")
    h_kv = _get(cfg, "num_key_value_heads", "num_kv_heads", default=h)
    d_ff = _get(cfg, "intermediate_size", "ffn_dim", "n_inner", default=4 * d)
    vocab = _get(cfg, "vocab_size")
    head_dim = _get(cfg, "head_dim", default=d // h)
    tie = _get(cfg, "tie_word_embeddings", default=False)

    mtype = cfg.get("model_type", "").lower()
    is_glu = (mtype in GLU_FAMILIES
              or _get(cfg, "hidden_act", default="").lower() in {"silu", "swiglu", "geglu"})

    # MoE: routed (top-k) + always-on shared experts contribute to per-token FLOPs.
    n_experts = _get(cfg, "num_experts", "num_local_experts", "n_routed_experts", default=1)
    n_active = _get(cfg, "num_experts_per_tok", "num_experts_per_token", "moe_topk", default=1)
    n_shared = _get(cfg, "n_shared_experts", "num_shared_experts", default=0) if n_experts > 1 else 0
    d_ff_expert = _get(cfg, "moe_intermediate_size", default=d_ff)
    d_ff_active = (n_active + n_shared) * d_ff_expert if n_experts > 1 else d_ff

    mlp_mm = 3 if is_glu else 2                           # gate+up+down vs up+down

    # Attention (MHA/GQA/MQA or MLA)
    attn_p, attn_flops_per_tok, attn_kind, attn_dims = _attn_block(cfg, d, h, h_kv, head_dim, seq_len)

    # MLP block
    mlp_flops_per_tok = 2 * mlp_mm * d * d_ff_active
    mlp_p_per_expert  = mlp_mm * d * d_ff_expert
    mlp_p             = (n_experts + n_shared) * mlp_p_per_expert
    router_p          = d * n_experts if n_experts > 1 else 0

    # Per-layer compute and parameters
    per_layer_flops = attn_flops_per_tok + mlp_flops_per_tok
    per_layer_p     = attn_p + mlp_p + router_p + 2 * d   # +2 layernorms

    # LM head + embeddings + final norm
    lm_head_flops = 2 * d * vocab
    fwd_flops_per_tok = L * per_layer_flops + lm_head_flops

    emb_p     = vocab * d
    lm_head_p = 0 if tie else vocab * d
    total_params = emb_p + L * per_layer_p + d + lm_head_p

    # KV cache scalars per token per layer (FP32 = 4 bytes each).
    if "q_lora_rank" in cfg:                              # MLA: latent + rope key
        kv_elems = _get(cfg, "kv_lora_rank", default=h_kv * head_dim) \
                   + _get(cfg, "qk_rope_head_dim", default=0)
    else:
        kv_elems = 2 * h_kv * head_dim                    # K + V

    return {
        "model_type": mtype or "(unknown)",
        "hidden_size": d, "num_layers": L,
        "num_heads": h, "num_kv_heads": h_kv, "head_dim": head_dim,
        "attn_kind": attn_kind, "attn_dims": attn_dims,
        "intermediate_size": d_ff, "vocab_size": vocab,
        "is_glu_mlp": is_glu,
        "num_experts": n_experts,
        "experts_per_token": n_active if n_experts > 1 else None,
        "shared_experts": n_shared if n_experts > 1 else None,
        "params_total": total_params,
        "fwd_flops_per_token": fwd_flops_per_tok,
        "fwd_flops_per_seq": fwd_flops_per_tok * seq_len,
        "seq_len": seq_len,
        "weight_bytes_fp32": 4 * total_params,
        "kv_bytes_fp32_per_seq": 4 * L * kv_elems * seq_len,
        "unmodelled": _unmodelled(cfg),
    }


def sweep(cfg: dict, seq_lens):
    """Run analyze() across a list of seq_lens.

    Returns a list of (seq_len, prefill_total_flops, decode_per_step_flops).
    The "prefill total" is a single forward over a sequence of that length,
    and the "decode per step" is one new token attending to seq_len of cache.
    """
    out = []
    for k in seq_lens:
        r = analyze(cfg, k)
        out.append((k, r["fwd_flops_per_seq"], r["fwd_flops_per_token"]))
    return out


def log_space(lo: int, hi: int, n: int):
    """Return n log-spaced integer seq_lens in [lo, hi] (deduped, sorted)."""
    lo = max(1, lo)
    hi = max(lo + 1, hi)
    n = max(2, n)
    return sorted({max(1, int(round(lo * (hi / lo) ** (i / (n - 1))))) for i in range(n)})


# ---------------------------------------------------------------------------
# Formatting + rendering
# ---------------------------------------------------------------------------

_SI_UNITS = [("Y", 1e24), ("Z", 1e21), ("E", 1e18), ("P", 1e15),
             ("T", 1e12), ("G", 1e9),  ("M", 1e6),  ("K", 1e3)]


def fmt(n: float) -> str:
    """Format a number with a SI suffix and 3 decimals (e.g. 1.500K)."""
    for suf, base in _SI_UNITS:
        if abs(n) >= base:
            return f"{n / base:.3f}{suf}"
    return f"{n:.0f}"


def fmt_axis(n: float) -> str:
    """Compact SI suffix, 3 significant digits, no trailing zeros
    (e.g. 1K / 1.5K / 100G / 1T / 524K)."""
    for suf, base in _SI_UNITS:
        if abs(n) >= base:
            return f"{n / base:.3g}{suf}"
    return f"{n:.3g}"


def ascii_plot(rows, width: int = 64, height: int = 18) -> str:
    """Render a log-log line chart of prefill vs decode-per-step into ASCII."""
    xs   = [r[0] for r in rows]
    ys_p = [r[1] for r in rows]
    ys_d = [r[2] for r in rows]
    log_x = [math.log10(x) for x in xs]
    series = [("prefill total  o", ys_p, "o"),
              ("decode / step  *", ys_d, "*")]

    all_y = [y for _, ys, _ in series for y in ys if y > 0]
    x_lo, x_hi = log_x[0], log_x[-1]
    y_lo = math.floor(math.log10(min(all_y)))
    y_hi = math.ceil(math.log10(max(all_y)))

    grid = [[" "] * width for _ in range(height)]

    # Walk each curve continuously across columns, interpolating in log-log.
    for _, ys, mark in series:
        log_y = [math.log10(y) for y in ys]
        for col in range(width):
            lx = x_lo + (x_hi - x_lo) * col / (width - 1)
            j = 0
            while j < len(log_x) - 2 and log_x[j + 1] < lx:
                j += 1
            t = (lx - log_x[j]) / (log_x[j + 1] - log_x[j])
            ly = log_y[j] + t * (log_y[j + 1] - log_y[j])
            row = height - 1 - int(round((ly - y_lo) / (y_hi - y_lo) * (height - 1)))
            if not (0 <= row < height):
                continue
            grid[row][col] = mark if grid[row][col] in (" ", mark) else "#"

    # Y-axis: only label rows whose integer decade differs from the row above,
    # so labels don't repeat. Other rows get blanks of the same width.
    label_w = max(len(fmt_axis(10.0 ** d)) for d in range(y_lo, y_hi + 1)) + 2
    lines = []
    last_decade = None
    for r in range(height):
        ly = y_hi - (y_hi - y_lo) * r / (height - 1)
        decade = int(round(ly))
        if decade != last_decade:
            lbl = fmt_axis(10.0 ** decade)
            last_decade = decade
        else:
            lbl = ""
        lines.append(f"{lbl:>{label_w - 1}} |" + "".join(grid[r]))
    lines.append(" " * label_w + "+" + "-" * width)

    # X-axis tick labels at quartiles
    xlbl = [" "] * (width + label_w + 1)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        col = label_w + 1 + int(frac * (width - 1))
        s = fmt_axis(10 ** (x_lo + (x_hi - x_lo) * frac))
        start = max(0, min(len(xlbl) - len(s), col - len(s) // 2))
        for i, ch in enumerate(s):
            xlbl[start + i] = ch
    lines.append("".join(xlbl))
    lines.append(" " * label_w + "  seq_len   (log-log axes)")
    lines.append(" " * label_w + "  legend: " +
                 "  ".join(name for name, _, _ in series) + "   # overlap")
    return "\n".join(lines)


def write_svg(path: str, rows, title: str = "FLOPs vs seq_len") -> None:
    """Render a log-log SVG with prefill and decode-per-step curves."""
    W, H = 760, 460
    L, R, T, B = 90, 30, 50, 60                       # margins
    pw, ph = W - L - R, H - T - B

    lx  = [math.log10(r[0]) for r in rows]
    yps = [r[1] for r in rows]
    yds = [r[2] for r in rows]
    lx_lo, lx_hi = math.floor(lx[0]), math.ceil(lx[-1])
    ly_lo = math.floor(math.log10(min(yps + yds)))
    ly_hi = math.ceil(math.log10(max(yps + yds)))

    def sx(v): return L + (v - lx_lo) / (lx_hi - lx_lo) * pw
    def sy(v): return T + ph - (v - ly_lo) / (ly_hi - ly_lo) * ph

    def polyline(ys, color):
        pts = " ".join(f"{sx(lx[i]):.1f},{sy(math.log10(y)):.1f}" for i, y in enumerate(ys))
        dots = "".join(
            f'<circle cx="{sx(lx[i]):.1f}" cy="{sy(math.log10(y)):.1f}" r="3" fill="{color}"/>'
            for i, y in enumerate(ys))
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.2"/>{dots}'

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'font-family="ui-monospace,Menlo,monospace" font-size="12">',
        f'<text x="{W/2}" y="24" text-anchor="middle" font-size="16" '
        f'font-weight="600">{title}</text>',
        f'<rect x="{L}" y="{T}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#888"/>',
    ]
    # X ticks: gridline (inside), short tick mark (outside), human-readable label.
    for d in range(lx_lo, lx_hi + 1):
        x = sx(d)
        parts.append(f'<line x1="{x}" y1="{T}" x2="{x}" y2="{T+ph}" stroke="#e5e5e5"/>')
        parts.append(f'<line x1="{x}" y1="{T+ph}" x2="{x}" y2="{T+ph+5}" stroke="#444"/>')
        parts.append(f'<text x="{x}" y="{T+ph+18}" text-anchor="middle">{fmt_axis(10**d)}</text>')
    # Y ticks
    for d in range(ly_lo, ly_hi + 1):
        y = sy(d)
        parts.append(f'<line x1="{L}" y1="{y}" x2="{L+pw}" y2="{y}" stroke="#e5e5e5"/>')
        parts.append(f'<line x1="{L-5}" y1="{y}" x2="{L}" y2="{y}" stroke="#444"/>')
        parts.append(f'<text x="{L-8}" y="{y+4}" text-anchor="end">{fmt_axis(10**d)}</text>')
    parts += [
        f'<text x="{L+pw/2}" y="{H-18}" text-anchor="middle">seq_len</text>',
        f'<text x="20" y="{T+ph/2}" text-anchor="middle" '
        f'transform="rotate(-90 20 {T+ph/2})">FLOPs</text>',
        polyline(yps, "#1f6feb"),
        polyline(yds, "#cf222e"),
    ]
    # legend
    bx, by = L + 14, T + 12
    parts += [
        f'<rect x="{bx}" y="{by}" width="200" height="46" fill="white" stroke="#888"/>',
        f'<line x1="{bx+8}" y1="{by+16}" x2="{bx+34}" y2="{by+16}" '
        f'stroke="#1f6feb" stroke-width="2.2"/>'
        f'<text x="{bx+40}" y="{by+20}">prefill total (per sequence)</text>',
        f'<line x1="{bx+8}" y1="{by+36}" x2="{bx+34}" y2="{by+36}" '
        f'stroke="#cf222e" stroke-width="2.2"/>'
        f'<text x="{bx+40}" y="{by+40}">decode FLOPs / step</text>',
        "</svg>",
    ]
    with open(path, "w") as f:
        f.write("\n".join(parts))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_summary(src: str, r: dict, batch: int, tokens: float | None) -> None:
    fwd_total = r["fwd_flops_per_seq"] * batch
    train_step = 3 * fwd_total

    print(f"Source            : {src}")
    print(f"Model type        : {r['model_type']}")
    print(f"Layers / d_model  : {r['num_layers']} / {r['hidden_size']}")
    print(f"Attention         : {r['attn_kind']}   {r['attn_dims']}")
    print(f"FFN d_ff          : {r['intermediate_size']}  GLU={r['is_glu_mlp']}")
    if r["num_experts"] > 1:
        sh = f" + {r['shared_experts']} shared" if r["shared_experts"] else ""
        print(f"MoE experts       : {r['num_experts']} routed, top-{r['experts_per_token']} active{sh}")
    print(f"Vocab             : {r['vocab_size']}")
    print(f"Params (total)    : {fmt(r['params_total'])}  ({r['params_total']:,})")
    if r["unmodelled"]:
        print(f"NOT counted       : " + "; ".join(r["unmodelled"]))
    print()
    print(f"Seq length        : {r['seq_len']}" + (f"    Batch: {batch}" if batch > 1 else ""))
    print(f"Fwd FLOPs / token : {fmt(r['fwd_flops_per_token'])}")
    print(f"Fwd FLOPs / seq   : {fmt(r['fwd_flops_per_seq'])}")
    if batch > 1:
        print(f"Fwd FLOPs (batch) : {fmt(fwd_total)}")
    print(f"Train FLOPs (1 step, 3x fwd): {fmt(train_step)}")
    w  = r["weight_bytes_fp32"]
    kv = r["kv_bytes_fp32_per_seq"] * batch
    print(f"Mem FP32 weights  : {fmt(w)}B")
    print(f"Mem FP32 KV cache : {fmt(kv)}B  (seq={r['seq_len']}, batch={batch})")
    print(f"Mem FP32 inference: {fmt(w + kv)}B")
    if tokens is not None:
        print(f"Train FLOPs ({tokens:.2e} tokens, 6ND): {fmt(6 * r['params_total'] * tokens)}")


def print_sweep(rows) -> None:
    lo, hi = rows[0][0], rows[-1][0]
    print()
    print(f"=== Sweep over seq_len  [{lo} .. {hi}], {len(rows)} points ===")
    print(f"{'seq_len':>10}  {'prefill total':>14}  {'decode/step':>12}")
    for k, p, d in rows:
        print(f"{k:>10d}  {fmt(p):>14}  {fmt(d):>12}")
    print()
    print(ascii_plot(rows))


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="Path to config.json, a model dir, or an HF repo id (org/name)")
    ap.add_argument("--seq-len", type=int, default=2048, help="Sequence length (default 2048)")
    ap.add_argument("--batch",   type=int, default=1,    help="Batch size (default 1)")
    ap.add_argument("--tokens",  type=float, default=None,
                    help="Training token count → report C = 6 N D")
    ap.add_argument("--json",    action="store_true", help="Emit raw JSON only")
    ap.add_argument("--plot",    action="store_true", help="Print an ASCII log-log sweep chart")
    ap.add_argument("--svg",     metavar="PATH", help="Write SVG plot; implies --plot")
    ap.add_argument("--plot-min",    type=int, default=1,         help="Sweep min seq_len")
    ap.add_argument("--plot-max",    type=int, default=1_048_576, help="Sweep max seq_len")
    ap.add_argument("--plot-points", type=int, default=21,        help="Sweep point count")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    r = analyze(cfg, args.seq_len)

    if args.json:
        fwd_total = r["fwd_flops_per_seq"] * args.batch
        out = dict(r)
        out["fwd_flops_total_batch"] = fwd_total
        out["train_flops_total_batch"] = 3 * fwd_total
        if args.tokens is not None:
            out["training_tokens"] = args.tokens
            out["train_flops_chinchilla"] = 6 * r["params_total"] * args.tokens
        print(json.dumps(out, indent=2))
        return

    print_summary(args.config, r, args.batch, args.tokens)

    if args.plot or args.svg:
        rows = sweep(cfg, log_space(args.plot_min, args.plot_max, args.plot_points))
        print_sweep(rows)
        if args.svg:
            title = f"{r['model_type']} ({fmt(r['params_total'])} params): prefill vs decode FLOPs"
            write_svg(args.svg, rows, title=title)
            print(f"\nSVG written: {args.svg}")


if __name__ == "__main__":
    sys.exit(main())
