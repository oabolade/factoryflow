---
title: FactoryFlow
emoji: ⚙️
colorFrom: orange
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# FactoryFlow

**Autonomous predictive maintenance and parts procurement for small manufacturers.**
Vibration sensor → MOMENT-1-large anomaly detection on AMD MI300X → CrewAI
agents (Qwen3-8B) → Proxlock authorization → X402 programmable payment.
End-to-end autonomous procurement with one human-in-the-loop step.

Built for the **AMD x LabLab.ai Developer Hackathon** (May 2026).

---

## What it does

Manufacturers lose **~$50k per hour** of unplanned machine downtime. FactoryFlow
turns a vibration sensor on the factory floor into an autonomous procurement
loop:

1. A simulated RPi sensor streams 512-point FFT windows over an MCP server
2. **MOMENT-1-large** scores each window for bearing/gear/imbalance faults on
   AMD GPU hardware (MI300X via ROCm)
3. The **Engineer Agent** maps the dominant fault frequency to a part SKU
4. The **Procurement Agent** (powered by **Qwen3-8B**) scrapes suppliers
   via Apify and selects the best price-vs-RUL trade-off
5. **Proxlock** gates the purchase with a human authorization step
6. **X402** executes the payment programmatically

From sensor spike to confirmed PO: under a minute.

---

## Architecture

```
[RPi sensor sim]
       │  MCP / SSE
       ▼
[AMD MI300X / ROCm]
   MOMENT-1-large  → anomaly_score, rul_hours, dominant_hz
   Qwen3-8B        → agent reasoning backbone
       │
       ▼
[CrewAI Crew]
   Engineer Agent      → identify SKU from fault signature
   Procurement Agent   → Apify scrape, pick best supplier
       │
       ▼
[Proxlock] ── human-in-the-loop authorization
       │
       ▼
[X402] ── autonomous programmable payment
       │
       ▼
[Gradio HF Space] — live demo UI (this Space)
```

---

## Prize tracks

- **AMD MI300X** — MOMENT inference + Qwen3-8B serving both run on AMD ROCm hardware
- **Qwen / vLLM** — Qwen3-8B is the CrewAI LLM backbone via vLLM (`OPENAI_API_BASE` swap)
- **Hugging Face** — this Docker Space; share the URL to drive likes
- **X402** — autonomous payment execution on agent decision
- **MCP** — sensor stream is exposed as an MCP server with SSE transport
- **Apify** — supplier discovery via the `apify/web-scraper` actor
- **MindsDB** — procurement history queried via SQL+AI

---

## Local run

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch                           # CPU/MPS for local dev
pip install -r requirements.txt
pip install --no-deps momentfm==0.1.4       # see note below

cp .env.example .env                        # add OPENAI_API_KEY
PYTHONPATH=. python -m src.demo.app
```

Open http://localhost:7860. The chart auto-polls every 2 seconds.

### Smoke tests by checkpoint

```bash
PYTHONPATH=. python -m src.inference.rocm_check       # device detection
PYTHONPATH=. python scripts/smoke.py                  # MOMENT inference
PYTHONPATH=. python -m src.agents.orchestrator        # full agent cycle
PYTHONPATH=. python -m src.sensor.mcp_server          # MCP SSE on :8765
```

---

## AMD GPU evidence

Run `python -m src.inference.rocm_check` on the AMD cloud box. Expected output:

```
torch:           2.x.x+rocm6.x
backend:         rocm
torch_device:    cuda
name:            AMD Instinct MI300X
vram_gb:         192.0
runtime_version: 6.x
✓ AMD ROCm GPU detected — ready for MI300X demo run.
```

A screenshot of this output is included in the LabLab submission.

---

## Demo mode vs live

`DEMO_MODE=true` (default) keeps the demo working without third-party API keys
by using fixtures for Apify, mock approvals for Proxlock, and simulated
transactions for X402. The UI is visually identical to live operation.

To run live, set `DEMO_MODE=false` and fill in: `APIFY_API_TOKEN`,
`PROXLOCK_API_KEY` + `PROXLOCK_DEVICE_ID`, `X402_API_KEY` + `X402_MERCHANT_ID`,
and (optional) `MINDSDB_*`.

---

## Notes

- **`momentfm` install:** the package on PyPI hard-pins old `numpy` /
  `transformers` that conflict with CrewAI and Gradio. Install it with
  `--no-deps` after the rest of `requirements.txt` — the actual code works
  fine on modern stacks.
- **HF Space hardware:** this Space runs MOMENT on CPU (~1–2s per window).
  The live judge demo runs on a separate AMD MI300X cloud box.

---

## Repo layout

See `CLAUDE.md` and `docs/architecture.md` for the full layout and per-file
responsibilities. `memory.md` tracks live build state.
