# FactoryFlow — Project Memory

> Claude Code reads this file to understand current build state.
> Update the Status column and Notes after completing each component.
> Never delete rows — mark them DONE or BLOCKED.

Last updated: 2026-05-10 — Checkpoint 6 deployed to HF Spaces; awaiting build + OpenAI secret

---

## Build status tracker

| Component | File | Status | Notes |
|---|---|---|---|
| ROCm check | `src/inference/rocm_check.py` | DONE | Detects rocm/cuda/mps/cpu; AMD_DEVICE=cpu forces CPU; printable summary via `python -m` |
| Sensor simulator | `src/sensor/simulator.py` | DONE | BearingFaultSimulator with normal/degrading/imminent_failure states; injects 85 Hz BPFO + 2x harmonic; 512-sample windows at 10 kHz |
| MCP server | `src/sensor/mcp_server.py` | DONE | FastMCP SSE on :8765; resources latest/stream/history, tools set_state/get_stats; background emit loop every 5s, 60-window history deque |
| Model loader | `src/inference/model_loader.py` | DONE | MOMENTPipeline singleton; fp16 on GPU, fp32 on CPU/MPS; logs load latency |
| Anomaly detector | `src/inference/anomaly_detector.py` | DONE | 512-pt window → reconstruction MSE; running calibration_max → score in [0,1]; heuristic RUL |
| LLM config | `src/agents/llm_config.py` | DONE | Single `get_llm()`; reads OPENAI_API_BASE for vLLM swap; default `gpt-4o-mini` |
| Sensor tool | `src/agents/tools/sensor_tool.py` | DONE | In-process simulator + anomaly_detector; `force_state()` helper for UI |
| Parts lookup tool | `src/agents/tools/parts_lookup.py` | DONE | Dominant-Hz band mapping (bearing/gear/imbalance) + urgency from RUL+score |
| Apify scraper tool | `src/agents/tools/apify_scraper.py` | DONE | DEMO_MODE fixture fallback; 60s in-memory cache; live mode via apify-client |
| Engineer Agent | `src/agents/engineer_agent.py` | DONE | role=Reliability Engineer; tools=read_sensor_anomaly + identify_part |
| Procurement Agent | `src/agents/procurement_agent.py` | DONE | Cheapest-within-RUL with critical→fastest override; tool=scrape_suppliers |
| Orchestrator | `src/agents/orchestrator.py` | DONE | Sequential Crew(eng→proc); `run_cycle(force_state=...)` returns merged JSON |
| MindsDB connector | `src/data/mindsdb_connector.py` | TODO | |
| Proxlock auth | `src/auth/proxlock.py` | DONE | Async `request_authorization`; budget check + auto-approve under $100; 3s mock in DEMO_MODE |
| Budget config | `src/auth/budget_config.py` | DONE | AUTO_APPROVE_LIMIT_USD=100, HARD_BUDGET_CEILING_USD=5000, AUTHORIZED_APPROVERS list |
| X402 payments | `src/payments/x402_client.py` | DONE | Async `execute_purchase(auth)`; PaymentResult dataclass; sim_* txn id in DEMO_MODE |
| Gradio UI | `src/demo/app.py` | DONE | 4-panel layout; gr.Timer auto-polls every 2s; run-cycle generator streams log + card |
| Demo components | `src/demo/components.py` | DONE | DemoState dataclass, gauge formatter, supplier-card markdown builder |
| Demo script | `src/demo/demo_script.md` | DONE | 3-min judge walkthrough + recovery cues |
| Dockerfile | `Dockerfile` | DONE | python:3.12-slim, CPU torch, momentfm --no-deps, port 7860, DEMO_MODE=true |
| HF Space | external | DONE | URL: https://huggingface.co/spaces/oabolade23/factoryflow (build in progress; needs OPENAI_API_KEY secret) |
| requirements.txt | `requirements.txt` | DONE | momentfm note added; install separately with --no-deps |
| .env.example | `.env.example` | DONE | Mirrored from env.example at repo root |
| README.md | `README.md` | DONE | HF Space frontmatter, talking points, AMD evidence section, smoke tests |

---

## Decisions log

Record every architectural or implementation decision made during the build.
This prevents re-litigating decisions under time pressure.

| Decision | Rationale | Date |
|---|---|---|
| Use MOMENT-1-large for anomaly detection | Handles anomaly detection directly without needing forecasting residuals | pre-build |
| Use Qwen3-8B as agent LLM via vLLM | Best tool-calling quality under 10B params; unlocks Qwen prize | pre-build |
| DEMO_MODE=true for Proxlock + X402 | Neither API confirmed available; mock is visually identical | pre-build |
| Cache Apify results for 60s | Prevents rate-limiting during rapid demo cycles | pre-build |
| MCP SSE on port 8765 | Avoids conflicts with vLLM (8000) and Gradio (7860) | pre-build |

---

## Known issues / blockers

Record blockers here as they arise. Include the error message and what you tried.

| Issue | Status | Resolution / Workaround |
|---|---|---|
| (none yet) | | |

---

## Environment status

Fill these in once verified:

```
AMD GPU detected:        [ ] yes  [ ] no   Device: ___________________
ROCm version:            ___________________
MOMENT model cached:     [ ] yes  [ ] no   Path: ___________________
Qwen3-8B vLLM serving:  [ ] yes  [ ] no   Endpoint: http://localhost:8000
Apify token valid:       [ ] yes  [ ] no
Proxlock API responding: [ ] yes  [ ] no
X402 API responding:     [ ] yes  [ ] no
MindsDB connected:       [ ] yes  [ ] no
HF Space URL:            ___________________
```

---

## Key constants (fill in during calibration)

```python
CALIBRATION_MAX = None          # set after warm-up in model_loader.py
ANOMALY_THRESHOLD = 0.75        # from .env
RUL_ALERT_HOURS = 48            # from .env
SENSOR_WINDOW_SIZE = 512        # FFT points per window
SENSOR_INTERVAL_SECONDS = 5     # emission rate
BEARING_FAULT_FREQ_HZ = 85.0    # BPFO for 6205 bearing at 1800 RPM
```

---

## API endpoints in use

| Service | Endpoint | Auth |
|---|---|---|
| vLLM (Qwen3-8B) | `http://localhost:8000/v1` | `OPENAI_API_KEY=fake` |
| MCP server | `http://localhost:8765` | none |
| Apify | `https://api.apify.com/v2` | `APIFY_API_TOKEN` |
| Proxlock | `https://api.proxlock.io/v1` | `PROXLOCK_API_KEY` |
| X402 | `https://api.x402.xyz/v1` | `X402_API_KEY` |
| MindsDB | `cloud.mindsdb.com` | `MINDSDB_USER` / `MINDSDB_PASSWORD` |
| HF Hub | `https://huggingface.co` | `HF_TOKEN` |

---

## Demo fixture data

If Apify is unavailable, use this fixture in `apify_scraper.py` when `DEMO_MODE=true`:

```python
APIFY_FIXTURE = {
    "SKU-BRG-6205": [
        {
            "supplier": "BearingPoint Industrial",
            "unit_price_usd": 47.00,
            "delivery_days": 2,
            "stock_status": "in_stock",
            "url": "https://bearingpoint.example.com/6205-2RS"
        },
        {
            "supplier": "GlobalBearings.com",
            "unit_price_usd": 39.50,
            "delivery_days": 5,
            "stock_status": "in_stock",
            "url": "https://globalbearings.example.com/catalog/6205"
        },
        {
            "supplier": "FastParts Express",
            "unit_price_usd": 62.00,
            "delivery_days": 1,
            "stock_status": "low_stock",
            "url": "https://fastparts.example.com/bearings/6205-2RS"
        }
    ]
}
```

MindsDB procurement history fixture:
```python
MINDSDB_FIXTURE = {
    "SKU-BRG-6205": {
        "avg_delivery_days": 2.1,
        "best_price_usd": 39.50,
        "order_count": 4,
        "last_ordered": "2025-11-14"
    }
}
```

---

## Submission links (fill in as ready)

- GitHub repo: https://github.com/oabolade/factoryflow
- HF Space: https://huggingface.co/spaces/oabolade23/factoryflow
- Demo video: ___________________
- LabLab submission: ___________________
- Pitch deck: ___________________
