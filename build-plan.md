# FactoryFlow — 24hr Build Plan

> **Rule:** At every checkpoint, the demo must work up to that layer.
> A partial demo that actually runs beats a full demo that crashes.

---

## Pre-build (30 min — do this first)

- [ ] Clone repo, create virtualenv, install base deps
- [ ] Create `.env` from `.env.example`, fill in all keys
- [ ] Run `src/inference/rocm_check.py` — confirm AMD GPU is visible
- [ ] Start vLLM serving Qwen3-8B — confirm OpenAI-compatible endpoint responds
- [ ] Create HF Space (empty, just a placeholder) to reserve the URL early
- [ ] Verify Apify account has credits and API token works

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/inference/rocm_check.py
curl http://localhost:8000/v1/models  # should return Qwen3-8B
```

---

## Hour 0–2 — Sensor layer + MCP server

**Goal:** Live vibration data streaming over MCP SSE.

Tasks:
- [ ] `src/sensor/simulator.py`
  - Implement `BearingFaultSimulator` class
  - `generate_window()` → returns 512-point FFT array based on current state
  - `set_state(state: str)` → switches between normal / degrading / imminent_failure
  - Degrading state: increment `degradation_level` by 0.01 every 5s (auto ramp)
  - Add `inject_fault_peak(fft, freq_hz, amplitude)` helper
- [ ] `src/sensor/mcp_server.py`
  - MCP server with SSE transport on port 8765
  - Register resource `sensor://vibration/stream`
  - Register tool `set_state`
  - Emit a new window every 5 seconds
- [ ] Smoke test: `mcp inspect http://localhost:8765` shows resources + tools
- [ ] Manual test: subscribe to stream, watch JSON windows arrive every 5s

**Done when:** You can run `mcp inspect`, subscribe to the stream, and see FFT windows
arriving. Hit the `set_state` tool to force `imminent_failure` and watch the output change.

---

## Hour 2–4 — AMD GPU inference

**Goal:** MOMENT scoring every incoming window. GPU evidence on screen.

Tasks:
- [ ] `src/inference/rocm_check.py`
  - Print: device name, VRAM total/free, ROCm version, torch version
  - Save output to `docs/gpu_evidence.txt` (include in README + submission)
- [ ] `src/inference/model_loader.py`
  - Download `AutonLab/MOMENT-1-large` (this will take time — start early)
  - Singleton pattern: `_model = None` at module level, `get_model()` lazy-loads
  - Move to `.cuda()` with `torch.float16`
  - Warm-up: run 5 dummy inferences to prime the GPU, calibrate `CALIBRATION_MAX`
- [ ] `src/inference/anomaly_detector.py`
  - `AnomalyResult` dataclass: `score, rul_hours, confidence, inference_ms`
  - `score_window(fft_window)` → `AnomalyResult`
  - RUL estimation: maintain a rolling deque of last 10 scores, fit linear regression,
    extrapolate to `ANOMALY_THRESHOLD` — that's the RUL in "windows", convert to hours
  - Log inference latency every call: `log.info("inference", ms=elapsed, score=score)`
- [ ] Unit test: `tests/test_anomaly.py`
  - Feed a normal window → expect score < 0.3
  - Feed an imminent_failure window → expect score > 0.75

**Done when:** `python -c "from src.inference.anomaly_detector import score_window; ..."` 
runs on GPU and returns a valid `AnomalyResult` in <100ms.

---

## Hour 4–8 — CrewAI agents

**Goal:** Engineer Agent reads anomaly, Procurement Agent finds a part price.
This is the longest block — protect it.

Tasks:
- [ ] `src/agents/tools/sensor_tool.py`
  - `@tool("get_latest_anomaly_reading")` — calls MCP server `sensor://vibration/latest`
    (via `anyio` HTTP client), returns formatted string with score + RUL
- [ ] `src/agents/tools/parts_lookup.py`
  - `@tool("lookup_replacement_part")` — takes `fault_type: str`, returns part SKU +
    description from hardcoded `FAULT_TO_PART` dict
- [ ] `src/agents/engineer_agent.py`
  - Define agent with Qwen3-8B as LLM
  - Define Task 1 (see architecture.md)
  - Smoke test: run agent alone, check it calls tools and returns structured output
- [ ] `src/agents/tools/apify_scraper.py`
  - `@tool("scrape_supplier_prices")` — calls Apify actor, returns top 3 results
  - **Cache layer:** store last result in a module-level dict keyed by SKU
    with a 60s TTL — avoids re-scraping during rapid demo cycles
  - `DEMO_MODE` fixture: return hardcoded 3-supplier list if env var is set
- [ ] `src/data/mindsdb_connector.py`
  - `query_procurement_history(part_sku)` → returns avg delivery days + order count
  - Seed with 3-5 fixture rows (see architecture.md)
- [ ] `src/agents/procurement_agent.py`
  - Define agent + Task 2
  - Smoke test: run agent with hardcoded Engineer output, check it returns supplier
- [ ] `src/agents/orchestrator.py`
  - `Crew` with `process=Process.sequential`, Engineer → Procurement
  - `run_cycle()` → returns `CycleResult(engineer_output, procurement_output)`
  - **Important:** set `verbose=True` on the Crew — the Gradio log panel needs
    the agent trace to stream into the UI

**Done when:** `python -c "from src.agents.orchestrator import run_cycle; print(run_cycle())"` 
produces a full `CycleResult` with a supplier recommendation.

---

## Hour 8–10 — Auth + payments

**Goal:** Proxlock gate works, X402 payment logs a transaction.

Tasks:
- [ ] `src/auth/budget_config.py`
  - `AUTHORIZED_USERS = ["demo_user"]`
  - `MAX_AUTO_APPROVE_USD = 500.0`
- [ ] `src/auth/proxlock.py`
  - `check_authorization(budget_amount_usd)` → `AuthResult`
  - DEMO_MODE: return mock approval after 3s `asyncio.sleep`
- [ ] `src/payments/x402_client.py`
  - `execute_payment(purchase_order)` → `PaymentResult`
  - DEMO_MODE: return mock result after 1.5s
- [ ] Wire auth → payment into orchestrator:
  - After Procurement Agent returns result, call Proxlock
  - If authorized, call X402
  - Return full `CycleResult` including `auth_result` and `payment_result`

**Done when:** Full pipeline runs end-to-end in DEMO_MODE, returns a transaction ID.
Run it once and confirm the log output matches what you want to show judges.

---

## Hour 10–13 — Gradio UI

**Goal:** Visual demo that judges can watch. This is what wins.

Tasks:
- [ ] `src/demo/app.py` skeleton — four-panel layout (see architecture.md)
- [ ] Panel 1 (Sensor): `gr.LinePlot` updating every 5s via `gr.Timer`
  - Pull latest 60 scores from a global `score_history` deque
  - Show a horizontal dashed line at `ANOMALY_THRESHOLD`
- [ ] Panel 2 (Inference): `gr.Number` gauge for current score, `gr.Textbox` for RUL
- [ ] Panel 3 (Agent log): `gr.Textbox` with `autoscroll=True`
  - Stream CrewAI verbose output by redirecting stdout to a queue
  - Gradio `gr.Timer` drains the queue into the textbox every 500ms
- [ ] Panel 4 (Procurement result): static cards, populated after cycle completes
- [ ] "Run agent cycle" button: triggers `run_cycle()` in a background thread,
  updates all panels as results arrive
- [ ] "Simulate failure" toggle: calls `set_state("imminent_failure")` on MCP server
- [ ] Status bar: shows idle / running / anomaly detected / procurement complete

**Polish touches (add if time permits):**
- Color the score gauge red when score > threshold, yellow when score > 0.5
- Show AMD GPU utilization % (poll `rocm-smi` every 5s, display in footer)
- Add a "timeline" of the last 5 agent cycles with timestamps

**Done when:** You can toggle "Simulate failure", watch the score climb, click
"Run agent cycle", and watch all four panels update through to a transaction ID.

---

## Hour 13–15 — Integration testing + polish

**Goal:** Eliminate all demo-breaking bugs. Run the full pipeline 5 times.

Tests to run:
- [ ] Full pipeline from cold start (fresh Python process)
- [ ] Simulate failure → agent cycle → transaction ID in <60s
- [ ] Toggle failure off → scores return to normal
- [ ] Crash Apify (disconnect network) → cached fixture serves cleanly
- [ ] Crash MCP server → sensor_tool returns graceful error message
- [ ] Run two agent cycles back-to-back (check for state corruption)

Edge cases to handle:
- [ ] MOMENT model returns NaN → catch, return score=0.0, log warning
- [ ] Qwen3-8B returns malformed JSON → retry once, then use fallback structured output
- [ ] Apify returns empty results → use fixture data, note in UI "using cached data"

---

## Hour 15–17 — HF Space deployment

**Goal:** Live public URL for the HF likes prize.

Tasks:
- [ ] `Dockerfile` — multi-stage build:
  ```dockerfile
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install -r requirements.txt
  COPY src/ src/
  COPY .env.example .env  # Space uses HF Secrets for real values
  EXPOSE 7860
  CMD ["python", "src/demo/app.py"]
  ```
- [ ] Set HF Space secrets for all env vars (Settings → Secrets)
- [ ] **Important:** HF Space hardware won't have AMD ROCm — configure Space to use
  CPU/T4 for the Gradio UI, and point `OPENAI_API_BASE` to your AMD cloud endpoint
  running Qwen3-8B. MOMENT inference calls AMD cloud endpoint too.
- [ ] Deploy, verify it loads, share the link
- [ ] Post the Space URL in the AMD Discord + LabLab community to gather likes

---

## Hour 17–20 — Demo recording + submission prep

**Goal:** Video demo recorded, pitch deck updated, submission draft ready.

Demo recording script:
1. Start with `rocm_check.py` output visible — proof of AMD GPU
2. Show Gradio UI idle with normal sensor data
3. Toggle "Simulate failure" — narrate as score climbs
4. Click "Run agent cycle" — narrate each agent step as it appears in the log
5. Point at supplier selection — explain price vs delivery tradeoff
6. Proxlock approval (3s pause — let it breathe)
7. X402 transaction ID appears — freeze on that screen for 3 seconds
8. Show HF Space URL — "live at huggingface.co/spaces/..."

Submission checklist:
- [ ] LabLab submission form filled (title, description, tags: AMD, MCP, CrewAI, Qwen)
- [ ] GitHub repo public with clear README
- [ ] HF Space deployed and accessible
- [ ] Video demo uploaded (Loom or YouTube unlisted, link in submission)
- [ ] Pitch deck (5 slides: problem / solution / architecture / demo / market)

---

## Hour 20–24 — Buffer + stretch goals

Use this time to fix anything that broke, polish the demo, or add stretch features:

**High value stretches (pick one):**
- Add a second fault type to the demo (gear mesh fault) — shows the system generalizes
- Add a "cost savings" counter to the UI: "Prevented downtime worth $50k"
- Add email notification via a CrewAI tool when anomaly is detected
- Add a simple `/health` endpoint to the FastAPI server for judge inspection

**Low risk stretches:**
- Add dark mode to Gradio UI
- Add AMD GPU utilization sparkline to the footer
- Record a second demo take with cleaner narration

---

## Emergency fallback plan

If the full pipeline isn't working at Hour 20, demo in this degraded order:

1. **Sensor + inference only:** Show MOMENT running on AMD GPU, anomaly score climbing.
   This alone is a strong demo of AMD compute usage.

2. **Sensor + inference + Engineer Agent only:** Show Qwen3-8B identifying the part.
   Skip Procurement Agent if Apify isn't cooperating.

3. **Mock everything except the UI:** Pre-record a JSON fixture for all agent outputs,
   play it back through the Gradio UI. The visual demo is identical — judges won't know.
   This is a last resort but it works.

**The non-negotiable:** AMD GPU inference must be live. Everything else can be mocked.
