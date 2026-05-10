# FactoryFlow — Claude Code Instructions

## Project identity

**FactoryFlow** is an autonomous predictive maintenance and parts procurement agent for
small-to-medium manufacturers. It monitors vibration sensor data in real time, detects
imminent machine failure using a time-series foundation model running on AMD GPU hardware,
and autonomously sources and pre-orders replacement parts — all without human intervention
until the final budget-authorization step.

**Hackathon:** AMD x LabLab.ai Developer Hackathon (May 2026)
**Build window:** 24 hours
**Demo target:** End-to-end live demo showing sensor → anomaly detection → procurement → payment

---

## Architecture at a glance

```
[Simulated RPi sensor]
       │  MCP server (SSE stream)
       ▼
[MindsDB connector]  ──────────────────────────────────────┐
       │                                                   │
       ▼                                                   │
[AMD MI300X / ROCm]                                        │
  MOMENT-1-large  ──► anomaly_score, rul_hours             │
  Qwen3-8B        ──► agent reasoning backbone             │
       │                                                   │
       ▼                                                   │
[CrewAI Orchestrator]                                      │
  Engineer Agent  ──► reads score, identifies part SKU     │
  Procurement Agent ─► Apify scrape, selects best price   ◄┘
       │
       ▼
[Proxlock]  ──► authorization gate (human-in-the-loop)
       │
       ▼
[X402 Payments]  ──► executes autonomous purchase
       │
       ▼
[Gradio HF Space]  ──► live demo UI (prize track)
```

---

## Repository structure — build this exactly

```
factoryflow/
├── CLAUDE.md                   ← you are here
├── docs/
│   ├── architecture.md         ← full technical reference
│   ├── build-plan.md           ← 24hr sprint breakdown
│   └── memory.md               ← project state tracker (update as you go)
├── src/
│   ├── sensor/
│   │   ├── simulator.py        ← generates synthetic vibration FFT data
│   │   └── mcp_server.py       ← MCP server streaming sensor events via SSE
│   ├── inference/
│   │   ├── model_loader.py     ← loads MOMENT-1-large via HF on ROCm
│   │   ├── anomaly_detector.py ← scores incoming windows, returns (score, rul)
│   │   └── rocm_check.py       ← verifies AMD GPU is visible, logs device info
│   ├── agents/
│   │   ├── engineer_agent.py   ← CrewAI Engineer Agent definition + tools
│   │   ├── procurement_agent.py← CrewAI Procurement Agent definition + tools
│   │   ├── orchestrator.py     ← CrewAI Crew wiring Engineer → Procurement
│   │   └── tools/
│   │       ├── sensor_tool.py  ← tool: read latest anomaly score from MCP stream
│   │       ├── parts_lookup.py ← tool: map anomaly type to part SKU
│   │       └── apify_scraper.py← tool: call Apify actor to scrape supplier prices
│   ├── auth/
│   │   ├── proxlock.py         ← Proxlock authorization gate integration
│   │   └── budget_config.py    ← budget thresholds, authorized user list
│   ├── payments/
│   │   └── x402_client.py      ← X402 payment execution (POST to payment endpoint)
│   ├── data/
│   │   └── mindsdb_connector.py← MindsDB SQL+AI queries for procurement history
│   └── demo/
│       ├── app.py              ← Gradio app (HF Space entry point)
│       ├── components.py       ← reusable Gradio UI blocks
│       └── demo_script.md      ← judge-facing demo walkthrough
├── tests/
│   ├── test_sensor.py
│   ├── test_anomaly.py
│   └── test_agents.py
├── requirements.txt
├── .env.example
├── Dockerfile                  ← for HF Space deployment
└── README.md
```

---

## Tech stack — locked decisions, do not change

| Layer | Tool / Library | Version / Notes |
|---|---|---|
| Anomaly detection | `AutonLab/MOMENT-1-large` | HF transformers, ROCm backend |
| Agent LLM | `Qwen/Qwen3-8B` | via HF or vLLM, AMD GPU |
| Agent framework | `crewai` | ≥0.80.0 |
| MCP transport | `mcp` Python SDK | SSE transport |
| Sensor simulation | Custom Python | numpy FFT synthesis |
| Procurement scraping | Apify Python client | `apify-client` |
| Auth gate | Proxlock SDK | See docs/architecture.md |
| Payments | X402 | REST calls via httpx |
| Data connector | MindsDB Python SDK | SQL+AI queries |
| Demo UI | `gradio` | ≥4.0, HF Space compatible |
| GPU runtime | AMD ROCm | `torch` with ROCm wheels |
| Python | 3.11 | |

---

## Environment variables required

Create `.env` from `.env.example`. Every key listed here must be present or the app crashes
with a clear error message — never silently fall back to a mock.

```
# AMD / HF
HF_TOKEN=                   # Hugging Face token for gated model access
AMD_DEVICE=cuda             # or 'cpu' for local dev without GPU

# CrewAI / Qwen
OPENAI_API_BASE=            # point to vLLM serving Qwen3-8B, e.g. http://localhost:8000/v1
OPENAI_API_KEY=fake         # vLLM doesn't need a real key but CrewAI requires the var

# Apify
APIFY_API_TOKEN=

# Proxlock
PROXLOCK_API_KEY=
PROXLOCK_DEVICE_ID=

# X402
X402_API_KEY=
X402_MERCHANT_ID=

# MindsDB
MINDSDB_HOST=cloud.mindsdb.com
MINDSDB_USER=
MINDSDB_PASSWORD=

# Demo config
DEMO_MODE=true              # if true, skips real payment execution, logs instead
ANOMALY_THRESHOLD=0.75      # score above which the Engineer Agent fires
RUL_ALERT_HOURS=48          # RUL below which procurement is triggered
```

---

## Coding standards for this project

### Always do
- Type-annotate every function signature
- Use `structlog` for all logging — every log entry must include `component=` and
  `event=` keys so the Gradio demo can filter and display them cleanly
- Wrap all external API calls (Apify, Proxlock, X402, MindsDB) in `try/except` with
  explicit error messages — the demo must never crash silently
- Use `asyncio` for the MCP server and sensor stream — don't block the event loop
- Keep each source file under 200 lines — split if it grows beyond that

### Never do
- Never hardcode API keys or tokens in source files
- Never use `time.sleep()` in agent code — use `asyncio.sleep()`
- Never call the real X402 payment endpoint when `DEMO_MODE=true`
- Never import `openai` directly — route all LLM calls through `crewai`'s LLM config
- Never use `print()` — use `structlog` logger only

### Naming conventions
- Files: `snake_case.py`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Agent task names: descriptive strings (CrewAI uses them in traces)

---

## Build sequence — follow this order exactly

This is ordered for maximum demo-ability at each checkpoint. If time runs short, stop
at the highest checkpoint you've completed — each one produces a working demo.

### Checkpoint 1 — Sensor + MCP server (target: 2hrs)
1. `src/sensor/simulator.py` — emit synthetic bearing-fault FFT windows every 5s.
   Simulate three states: `normal`, `degrading` (score creeps 0.4→0.8 over 2 min),
   `imminent_failure` (score >0.85). State cycles automatically for demo purposes.
2. `src/sensor/mcp_server.py` — MCP server over SSE transport, exposes one resource:
   `sensor://vibration/stream` returning JSON `{timestamp, fft_window, state_label}`.
   Verify with `mcp inspect` before moving on.

### Checkpoint 2 — AMD GPU inference (target: 2hrs)
1. `src/inference/rocm_check.py` — print AMD GPU name, VRAM, ROCm version to stdout.
   This is a demo talking point — judges need to see the hardware being used.
2. `src/inference/model_loader.py` — load `AutonLab/MOMENT-1-large` with
   `torch_dtype=torch.float16` on AMD device. Cache the model object as a module-level
   singleton — do not reload on every inference call.
3. `src/inference/anomaly_detector.py` — accepts `fft_window: np.ndarray` (512 points),
   returns `AnomalyResult(score: float, rul_hours: float, confidence: float)`.
   MOMENT works in patch-based windows — chunk the 512-point input into 64-point patches.
   Log inference latency in ms on every call.

### Checkpoint 3 — CrewAI agents (target: 4hrs)
1. Configure Qwen3-8B as the CrewAI LLM — point `OPENAI_API_BASE` at vLLM serving the
   model. Use `crewai.LLM(model="openai/qwen3-8b", base_url=..., api_key="fake")`.
2. `src/agents/tools/sensor_tool.py` — CrewAI `@tool` that calls the MCP server and
   returns the latest `AnomalyResult` as a formatted string.
3. `src/agents/tools/parts_lookup.py` — maps fault signatures to SKUs. Hardcode a
   lookup table for the demo: bearing fault → `SKU-BRG-6205`, gear mesh fault →
   `SKU-GBX-HELICAL-32T`, imbalance → `SKU-BAL-WEIGHT-KIT`.
4. `src/agents/engineer_agent.py` — goal: "Monitor sensor data and identify which
   replacement part is needed if anomaly score exceeds threshold." Uses `sensor_tool`
   and `parts_lookup`. Output: structured dict `{part_sku, anomaly_score, rul_hours,
   urgency}`.
5. `src/agents/tools/apify_scraper.py` — calls Apify actor `apify/web-scraper` or a
   pre-built industrial parts actor. Input: part SKU + supplier list. Output: ranked
   list of `{supplier, price, delivery_days, url}`.
6. `src/agents/procurement_agent.py` — goal: "Find the best-priced supplier for the
   given part SKU, balancing price and delivery time given the RUL window." Uses
   `apify_scraper`. Output: `{selected_supplier, price, delivery_days, purchase_url}`.
7. `src/agents/orchestrator.py` — CrewAI `Crew` wiring Engineer → Procurement as a
   sequential process. The Engineer's output feeds the Procurement Agent's context.

### Checkpoint 4 — Auth + payments (target: 2hrs)
1. `src/auth/proxlock.py` — POST to Proxlock API to check authorization status for
   a given `device_id` and `budget_action`. Return `AuthResult(authorized: bool,
   approver: str, timestamp: str)`. In `DEMO_MODE`, return a mock approval after 3s.
2. `src/payments/x402_client.py` — POST purchase payload to X402. In `DEMO_MODE`,
   log the payload and return a mock `{transaction_id, status: "simulated"}`.

### Checkpoint 5 — Gradio demo UI (target: 3hrs)
1. `src/demo/app.py` — single-page Gradio app with four panels:
   - **Sensor feed**: live updating line chart of anomaly score over time
   - **Inference panel**: current `AnomalyResult` with score gauge + RUL countdown
   - **Agent activity log**: scrolling log of CrewAI agent actions (Engineer → Procurement)
   - **Procurement result**: supplier card with price, delivery, auth status, payment status
2. Wire a "Run agent cycle" button that triggers the full Crew.kickoff() and streams
   output back to the UI in real time using Gradio's `gr.State` + generator pattern.
3. Add a toggle: "Simulate imminent failure" — forces the sensor simulator into
   `imminent_failure` state so judges can trigger the full pipeline on demand.

### Checkpoint 6 — HF Space + README (target: 1hr)
1. `Dockerfile` — build image, install ROCm-compatible torch wheels, expose port 7860.
2. Deploy to HF Spaces (hardware: A10G or T4 if MI300X not available on Spaces).
   The demo will run on AMD cloud separately — Space is for the HF prize track.
3. `README.md` — include the demo talking points, architecture diagram link, and
   the AMD GPU inference evidence screenshot.

---

## Demo script (memorize this)

The judge demo is 3 minutes. Hit these beats in order:

1. **Hook (15s):** "Unplanned downtime costs manufacturers $50k per hour. FactoryFlow
   eliminates it by connecting a vibration sensor directly to autonomous procurement."

2. **Show the sensor (30s):** Point at the live anomaly score chart. "This is a bearing
   on a CNC spindle. MOMENT — a time-series foundation model — is running inference on
   AMD MI300X hardware right now, scoring every FFT window in under 50ms."

3. **Trigger failure (30s):** Hit "Simulate imminent failure." Watch the score climb.
   "The model detects the bearing's characteristic 3kHz fault frequency. RUL: 31 hours."

4. **Show agents (45s):** "The Engineer Agent identifies SKU-BRG-6205. The Procurement
   Agent — powered by Qwen3-8B — scrapes three suppliers via Apify and selects the
   fastest delivery within budget: $47 from BearingPoint, arrives in 18 hours."

5. **Auth + payment (30s):** "Proxlock gates the purchase — only authorized personnel
   can unlock the budget. Approved. X402 executes the programmable payment autonomously."

6. **Close (30s):** "From sensor spike to confirmed purchase order: 47 seconds. No
   human in the loop except the one authorization step. This is what the MCP-connected
   factory looks like."

---

## Known risks and mitigations

| Risk | Mitigation |
|---|---|
| MOMENT model too slow on available GPU | Fall back to `amazon/chronos-t5-small` — same interface, faster inference |
| Qwen3-8B OOM on single GPU | Use `Qwen/Qwen2.5-7B-Instruct` (slightly smaller, same tool-use quality) |
| Apify actor rate-limited | Cache the last scrape result for 60s; in DEMO_MODE serve hardcoded fixture data |
| Proxlock API not available | DEMO_MODE mock returns approval after 3s delay — looks identical in the UI |
| X402 integration incomplete | DEMO_MODE payment log is visually identical to real transaction in the UI |
| MCP SSE stream drops | Reconnect with exponential backoff; sensor_tool catches the exception |
| HF Space can't run ROCm | Separate the AMD MI300X inference from the HF Space — Space calls AMD cloud endpoint |

---

## Prize checklist — verify before submission

- [ ] AMD MI300X inference is demonstrably running (rocm_check.py output in README)
- [ ] Qwen3-8B is the agent backbone (show `OPENAI_API_BASE` pointing to Qwen vLLM)
- [ ] HF Space is deployed and has the demo live (needed for HF likes prize)
- [ ] X402 payment flow is wired (needed for X402 challenge prize)
- [ ] Gradio app is functional end-to-end with the "Simulate imminent failure" trigger
- [ ] Video demo is recorded and uploaded to LabLab submission
- [ ] Pitch deck covers: problem → solution → architecture → demo → market size
