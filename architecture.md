# FactoryFlow — Technical Architecture

## System overview

FactoryFlow is a four-layer system: edge data collection, GPU inference, multi-agent
orchestration, and autonomous procurement execution. Each layer is independently testable
and produces observable output — critical for a 24hr hackathon build.

---

## Layer 1 — Edge / sensor (MCP)

### Vibration sensor simulator

In production this is a MEMS accelerometer on a Raspberry Pi 4 sampling at 10kHz.
For the hackathon demo we synthesize bearing-fault FFT data in Python.

**Bearing fault physics (simplified):**
A healthy bearing's FFT shows broadband noise with no dominant peaks. A failing bearing
develops characteristic peaks at the Ball Pass Frequency Outer race (BPFO):

```
BPFO = (n/2) * RPM/60 * (1 - Bd/Pd * cos(α))
```

For a 6205 bearing at 1800 RPM: BPFO ≈ 85 Hz. We simulate fault by injecting a growing
sinusoidal component at 85 Hz whose amplitude scales with the `degradation_level` (0→1).

**Simulator states:**
```python
STATES = {
    "normal":           {"degradation": 0.05, "noise_scale": 1.0},
    "degrading":        {"degradation": 0.0→0.8, "noise_scale": 1.2},  # ramps over 2min
    "imminent_failure": {"degradation": 0.92, "noise_scale": 1.5},
}
```

**Output schema:**
```json
{
  "timestamp": "2026-05-09T14:32:01.123Z",
  "state_label": "degrading",
  "fft_window": [0.021, 0.019, ..., 0.847, ...],  // 512 float32 values
  "dominant_freq_hz": 85.3,
  "rms_velocity": 4.2
}
```

### MCP server

Transport: SSE (Server-Sent Events) over HTTP on port 8765.

Resources exposed:
- `sensor://vibration/stream` — subscribe to live FFT windows
- `sensor://vibration/latest` — single read of the most recent window
- `sensor://vibration/history` — last 60 windows as a batch (for model warm-up)

Tools exposed:
- `set_state(state: str)` — force simulator into named state (used by Gradio toggle)
- `get_stats()` — returns current RMS, dominant frequency, sample count

Test with: `mcp inspect http://localhost:8765`

---

## Layer 2 — AMD GPU inference

### ROCm setup

```bash
# Verify AMD GPU is visible
rocm-smi
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Install ROCm-compatible torch (adjust rocm version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

### MOMENT-1-large

**Model:** `AutonLab/MOMENT-1-large`
**Task:** Anomaly detection on time-series patches
**Input:** `(batch, n_channels, sequence_length)` — we use `(1, 1, 512)` per window
**Output:** Reconstruction error per patch → normalized to `anomaly_score ∈ [0, 1]`

**How it works (the intuition):**
MOMENT is trained to reconstruct "normal" time-series patterns. When it sees an anomaly
(the bearing fault peak), reconstruction error spikes because the pattern is outside its
normal distribution. Think of it like a spell-checker that flags unfamiliar words — the
"spell-checker" was trained on normal vibration, so the fault peak looks like a typo.

**Inference pipeline:**
```python
from momentfm import MOMENTPipeline

model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={"task_name": "reconstruction"},
).to("cuda")  # AMD GPU via ROCm

def score_window(fft_window: np.ndarray) -> AnomalyResult:
    # 1. Reshape to (1, 1, 512)
    x = torch.tensor(fft_window, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
    # 2. Normalize (z-score per window)
    x = (x - x.mean()) / (x.std() + 1e-8)
    # 3. Run reconstruction
    with torch.no_grad():
        output = model(x)
    # 4. Reconstruction error → anomaly score
    recon_error = torch.nn.functional.mse_loss(output.reconstruction, x).item()
    anomaly_score = min(recon_error / CALIBRATION_MAX, 1.0)
    # 5. Estimate RUL from score trajectory (linear regression over last 10 scores)
    rul_hours = estimate_rul(anomaly_score)
    return AnomalyResult(score=anomaly_score, rul_hours=rul_hours, confidence=0.87)
```

**CALIBRATION_MAX:** Set to the 99th percentile reconstruction error on normal data
during warm-up (first 30 windows). Store as a module-level constant after warm-up.

**Fallback model:** `amazon/chronos-t5-small` — treats anomaly detection as a
forecasting task (high forecast error = anomaly). Slower but smaller VRAM footprint.

### Qwen3-8B via vLLM

Serve locally with vLLM on AMD GPU:
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --dtype float16 \
  --port 8000 \
  --device cuda
```

CrewAI connects to this as an OpenAI-compatible endpoint:
```python
from crewai import LLM
llm = LLM(
    model="openai/qwen3-8b",
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)
```

**Why Qwen3-8B over other models:**
- Native function/tool-calling support (critical for CrewAI tools)
- 32k context window (Engineer Agent output is long)
- Apache 2.0 license
- Unlocks the Qwen hackathon bonus prize (10M tokens per team member)

---

## Layer 3 — CrewAI multi-agent orchestration

### Agent definitions

**Engineer Agent**
```
Role: Senior Maintenance Engineer
Goal: Monitor real-time vibration sensor data and identify the specific replacement
      part needed when anomaly score exceeds the alert threshold.
Backstory: 15 years experience diagnosing CNC machine failures from vibration signatures.
           Expert in bearing fault frequencies and gear mesh analysis.
Tools: [sensor_tool, parts_lookup]
```

**Procurement Agent**
```
Role: Industrial Procurement Specialist
Goal: Source the identified part from the best available supplier, balancing price
      against delivery time given the machine's remaining useful life window.
Backstory: Specialized in industrial MRO procurement with access to 50+ supplier catalogs.
Tools: [apify_scraper, mindsdb_history_tool]
```

### Task flow

```
Task 1 (Engineer Agent):
  "Review the latest sensor reading. If anomaly_score > {ANOMALY_THRESHOLD},
   identify the failing component and the replacement part SKU.
   Output a JSON object: {part_sku, fault_type, anomaly_score, rul_hours, urgency}."

Task 2 (Procurement Agent, receives Task 1 output as context):
  "Given part SKU {part_sku} and RUL of {rul_hours} hours, find the cheapest supplier
   that can deliver before the predicted failure. Return:
   {selected_supplier, unit_price_usd, delivery_days, stock_status, purchase_url}."
```

### Parts lookup table (hardcoded for demo)

```python
FAULT_TO_PART = {
    "bearing_outer_race": {
        "sku": "SKU-BRG-6205",
        "description": "Deep groove ball bearing 6205-2RS",
        "typical_price_usd": 12.50,
    },
    "gear_mesh": {
        "sku": "SKU-GBX-HELICAL-32T",
        "description": "Helical gearbox pinion 32T module 2",
        "typical_price_usd": 89.00,
    },
    "imbalance": {
        "sku": "SKU-BAL-WEIGHT-KIT",
        "description": "Dynamic balancing weight kit",
        "typical_price_usd": 34.00,
    },
}
```

---

## Layer 4 — Auth and payments

### Proxlock integration

Proxlock is a physical + digital authorization layer. For the demo, we use the REST API
to check whether the current session user is authorized to approve procurement actions.

```python
import httpx

async def check_authorization(budget_amount_usd: float) -> AuthResult:
    response = await httpx.AsyncClient().post(
        "https://api.proxlock.io/v1/authorize",
        headers={"X-API-Key": os.environ["PROXLOCK_API_KEY"]},
        json={
            "device_id": os.environ["PROXLOCK_DEVICE_ID"],
            "action": "procurement_approval",
            "metadata": {"amount_usd": budget_amount_usd}
        }
    )
    data = response.json()
    return AuthResult(
        authorized=data["status"] == "approved",
        approver=data.get("approver_name", "unknown"),
        timestamp=data.get("approved_at", "")
    )
```

In `DEMO_MODE=true`, this function sleeps 3 seconds then returns a mock approval.
The 3-second delay makes it feel real in the demo.

### X402 payment execution

X402 is programmable payments infrastructure for agentic systems. The Procurement Agent
calls this after Proxlock authorization to execute the actual purchase.

```python
async def execute_payment(purchase_order: PurchaseOrder) -> PaymentResult:
    if os.environ.get("DEMO_MODE") == "true":
        await asyncio.sleep(1.5)
        return PaymentResult(
            transaction_id=f"X402-DEMO-{uuid4().hex[:8].upper()}",
            status="simulated",
            amount_usd=purchase_order.unit_price_usd,
        )
    # Real execution path
    response = await httpx.AsyncClient().post(
        "https://api.x402.xyz/v1/payments",
        headers={"Authorization": f"Bearer {os.environ['X402_API_KEY']}"},
        json={
            "merchant_id": os.environ["X402_MERCHANT_ID"],
            "amount": purchase_order.unit_price_usd,
            "currency": "USD",
            "metadata": {
                "part_sku": purchase_order.part_sku,
                "supplier": purchase_order.supplier_name,
                "purchase_url": purchase_order.purchase_url,
            }
        }
    )
    return PaymentResult(**response.json())
```

---

## MindsDB integration

MindsDB provides SQL+AI queries against the procurement history database. The Procurement
Agent uses it to check whether this part has been ordered before and what the lead time
was historically.

```sql
-- Example query the Procurement Agent runs via MindsDB
SELECT
    part_sku,
    AVG(actual_delivery_days) as avg_delivery,
    MIN(unit_price_usd) as best_price,
    COUNT(*) as order_count
FROM procurement_history
WHERE part_sku = 'SKU-BRG-6205'
  AND order_date > NOW() - INTERVAL '1 year'
GROUP BY part_sku;
```

For the demo, seed `procurement_history` with 3-5 rows of fake history so the
agent can say "we've ordered this bearing 4 times, average delivery 2.1 days."
That detail makes the demo feel production-ready.

---

## Gradio UI layout

```
┌──────────────────────────────────────────────────────────────┐
│  FactoryFlow  ·  Predictive Parts Agent                       │
├──────────────────────────────┬───────────────────────────────┤
│  SENSOR FEED                 │  ANOMALY INFERENCE            │
│  Live score chart (line)     │  Score gauge  |  RUL: 31h     │
│  Last 60 windows             │  Confidence: 87%              │
│  [Toggle: simulate failure]  │  Fault: bearing_outer_race    │
├──────────────────────────────┼───────────────────────────────┤
│  AGENT ACTIVITY LOG          │  PROCUREMENT RESULT           │
│  [Engineer] Anomaly at 0.87  │  Part: SKU-BRG-6205           │
│  [Engineer] Fault: bearing   │  Supplier: BearingPoint.com   │
│  [Procurement] Searching...  │  Price: $47.00 · Delivery: 2d │
│  [Procurement] Found 3 supp  │  Auth: ✓ Proxlock approved    │
│  [Procurement] Selecting...  │  Payment: ✓ X402 executed     │
│                              │  TX: X402-7F3A2B1C            │
├──────────────────────────────┴───────────────────────────────┤
│  [▶ Run agent cycle]   [⚡ Simulate failure]   Status: idle  │
└──────────────────────────────────────────────────────────────┘
```

---

## Performance targets

| Step | Target latency | Notes |
|---|---|---|
| Sensor window generation | 5s interval | Matches real RPi sampling cycle |
| MOMENT inference | <100ms | On MI300X; <500ms on T4 |
| Engineer Agent (Qwen3-8B) | <8s | Including tool calls |
| Apify scrape (cached) | <3s | First call may be 10-15s |
| Procurement Agent | <10s | Including Apify + MindsDB |
| Proxlock auth (demo) | 3s | Deliberate delay for effect |
| X402 payment (demo) | 1.5s | Deliberate delay for effect |
| **Total pipeline** | **~30s** | Target for "wow" demo moment |
