# FactoryFlow — 3-Minute Judge Demo

Open the app at `http://localhost:7860` (or the HF Space URL). The sensor feed
and live inference panel start polling automatically.

## 0:00 — Hook (15s)
> "Unplanned downtime costs manufacturers $50k per hour. FactoryFlow eliminates
> it by connecting a vibration sensor directly to autonomous procurement —
> no humans in the loop except the final budget approval."

## 0:15 — Show the sensor (30s)
Point at the **anomaly score** chart and the **Live Inference** panel:
> "This is a bearing on a CNC spindle. MOMENT — a time-series foundation
> model — is running on AMD MI300X right now, scoring every FFT window in
> under 50ms. Right now we're in the `normal` band; score around 0.3."

(Optional flex) Pull up the terminal where `python -m src.inference.rocm_check`
showed the AMD device name and VRAM — that's the proof MOMENT is GPU-served.

## 0:45 — Trigger failure (30s)
Click **Imminent failure**. Watch the score climb past 0.85 within a few polls:
> "The model picks up the bearing's characteristic 85 Hz BPFO fault frequency.
> Score 0.92, RUL down to roughly 6 hours."

## 1:15 — Run the agent cycle (45s)
Click **▶ Run agent cycle**. The log streams in real time:
> "The Engineer Agent reads the latest window, identifies it as a bearing
> fault, and looks up SKU-BRG-6205. The Procurement Agent — Qwen3-8B on
> AMD — queries three suppliers via Apify and picks the best price-vs-RUL
> trade-off: BearingPoint at $47, two-day delivery."

## 2:00 — Auth + payment (30s)
The same cycle continues into Proxlock and X402 in the same log:
> "Proxlock authorizes — only the factory lead's identity unlocks the budget.
> Approved. X402 fires the programmable payment. Transaction ID `sim_…`,
> simulated for the demo but the call shape is identical to live."

## 2:30 — Close (30s)
> "Sensor spike to confirmed PO: under a minute. Every component runs on
> AMD: MOMENT for inference, Qwen3-8B for agent reasoning. The MCP-connected
> factory floor isn't a roadmap item — it's running on screen."

---

## Recovery cues if something goes wrong

- **Chart isn't updating.** Click **Normal** then **Imminent failure** to force
  a state change; that re-arms the simulator and the next poll will land.
- **Run agent cycle hangs.** The OpenAI/vLLM call is slow. Talk to the auth +
  payment slide while you wait — it's the same scripted beat.
- **Procurement returns no action.** The Engineer flagged the score as
  routine. Click **Imminent failure** again and re-run.
