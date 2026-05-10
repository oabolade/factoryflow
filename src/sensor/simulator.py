"""Synthetic bearing-fault vibration simulator for FactoryFlow.

Generates 512-sample time-domain windows at 10 kHz that mimic the vibration
signature of a 6205 ball bearing. State transitions between healthy and
imminent_failure inject a growing sinusoid at the BPFO frequency (~85 Hz at
1800 RPM), which MOMENT later flags as a reconstruction anomaly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import structlog

log = structlog.get_logger()

WINDOW_SIZE: int = 512
SAMPLE_RATE_HZ: float = 10_000.0
BEARING_FAULT_FREQ_HZ: float = 85.0
DEGRADATION_RAMP_PER_TICK: float = 0.01

State = Literal["normal", "degrading", "imminent_failure"]

STATE_PROFILES: dict[str, dict[str, float]] = {
    "normal": {"degradation_floor": 0.05, "noise_scale": 1.0},
    "degrading": {"degradation_floor": 0.0, "noise_scale": 1.2},
    "imminent_failure": {"degradation_floor": 0.92, "noise_scale": 1.5},
}


@dataclass
class SensorWindow:
    timestamp: str
    state_label: str
    fft_window: list[float]
    dominant_freq_hz: float
    rms_velocity: float

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "state_label": self.state_label,
            "fft_window": self.fft_window,
            "dominant_freq_hz": self.dominant_freq_hz,
            "rms_velocity": self.rms_velocity,
        }


@dataclass
class BearingFaultSimulator:
    state: State = "normal"
    degradation_level: float = 0.05
    sample_rate_hz: float = SAMPLE_RATE_HZ
    window_size: int = WINDOW_SIZE
    fault_freq_hz: float = BEARING_FAULT_FREQ_HZ
    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(seed=42)
    )
    _tick: int = 0

    def set_state(self, state: str) -> None:
        if state not in STATE_PROFILES:
            raise ValueError(
                f"unknown state '{state}'; must be one of {list(STATE_PROFILES)}"
            )
        previous = self.state
        self.state = state  # type: ignore[assignment]
        floor = STATE_PROFILES[state]["degradation_floor"]
        self.degradation_level = max(self.degradation_level, floor)
        if state == "normal":
            self.degradation_level = floor
        log.info(
            "state_change",
            component="sensor.simulator",
            previous=previous,
            new=state,
            degradation_level=round(self.degradation_level, 3),
        )

    def _advance_degradation(self) -> None:
        if self.state == "degrading":
            self.degradation_level = min(
                0.85, self.degradation_level + DEGRADATION_RAMP_PER_TICK
            )
        elif self.state == "imminent_failure":
            self.degradation_level = min(0.98, self.degradation_level + 0.005)
        # normal: leave at floor

    def inject_fault_peak(
        self, signal: np.ndarray, freq_hz: float, amplitude: float
    ) -> np.ndarray:
        t = np.arange(signal.size, dtype=np.float64) / self.sample_rate_hz
        phase = self._rng.uniform(0.0, 2 * math.pi)
        # Add fundamental + 2x harmonic — bearing faults excite harmonics too.
        peak = amplitude * np.sin(2 * math.pi * freq_hz * t + phase)
        peak += 0.4 * amplitude * np.sin(2 * math.pi * 2 * freq_hz * t + phase)
        return signal + peak

    def generate_window(self) -> SensorWindow:
        self._advance_degradation()
        self._tick += 1

        profile = STATE_PROFILES[self.state]
        noise_scale = profile["noise_scale"]

        # Broadband mechanical noise (healthy bearing baseline).
        signal = self._rng.normal(0.0, 0.05 * noise_scale, size=self.window_size)

        # Always include a small running-machine 30 Hz shaft component.
        t = np.arange(self.window_size, dtype=np.float64) / self.sample_rate_hz
        signal += 0.08 * np.sin(2 * math.pi * 30.0 * t)

        # Inject the fault peak scaled by current degradation.
        fault_amplitude = 0.6 * self.degradation_level
        if fault_amplitude > 0.01:
            signal = self.inject_fault_peak(
                signal, self.fault_freq_hz, fault_amplitude
            )

        # Impulsive transients spike during imminent failure.
        if self.state == "imminent_failure" and self._rng.random() < 0.5:
            idx = int(self._rng.integers(0, self.window_size))
            signal[idx] += self._rng.choice([-1.0, 1.0]) * 0.7

        dominant_hz, rms = _spectral_stats(signal, self.sample_rate_hz)

        window = SensorWindow(
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            state_label=self.state,
            fft_window=[float(x) for x in signal],
            dominant_freq_hz=float(dominant_hz),
            rms_velocity=float(rms),
        )
        log.debug(
            "window_emitted",
            component="sensor.simulator",
            tick=self._tick,
            state=self.state,
            degradation=round(self.degradation_level, 3),
            dominant_hz=round(dominant_hz, 1),
            rms=round(rms, 3),
        )
        return window


def _spectral_stats(signal: np.ndarray, sample_rate_hz: float) -> tuple[float, float]:
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate_hz)
    # Ignore DC component when picking dominant frequency.
    if spectrum.size > 1:
        dominant_hz = float(freqs[1 + int(np.argmax(spectrum[1:]))])
    else:
        dominant_hz = 0.0
    rms = float(np.sqrt(np.mean(signal**2)))
    return dominant_hz, rms
