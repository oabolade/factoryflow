import numpy as np

from src.sensor.simulator import BearingFaultSimulator
from src.inference.anomaly_detector import detect
sim = BearingFaultSimulator()
sim.set_state('imminent_failure')
w = np.array(sim.generate_window().fft_window)
print(detect(w).as_dict())