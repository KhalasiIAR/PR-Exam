import numpy as np
import matplotlib.pyplot as plt

# Frequency
freq = 0.2 * np.pi

# Angular frequency
w = 2 * np.pi * freq

# Amplitude
A = 1.25

# Values of variable argument
t = np.linspace(0, np.pi, 10000)

# Sin wave function
ft = A * np.sin(w * t)

# Sine wave plot
plt.plot(t, ft)
plt.title("Sine Wave")
plt.xlabel("Time")
plt.ylabel("Sine wave function")
plt.show()
