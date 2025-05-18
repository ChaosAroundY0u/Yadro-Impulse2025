import numpy as np
import matplotlib.pyplot as plt

freq_sin = 0.5 # Hz
T = 1 # Sec
sampling_rate = 100 # Hz
t = np.arange(0, T + 1/sampling_rate, 1/sampling_rate)

def generate_sin(t, sampling_rate, freq):
    T = 1 / sampling_rate
    omega = 2 * np.pi * freq
    A = np.array([[0, omega], [-omega, 0]])
    # Начальное состояние
    x = np.array([[1], [0]])
    # Матрица дискретизации
    Ad = expm(A * T)
    xl = np.zeros((2, len(t)))
    for i in range(len(t)):
        xl[:, i:i+1] = x
        x = Ad @ x
    return xl[0, :]

sinn = generate_sin(t, sampling_rate, freq_sin)
plt.figure(2)
plt.plot(t, sinn, color = "blue", label = "generated")
plt.legend()
plt.grid(True)
