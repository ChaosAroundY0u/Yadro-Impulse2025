import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

freq_sin = 0.5 # Hz
T = 1 # Sec
sampling_rate = 100 # Hz
t = np.arange(0, T + 1/sampling_rate, 1/sampling_rate)
t_decimated = t[::2]


# Генератор синуса    
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

def decimator(signal):
    return signal[::2]

def interpolator(x, y, x_new):
    #cubic spline
    n = len(x)
    h = np.diff(x)
    
    A = np.zeros((n, n))
    B = np.zeros(n)
    
    #natural cubic spline
    A[0, 0] = 1
    A[-1, -1] = 1
    
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        B[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
        
    # решение матричного уравнения встройкой
    c = np.linalg.solve(A, B)
    
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)
    
    for i in range(n - 1):
        b[i] = (y[i + 1] - y[i]) / h[i] - h[i] * (2 * c[i] + c[i + 1]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])
    
    y_new = np.zeros_like(x_new)
    for i in range(n - 1):
        mask = (x[i] <= x_new) & (x_new <= x[i + 1])
        dx = x_new[mask] - x[i]
        y_new[mask] = y[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
        
    return y_new

def MSE(inp, outp):
    h = inp - outp
    return np.mean(h**2)**0.5

mse = []
freqs = range(0, 51, 1)
for f in freqs:
    sinn = generate_sin(t, sampling_rate, freq = f) # 
    sinn_decimated = decimator(sinn)
    sinn_interpolated = interpolator(t_decimated, sinn_decimated, t)
    est = MSE(sinn, sinn_interpolated)
    mse.append(est)
    

plt.figure(1)
plt.grid(True)
plt.title("MSE(freq)")
plt.xlabel("Frequency, Hz")
plt.ylabel("MSE(normed)")

plt.plot(freqs, mse/max(mse), color = "blue", marker = ".")
plt.show()
