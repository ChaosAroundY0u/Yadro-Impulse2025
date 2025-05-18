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
