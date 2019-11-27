import numpy as np

def ode4(dynamics, X_0, dt, control):
    dt2 = dt / 2
    dt6 = dt / 6

    dxdt = dynamics(X_0, control)  # , alphadot, control)
    x0t = X_0 + dt2 * dxdt

    dx0t = dynamics(x0t, control)  # , alphadot, control)
    x0t = X_0 + dt2 * dx0t

    dxmold = dynamics(x0t, control)  # , alphadot, control)
    x0t = X_0 + dt * dxmold

    dxm = dx0t + dxmold

    dx0t = dynamics(x0t, control)  # , alphadot, control)
    statesdot = dxdt + dx0t + 2 * dxm
    X = X_0 + dt6 * statesdot

    return X