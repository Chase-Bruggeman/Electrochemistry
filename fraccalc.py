'''
Fractional Calculus for the Electrochemist
'''
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

def diffint(f: np.array, t: np.array, q = -1/2):
    '''
    Evaluates the derivative of fractional order `q` of `f` with respect to `t`.
    Adapted from Oldham & Spainer J. Electroanal. Chem. 26 (1970) 331-341.
    Variables in the code match those in Eq. 7 in the reference.
    If `q` is not specified, `diffint(f,t)` returns the semiintegral of `f` with respect to `t` (q = -1/2).
    '''
    N = len(f) # Number of elements in the function. 
    m = np.zeros_like(f)
    G = np.zeros_like(f)
    F = np.zeros((len(f),len(f)),dtype=f.dtype)
    T = np.zeros((len(t),len(t)),dtype=t.dtype)
    # Define the first terms of G, F, and T arrays
    G[0] = 1
    F[0,:] = f
    T[0,0] = 1/t[0]**0.5
    for j in range(1,N): # j is counted from 1 thru N-1
        G[j] = G[j-1] * (j - 1 - q) / j
    # Carry out the sum. The equation in the reference finds the derivative at f(t) by summing values over f(t), f(t-dt), f(t-2dt), etc.
    # This sum is equivalent to a sum with f[n], f[n-1], f[n-2], etc., where [n] is the index of the function.
    # The code below evaluates the sum at each index f[n], starting at f[0] and finishing at f[N-1].
    for j in range(1,N):
        T[j,j] = ((j+1)/t[j])**(q)
        F[j,j:] = f[:N-j]
    return G @ F @ T

def bl(f, x, i1, i2):
    '''
    Subtract from function `f` (of variable `x`) a best fit line from index `i1` to index `i2`
    '''
    ydata = f[i1:i2]
    xdata = x[i1:i2]
    def baseline(x, A, B):
        return A*x + B
    popt = curve_fit(baseline, xdata, ydata)[0]
    f2 = np.zeros_like(f)
    for k in range(len(f)):
        f2[k] = f[k] - interp1d(xdata, baseline(xdata,*popt), fill_value = 'extrapolate')(x[k])
    return f2

