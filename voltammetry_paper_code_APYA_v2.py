import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.special import iv
from math import gamma
F = 96485 # C/mol e-, Faraday's constant
R = 8.314 # J/K.mol, ideal gas constant
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['grid.linewidth'] = 0.1
plt.rcParams['font.size'] = 10
plt.rcParams['lines.markersize'] = 1
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = (3.35,3.35)

def CV(nu = 0.05, Ei = 0.5, Ef = -0.5, tstep = 0.02):
    '''
    Returns a potential function for cyclic voltammetry
    Inputs are:
        `nu` (scan rate, V/s)
        `Ei` (initial potential, V)
        `Ef` (final potential, V)
    Outputs are:
        `t` (time array, s, with increments of `tstep` seconds)
        `E` (potential waveform, V, made of a linear sweep from `Ei` to `Ef`, followed be a return sweep from `Ef` to `Ei`, all at scan rate `nu`)
    '''
    tf = 2*abs((Ei-Ef)/nu)
    t = np.linspace(0,tf,int(tf/tstep)+1)
    E = np.zeros_like(t)
    E[:len(t)//2] = Ei + np.sign(Ef-Ei)*abs(nu*t[:len(t)//2])
    E[len(t)//2:] = Ef - np.sign(Ef-Ei)*abs(nu*(t[len(t)//2:] - t[len(t)//2]))
    return t, E

def LSV(nu = 0.05, Ei = 0.5, Ef = -0.5, tstep = 0.02):
    '''
    Returns a potential function for linear sweep voltammetry
    Inputs are:
        `nu` (scan rate, V/s)
        `Ei` (initial potential, V)
        `Ef` (final potential, V)
    Outputs are:
        `t` (time array, s, with increments of `tstep` seconds)
        `E` (potential waveform, V, a linear sweep from `Ei` to `Ef` at scan rate `nu`)
    '''
    tf = abs((Ei-Ef)/nu)
    t = np.linspace(0,tf,int(tf/tstep)+1)
    E = np.zeros_like(t)
    E = Ei + np.sign(Ef-Ei)*abs(nu*t)
    return t, E

def QR(t, E_DC, n = 1, T = 298, A = 7.07e-6, c0 = 1, E0 = 0, alpha = 0.5, D = 1e-9, k0 = 1e-3, nharms = 3, dE = 0.035, freq = 20, Ru = 0, q = 1.0, Qdl = 0, dQdE = 0):
    '''
    Predicts the alternating current for single-step, quasireversible charge transfer, according to the scheme:
        A + ne- <-> B
    Required inputs are:
        `t` (array of time points, s)
        'E_DC' (applied linear sweep potential, V)
    Optional inputs are:
        `n` (number of electrons transferred per molecule)
        `T` (temperature, K)
        `A` (electrode surface area, m^2)
        `c0` (bulk concentration of starting material, mol.m^(-3))
        `E0` (thermodynamic potential of redox couple)
        `alpha` (electron transfer coefficient)
        `D` (diffusion coefficient, m^2.s^(-1))
        `k0` (heterogeneous electron transfer rate constant, m.s^(-1))
        `nharms` (number of harmonics in the experiment)
        'dE` (alternating potenial magnitude, V)
        `freq` (alternating potential frequency, Hz)
        `Ru` (uncompensated cell resistance, Ohm)
        `q` (phase angle of the constant phase element responsible for double layer charging current; q=1 corresponds to an ideal capacitor (90 degrees), q=0 to an ideal resistor)
        `Qdl` (effective capacitcance of the constant phase element; if q=1, the units of Qdl are F/m^2)
    Output is:
        `I' (total faradaic current, uA, as an array with length 2*nharms + 1. See paper for details.)
    '''
    Qdl = np.ones_like(E_DC)*(Qdl + dQdE*(E_DC - E_DC[0]))
    dt = t[1]-t[0]
    om = 2*np.pi*freq
    beta = 1 - alpha
    nFRT = n*F/(R*T)
    nFAcD = n*F*A*c0*D**0.5
    N = len(t)
    Psi_far = np.zeros((2*nharms+1,N), dtype = complex)
    I_dl = np.zeros((2*nharms+1,N), dtype = complex)
    I_meas = np.zeros((2*nharms+1,N), dtype = complex)
    E_array = np.zeros((3,N),dtype=complex)
    E_array[0,:] = np.ones(N)*dE/(2*1j)
    E_array[1,:] = E_DC
    E_array[2,:] = np.ones(N)*dE/(-2*1j)
    E_corr = E_array.copy()
    G_array = np.ones(N)
    for i in range(1,N):
        G_array[i] *= G_array[i-1]*(i-q-1)/i
    def Po(harm, n):
        if harm == 0:
            return 2*np.ones(n)/(np.pi**0.5)*((t[n]-t[:n])**0.5 - (t[n]-t[1:n+1])**0.5)
        else:
            return np.ones(1)/(harm*1j*om)**0.5
    def vo(n):
        return 1 + sum(Po(0,n)[:n-1]*Psi_far[nharms,1:n])
    def Pr(harm, n):
        if harm == 0:
            return 2*np.ones(n)/(np.pi**0.5)*((t[n]-t[:n])**0.5 - (t[n]-t[1:n+1])**0.5)
        elif harm != 0:
            return np.ones(1)/(harm*1j*om)**0.5
    def vr(n):
        return sum(Pr(0,n)[:n-1]*Psi_far[nharms,1:n])
    def Helement(mu,nu,n):
        return (iv(mu,-alpha*d)*np.exp(mu*1j*(theta - np.pi/2))*Po(nu,n)[-1] + np.exp(j)*iv(mu,beta*d)*np.exp(mu*1j*(theta - np.pi/2))*Pr(nu,n)[-1]) / (1 + np.exp(j)) 
    def Belement(mu,n):
        return (iv(mu,-alpha*d)*np.exp(mu*1j*(theta - np.pi/2))*vo(n) + np.exp(j)*iv(mu,beta*d)*np.exp(mu*1j*(theta - np.pi/2))*vr(n)) / (1 + np.exp(j))
    Hmat = np.zeros((2*nharms+1, 2*nharms+1),dtype = complex)
    Bmat = np.zeros(2*nharms+1, dtype = complex)
    for i in range(1,N):
        nu = (E_DC[i]-E_DC[i-1])/dt
        I_est = 2*I_meas[:,i-1] - I_meas[:,i-2]
        counter = 0
        while True:
            I_meas[:,i] = I_est
            E_corr[:,i] = E_array[:,i] - I_est[nharms-1:nharms+2]*Ru
            j = nFRT*(np.real(E_corr[1,i]) - E0)
            theta = np.angle(2j*E_corr[0,i])
            d = nFRT*2*abs(E_corr[0,i])
            for x in range(2*nharms + 1):
                for y in range(int((x+1)/2),int(nharms+x/2)+1):
                    Hmat[x,y] = Helement(y-x,nharms-y,i)
                Bmat[x] = Belement(nharms-x,i)
            eps = D**0.5 / (k0 * (np.exp(-alpha*j) + np.exp(beta*j)))
            Psi_far[:,i] = np.linalg.solve(eps*np.identity(2*nharms+1) + Hmat,-Bmat)
            if nharms > 0:
                I_dl[nharms-1,i] = Qdl[i]*A*(1j*om)**q*E_corr[0,i]
                I_dl[nharms+1,i] = Qdl[i]*A*(-1j*om)**q*E_corr[2,i]
            I_dl[nharms,i] = A*(Qdl[0]*nu*t[i]**(1-q)/(gamma(2-q)) - Ru/dt**q *(sum(G_array[:i]*I_meas[nharms,i:0:-1]*Qdl[i:0:-1])) + dQdE*nu**2*t[i]**(2-q)*gamma(3)/(gamma(3-q)))
            I_est = (I_est + nFAcD*Psi_far[:,i] + I_dl[:,i])/2
            if sum(abs(I_est - I_meas[:,i])) < sum(abs(I_est))/1000:
                break
            counter += 1
            if counter > 30:
                print(f'error - not converging at t = {t[i]}')
                break
        I_meas[:,i] = I_est
    I_meas[:,0] = I_meas[:,1]
    return I_meas*1e6

def FOC(t, E_DC, k = 1, K = 1, n = 1, T = 298, A = 7.07e-6, c0 = 1, E0 = 0, alpha = 0.5, D = 1e-9, k0 = 1e-3, nharms = 3, dE = 0.035, freq = 20, Ru = 0, q = 1.0, Qdl = 0, dQdE = 0):
    '''
    Predicts the alternating current for single-step, quasireversible charge transfer with a consecutive first-order reaction, according to the scheme:
        A + ne- <-> B
        B <-> C
    Required inputs are:
        `t` (array of time points, s)
        'E_DC' (applied linear sweep potential, V)
    Optional inputs are:
        'k' (kinetic rate constant, sum of forward and reverse rate constants, s^(-1))
        'K' (equilibrium constant)
        `n` (number of electrons transferred per molecule)
        `T` (temperature, K)
        `A` (electrode surface area, m^2)
        `c0` (bulk concentration of starting material, mol.m^(-3))
        `E0` (thermodynamic potential of redox couple)
        `alpha` (electron transfer coefficient)
        `D` (diffusion coefficient, m^2.s^(-1))
        `k0` (heterogeneous electron transfer rate constant, m.s^(-1))
        `nharms` (number of harmonics in the experiment)
        'dE` (alternating potenial magnitude, V)
        `freq` (alternating potential frequency, Hz)
        `Ru` (uncompensated cell resistance, Ohm)
        `q` (phase angle of the constant phase element responsible for double layer charging current; q=1 corresponds to an ideal capacitor (90 degrees), q=0 to an ideal resistor)
        `Qdl` (effective capacitcance of the constant phase element; if q=1, the units of Qdl are F/m^2)
    Output is:
        `I' (total faradaic current, uA, as an array with length 2*nharms + 1. See paper for details.)
    '''
    Qdl = np.ones_like(E_DC)*(Qdl + dQdE*(E_DC - E_DC[0]))
    dt = t[1]-t[0]
    om = 2*np.pi*freq
    beta = 1 - alpha
    nFRT = n*F/(R*T)
    nFAcD = n*F*A*c0*D**0.5
    N = len(t)
    Psi_far = np.zeros((2*nharms+1,N), dtype = complex)
    I_dl = np.zeros((2*nharms+1,N), dtype = complex)
    I_meas = np.zeros((2*nharms+1,N), dtype = complex)
    E_array = np.zeros((3,N),dtype=complex)
    E_array[0,:] = np.ones(N)*dE/(2*1j)
    E_array[1,:] = E_DC
    E_array[2,:] = np.ones(N)*dE/(-2*1j)
    E_corr = E_array.copy()
    G_array = np.ones(N)
    for i in range(1,N):
        G_array[i] *= G_array[i-1]*(i-q-1)/i
    def Po(harm, n):
        if harm == 0:
            return 2*np.ones(n)/(np.pi**0.5)*((t[n]-t[:n])**0.5 - (t[n]-t[1:n+1])**0.5)
        else:
            return np.ones(1)/(harm*1j*om)**0.5
    def vo(n):
        return 1 + sum(Po(0,n)[:n-1]*Psi_far[nharms,1:n])
    def Pr(harm, n):
        if harm == 0:
            return 2*np.ones(n)*((1+K*np.exp(-k*t[n-1::-1]))/(1+K))/(np.pi**0.5)*((t[n]-t[:n])**0.5 - (t[n]-t[1:n+1])**0.5)
        elif harm != 0:
            return np.ones(1)/(harm*1j*om)**0.5*(1/(1+K))*(1+K/(1+k/(harm*1j*om))**0.5)
    def vr(n):
        return sum(Pr(0,n)[:n-1]*Psi_far[nharms,1:n])
    def Helement(mu,nu,n):
        return (iv(mu,-alpha*d)*np.exp(mu*1j*(theta - np.pi/2))*Po(nu,n)[-1] + np.exp(j)*iv(mu,beta*d)*np.exp(mu*1j*(theta - np.pi/2))*Pr(nu,n)[-1]) / (1 + np.exp(j)) 
    def Belement(mu,n):
        return (iv(mu,-alpha*d)*np.exp(mu*1j*(theta - np.pi/2))*vo(n) + np.exp(j)*iv(mu,beta*d)*np.exp(mu*1j*(theta - np.pi/2))*vr(n)) / (1 + np.exp(j))
    Hmat = np.zeros((2*nharms+1, 2*nharms+1),dtype = complex)
    Bmat = np.zeros(2*nharms+1, dtype = complex)
    for i in range(1,N):
        nu = (E_DC[i]-E_DC[i-1])/dt
        I_est = 2*I_meas[:,i-1] - I_meas[:,i-2]
        counter = 0
        while True:
            I_meas[:,i] = I_est
            E_corr[:,i] = E_array[:,i] - I_est[nharms-1:nharms+2]*Ru
            j = nFRT*(np.real(E_corr[1,i]) - E0)
            theta = np.angle(2j*E_corr[0,i])
            d = nFRT*2*abs(E_corr[0,i])
            for x in range(2*nharms + 1):
                for y in range(int((x+1)/2),int(nharms+x/2)+1):
                    Hmat[x,y] = Helement(y-x,nharms-y,i)
                Bmat[x] = Belement(nharms-x,i)
            eps = D**0.5 / (k0 * (np.exp(-alpha*j) + np.exp(beta*j)))
            Psi_far[:,i] = np.linalg.solve(eps*np.identity(2*nharms+1) + Hmat,-Bmat)
            if nharms > 0:
                I_dl[nharms-1,i] = Qdl[i]*A*(1j*om)**q*E_corr[0,i]
                I_dl[nharms+1,i] = Qdl[i]*A*(-1j*om)**q*E_corr[2,i]
            I_dl[nharms,i] = A*(Qdl[0]*nu*t[i]**(1-q)/(gamma(2-q)) - Ru/dt**q *(sum(G_array[:i]*I_meas[nharms,i:0:-1]*Qdl[i:0:-1])) + dQdE*nu**2*t[i]**(2-q)*gamma(3)/(gamma(3-q)))
            I_est = (I_est + nFAcD*Psi_far[:,i] + I_dl[:,i]) / 2
            if sum(abs(I_est - I_meas[:,i])) < sum(abs(I_est))/1000:
                break
            counter += 1
            if counter > 30:
                print(f'error - not converging at t = {t[i]}')
                break
        I_meas[:,i] = I_est
    I_meas[:,0] = I_meas[:,1]
    return I_meas*1e6

def FOA(t, E_DC, k = 1, K = 1, n = 1, T = 298, A = 7.07e-6, c0 = 1, E0 = 0, alpha = 0.5, D = 1e-9, k0 = 1e-3, nharms = 3, dE = 0.035, freq = 20, Ru = 0, q = 1.0, Qdl = 0, dQdE = 0):
    '''
    Predicts the alternating current for single-step, quasireversible charge transfer with an antecedent first-order reaction, according to the scheme:
        A <-> B
        B + ne- <-> C
    Required inputs are:
        `t` (array of time points, s)
        'E_DC' (applied linear sweep potential, V)
    Optional inputs are:
        'k' (kinetic rate constant, sum of forward and reverse rate constants, s^(-1))
        'K' (equilibrium constant)
        `n` (number of electrons transferred per molecule)
        `T` (temperature, K)
        `A` (electrode surface area, m^2)
        `c0` (bulk concentration of starting material, mol.m^(-3))
        `E0` (thermodynamic potential of redox couple)
        `alpha` (electron transfer coefficient)
        `D` (diffusion coefficient, m^2.s^(-1))
        `k0` (heterogeneous electron transfer rate constant, m.s^(-1))
        `nharms` (number of harmonics in the experiment)
        'dE` (alternating potenial magnitude, V)
        `freq` (alternating potential frequency, Hz)
        `Ru` (uncompensated cell resistance, Ohm)
        `q` (phase angle of the constant phase element responsible for double layer charging current; q=1 corresponds to an ideal capacitor (90 degrees), q=0 to an ideal resistor)
        `Qdl` (effective capacitcance of the constant phase element; if q=1, the units of Qdl are F/m^2)
    Output is:
        `I' (total faradaic current, uA, as an array with length 2*nharms + 1. See paper for details.)
    '''
    Qdl = np.ones_like(E_DC)*(Qdl + dQdE*(E_DC - E_DC[0]))
    dt = t[1]-t[0]
    om = 2*np.pi*freq
    beta = 1 - alpha
    nFRT = n*F/(R*T)
    nFAcD = n*F*A*c0*D**0.5
    N = len(t)
    Psi_far = np.zeros((2*nharms+1,N), dtype = complex)
    I_dl = np.zeros((2*nharms+1,N), dtype = complex)
    I_meas = np.zeros((2*nharms+1,N), dtype = complex)
    E_array = np.zeros((3,N),dtype=complex)
    E_array[0,:] = np.ones(N)*dE/(2*1j)
    E_array[1,:] = E_DC
    E_array[2,:] = np.ones(N)*dE/(-2*1j)
    E_corr = E_array.copy()
    G_array = np.ones(N)
    for i in range(1,N):
        G_array[i] *= G_array[i-1]*(i-q-1)/i
    def Po(harm, n):
        if harm == 0:
            return 2*np.ones(n)*((K+np.exp(-k*t[n-1::-1]))/(1+K))/(np.pi**0.5)*((t[n]-t[:n])**0.5 - (t[n]-t[1:n+1])**0.5)
        else:
            return np.ones(1)/(harm*1j*om)**0.5*(1/(1+K))*(K+1/(1+k/(harm*1j*om))**0.5)
    def vo(n):
        return K/(1+K) + sum(Po(0,n)[:n-1]*Psi_far[nharms,1:n])
    def Pr(harm, n):
        if harm == 0:
            return 2*np.ones(n)/(np.pi**0.5)*((t[n]-t[:n])**0.5 - (t[n]-t[1:n+1])**0.5)
        elif harm != 0:
            return np.ones(1)/(harm*1j*om)**0.5
    def vr(n):
        return sum(Pr(0,n)[:n-1]*Psi_far[nharms,1:n])
    def Helement(mu,nu,n):
        return (iv(mu,-alpha*d)*np.exp(mu*1j*(theta - np.pi/2))*Po(nu,n)[-1] + np.exp(j)*iv(mu,beta*d)*np.exp(mu*1j*(theta - np.pi/2))*Pr(nu,n)[-1]) / (1 + np.exp(j)) 
    def Belement(mu,n):
        return (iv(mu,-alpha*d)*np.exp(mu*1j*(theta - np.pi/2))*vo(n) + np.exp(j)*iv(mu,beta*d)*np.exp(mu*1j*(theta - np.pi/2))*vr(n)) / (1 + np.exp(j))
    Hmat = np.zeros((2*nharms+1, 2*nharms+1),dtype = complex)
    Bmat = np.zeros(2*nharms+1, dtype = complex)
    for i in range(1,N):
        nu = (E_DC[i]-E_DC[i-1])/dt
        I_est = 2*I_meas[:,i-1] - I_meas[:,i-2]
        counter = 0
        while True:
            I_meas[:,i] = I_est
            E_corr[:,i] = E_array[:,i] - I_est[nharms-1:nharms+2]*Ru
            j = nFRT*(np.real(E_corr[1,i]) - E0)
            theta = np.angle(2j*E_corr[0,i])
            d = nFRT*2*abs(E_corr[0,i])
            for x in range(2*nharms + 1):
                for y in range(int((x+1)/2),int(nharms+x/2)+1):
                    Hmat[x,y] = Helement(y-x,nharms-y,i)
                Bmat[x] = Belement(nharms-x,i)
            eps = D**0.5 / (k0 * (np.exp(-alpha*j) + np.exp(beta*j)))
            Psi_far[:,i] = np.linalg.solve(eps*np.identity(2*nharms+1) + Hmat,-Bmat)
            if nharms > 0:
                I_dl[nharms-1,i] = Qdl[i]*A*(1j*om)**q*E_corr[0,i]
                I_dl[nharms+1,i] = Qdl[i]*A*(-1j*om)**q*E_corr[2,i]
            I_dl[nharms,i] = A*(Qdl[0]*nu*t[i]**(1-q)/(gamma(2-q)) - Ru/dt**q *(sum(G_array[:i]*I_meas[nharms,i:0:-1]*Qdl[i:0:-1])) + dQdE*nu**2*t[i]**(2-q)*gamma(3)/(gamma(3-q)))
            I_est = (I_est + nFAcD*Psi_far[:,i] + I_dl[:,i]) / 2
            if sum(abs(I_est - I_meas[:,i])) < sum(abs(I_est))/1000:
                break
            counter += 1
            if counter > 30:
                print(f'error - not converging at t = {t[i]}')
                break
        I_meas[:,i] = I_est
    I_meas[:,0] = I_meas[:,1]
    return I_meas*1e6

def FOAC(t, E_DC, k1 = 1, K1 = 1, k2 = 1, K2 = 1, n = 1, T = 298, A = 7.07e-6, c0 = 1, E0 = 0, alpha = 0.5, D = 1e-9, k0 = 1e-3, nharms = 3, dE = 0.035, freq = 20, Ru = 0, q = 1.0, Qdl = 0, dQdE = 0):
    '''
    Predicts the alternating current for single-step, quasireversible charge transfer with both antecedent and consecutive first-order reactions, according to the scheme:
        A <-> B
        B + ne- <-> C
        C <-> D
    Required inputs are:
        `t` (array of time points, s)
        'E_DC' (applied linear sweep potential, V)
    Optional inputs are:
        'k1' (kinetic rate constant for antecedent reaction, sum of forward and reverse rate constants, s^(-1))
        'K1' (equilibrium constant for antecedent reaction)
        'k2' (kinetic rate constant for consecutive reaction, sum of forward and reverse rate constants, s^(-1))
        'K2' (equilibrium constant for consecutive reaction)
        `n` (number of electrons transferred per molecule)
        `T` (temperature, K)
        `A` (electrode surface area, m^2)
        `c0` (bulk concentration of starting material, mol.m^(-3))
        `E0` (thermodynamic potential of redox couple)
        `alpha` (electron transfer coefficient)
        `D` (diffusion coefficient, m^2.s^(-1))
        `k0` (heterogeneous electron transfer rate constant, m.s^(-1))
        `nharms` (number of harmonics in the experiment)
        'dE` (alternating potenial magnitude, V)
        `freq` (alternating potential frequency, Hz)
        `Ru` (uncompensated cell resistance, Ohm)
        `q` (phase angle of the constant phase element responsible for double layer charging current; q=1 corresponds to an ideal capacitor (90 degrees), q=0 to an ideal resistor)
        `Qdl` (effective capacitcance of the constant phase element; if q=1, the units of Qdl are F/m^2)
    Output is:
        `I' (total faradaic current, uA, as an array with length 2*nharms + 1. See paper for details.)
    '''
    Qdl = np.ones_like(E_DC)*(Qdl + dQdE*(E_DC - E_DC[0]))
    dt = t[1]-t[0]
    om = 2*np.pi*freq
    beta = 1 - alpha
    nFRT = n*F/(R*T)
    nFAcD = n*F*A*c0*D**0.5
    N = len(t)
    Psi_far = np.zeros((2*nharms+1,N), dtype = complex)
    I_dl = np.zeros((2*nharms+1,N), dtype = complex)
    I_meas = np.zeros((2*nharms+1,N), dtype = complex)
    E_array = np.zeros((3,N),dtype=complex)
    E_array[0,:] = np.ones(N)*dE/(2*1j)
    E_array[1,:] = E_DC
    E_array[2,:] = np.ones(N)*dE/(-2*1j)
    E_corr = E_array.copy()
    G_array = np.ones(N)
    for i in range(1,N):
        G_array[i] *= G_array[i-1]*(i-q-1)/i
    def Po(harm, n):
        if harm == 0:
            return 2*np.ones(n)*((K1+np.exp(-k1*t[n-1::-1]))/(1+K1))/(np.pi**0.5)*((t[n]-t[:n])**0.5 - (t[n]-t[1:n+1])**0.5)
        else:
            return np.ones(1)/(harm*1j*om)**0.5*(1/(1+K1))*(K1+1/(1+k1/(harm*1j*om))**0.5)
    def vo(n):
        return K1/(1+K1) + sum(Po(0,n)[:n-1]*Psi_far[nharms,1:n])
    def Pr(harm, n):
        if harm == 0:
            return 2*np.ones(n)*((1+K2*np.exp(-k2*t[n-1::-1]))/(1+K2))/(np.pi**0.5)*((t[n]-t[:n])**0.5 - (t[n]-t[1:n+1])**0.5)
        elif harm != 0:
            return np.ones(1)/(harm*1j*om)**0.5*(1/(1+K2))*(1+K2/(1+k2/(harm*1j*om))**0.5)
    def vr(n):
        return sum(Pr(0,n)[:n-1]*Psi_far[nharms,1:n])
    def Helement(mu,nu,n):
        return (iv(mu,-alpha*d)*np.exp(mu*1j*(theta - np.pi/2))*Po(nu,n)[-1] + np.exp(j)*iv(mu,beta*d)*np.exp(mu*1j*(theta - np.pi/2))*Pr(nu,n)[-1]) / (1 + np.exp(j)) 
    def Belement(mu,n):
        return (iv(mu,-alpha*d)*np.exp(mu*1j*(theta - np.pi/2))*vo(n) + np.exp(j)*iv(mu,beta*d)*np.exp(mu*1j*(theta - np.pi/2))*vr(n)) / (1 + np.exp(j))
    Hmat = np.zeros((2*nharms+1, 2*nharms+1),dtype = complex)
    Bmat = np.zeros(2*nharms+1, dtype = complex)
    for i in range(1,N):
        nu = (E_DC[i]-E_DC[i-1])/dt
        I_est = 2*I_meas[:,i-1] - I_meas[:,i-2]
        counter = 0
        while True:
            I_meas[:,i] = I_est
            E_corr[:,i] = E_array[:,i] - I_est[nharms-1:nharms+2]*Ru
            j = nFRT*(np.real(E_corr[1,i]) - E0)
            theta = np.angle(2j*E_corr[0,i])
            d = nFRT*2*abs(E_corr[0,i])
            for x in range(2*nharms + 1):
                for y in range(int((x+1)/2),int(nharms+x/2)+1):
                    Hmat[x,y] = Helement(y-x,nharms-y,i)
                Bmat[x] = Belement(nharms-x,i)
            eps = D**0.5 / (k0 * (np.exp(-alpha*j) + np.exp(beta*j)))
            Psi_far[:,i] = np.linalg.solve(eps*np.identity(2*nharms+1) + Hmat,-Bmat)
            if nharms > 0:
                I_dl[nharms-1,i] = Qdl[i]*A*(1j*om)**q*E_corr[0,i]
                I_dl[nharms+1,i] = Qdl[i]*A*(-1j*om)**q*E_corr[2,i]
            I_dl[nharms,i] = A*(Qdl[0]*nu*t[i]**(1-q)/(gamma(2-q)) - Ru/dt**q *(sum(G_array[:i]*I_meas[nharms,i:0:-1]*Qdl[i:0:-1])) + dQdE*nu**2*t[i]**(2-q)*gamma(3)/(gamma(3-q)))
            I_est = (I_est + nFAcD*Psi_far[:,i] + I_dl[:,i]) / 2
            if sum(abs(I_est - I_meas[:,i])) < sum(abs(I_est))/1000:
                break
            counter += 1
            if counter > 30:
                print(f'error - not converging at t = {t[i]}')
                break
        I_meas[:,i] = I_est
    I_meas[:,0] = I_meas[:,1]
    return I_meas*1e6

def CAT(t, E_DC, k = 1e-1, n = 1, T = 298, A = 7.07e-6, c0 = 1, E0 = 0, alpha = 0.5, D = 1e-9, k0 = 1e-3, nharms = 3, dE = 0.035, freq = 20, Ru = 0, q = 1.0, Qdl = 0, dQdE = 0):
    '''
    Predicts the alternating current for single-step, quasireversible charge transfer followed by catalytic regeneration of starting material, according to the scheme:
        A + ne- <-> B
        B -> A
    Required inputs are:
        `t` (array of time points, s)
        'E_DC' (applied linear sweep potential, V)
    Optional inputs are:
        'k' (catalytic rate constant, s^(-1))
        `n` (number of electrons transferred per molecule)
        `T` (temperature, K)
        `A` (electrode surface area, m^2)
        `c0` (bulk concentration of starting material, mol.m^(-3))
        `E0` (thermodynamic potential of redox couple)
        `alpha` (electron transfer coefficient)
        `D` (diffusion coefficient, m^2.s^(-1))
        `k0` (heterogeneous electron transfer rate constant, m.s^(-1))
        `nharms` (number of harmonics in the experiment)
        'dE` (alternating potenial magnitude, V)
        `freq` (alternating potential frequency, Hz)
        `Ru` (uncompensated cell resistance, Ohm)
        `q` (phase angle of the constant phase element responsible for double layer charging current; q=1 corresponds to an ideal capacitor (90 degrees), q=0 to an ideal resistor)
        `Qdl` (effective capacitcance of the constant phase element; if q=1, the units of Qdl are F/m^2)
    Output is:
        `I' (total faradaic current, uA, as an array with length 2*nharms + 1. See paper for details.)
    '''
    Qdl = np.ones_like(E_DC)*(Qdl + dQdE*(E_DC - E_DC[0]))
    dt = t[1]-t[0]
    om = 2*np.pi*freq
    beta = 1 - alpha
    nFRT = n*F/(R*T)
    nFAcD = n*F*A*c0*D**0.5
    N = len(t)
    Psi_far = np.zeros((2*nharms+1,N), dtype = complex)
    I_dl = np.zeros((2*nharms+1,N), dtype = complex)
    I_meas = np.zeros((2*nharms+1,N), dtype = complex)
    E_array = np.zeros((3,N),dtype=complex)
    E_array[0,:] = np.ones(N)*dE/(2*1j)
    E_array[1,:] = E_DC
    E_array[2,:] = np.ones(N)*dE/(-2*1j)
    E_corr = E_array.copy()
    G_array = np.ones(N)
    for i in range(1,N):
        G_array[i] *= G_array[i-1]*(i-q-1)/i
    def Po(harm, n):
        if harm == 0:
            return 2*np.ones(n)*np.exp(-k*t[n-1::-1])/(np.pi**0.5)*((t[n]-t[:n])**0.5 - (t[n]-t[1:n+1])**0.5)
        else:
            return np.ones(1)/(harm*1j*om)**0.5 * 1/(1+k/(harm*1j*om))**0.5
    def vo(n):
        return 1 + sum(Po(0,n)[:n-1]*Psi_far[nharms,1:n])
    def Pr(harm, n):
        if harm == 0:
            return 2*np.ones(n)*np.exp(-k*t[n-1::-1])/(np.pi**0.5)*((t[n]-t[:n])**0.5 - (t[n]-t[1:n+1])**0.5)
        elif harm != 0:
            return np.ones(1)/(harm*1j*om)**0.5 * 1/(1+k/(harm*1j*om))**0.5
    def vr(n):
        return sum(Pr(0,n)[:n-1]*Psi_far[nharms,1:n])
    def Helement(mu,nu,n):
        return (iv(mu,-alpha*d)*np.exp(mu*1j*(theta - np.pi/2))*Po(nu,n)[-1] + np.exp(j)*iv(mu,beta*d)*np.exp(mu*1j*(theta - np.pi/2))*Pr(nu,n)[-1]) / (1 + np.exp(j)) 
    def Belement(mu,n):
        return (iv(mu,-alpha*d)*np.exp(mu*1j*(theta - np.pi/2))*vo(n) + np.exp(j)*iv(mu,beta*d)*np.exp(mu*1j*(theta - np.pi/2))*vr(n)) / (1 + np.exp(j))
    Hmat = np.zeros((2*nharms+1, 2*nharms+1),dtype = complex)
    Bmat = np.zeros(2*nharms+1, dtype = complex)
    for i in range(1,N):
        nu = (E_DC[i]-E_DC[i-1])/dt
        I_est = 2*I_meas[:,i-1] - I_meas[:,i-2]
        counter = 0
        while True:
            I_meas[:,i] = I_est
            E_corr[:,i] = E_array[:,i] - I_est[nharms-1:nharms+2]*Ru
            j = nFRT*(np.real(E_corr[1,i]) - E0)
            theta = np.angle(2j*E_corr[0,i])
            d = nFRT*2*abs(E_corr[0,i])
            for x in range(2*nharms + 1):
                for y in range(int((x+1)/2),int(nharms+x/2)+1):
                    Hmat[x,y] = Helement(y-x,nharms-y,i)
                Bmat[x] = Belement(nharms-x,i)
            eps = D**0.5 / (k0 * (np.exp(-alpha*j) + np.exp(beta*j)))
            Psi_far[:,i] = np.linalg.solve(eps*np.identity(2*nharms+1) + Hmat,-Bmat)
            if nharms > 0:
                I_dl[nharms-1,i] = Qdl[i]*A*(1j*om)**q*E_corr[0,i]
                I_dl[nharms+1,i] = Qdl[i]*A*(-1j*om)**q*E_corr[2,i]
            I_dl[nharms,i] = A*(Qdl[0]*nu*t[i]**(1-q)/(gamma(2-q)) - Ru/dt**q *(sum(G_array[:i]*I_meas[nharms,i:0:-1]*Qdl[i:0:-1])) + dQdE*nu**2*t[i]**(2-q)*gamma(3)/(gamma(3-q)))
            I_est = (I_est + nFAcD*Psi_far[:,i] + I_dl[:,i]) / 2
            if sum(abs(I_est - I_meas[:,i])) < sum(abs(I_est))/1000:
                break
            counter += 1
            if counter > 30:
                print(f'error - not converging at t = {t[i]}')
                break
        I_meas[:,i] = I_est
    I_meas[:,0] = I_meas[:,1]
    return I_meas*1e6

def MS(t, E_DC, n1 = 1, n2 = 1, T = 298, A = 7.07e-6, c0 = 1, E01 = 0, E02 = -0.1, alpha1 = 0.5, alpha2 = 0.5, D = 1e-9, k01 = 1e-3, k02 = 1e-3, nharms = 3, dE = 0.035, freq = 20, Ru = 0, q = 1.0, Qdl = 0, dQdE = 0):
    '''
    Predicts the alternating current for two-step, quasireversible charge transfer, according to the scheme:
        A + n1e- <-> B
        B + n2e- <-> C
    Required inputs are:
        `t` (array of time points, s)
        'E' (applied linear sweep potential, V)
    Optional inputs are:
        `n1` (number of electrons transferred per molecule in the first charge transfer step)
        `n2` (number of electrons transferred per molecule in the second charge transfer step)
        `T` (temperature, K)
        `A` (electrode surface area, m^2)
        `c0` (bulk concentration of starting material, mol.m^(-3))
        `E01` (thermodynamic potential of the first redox couple)
        `E02` (thermodynamic potential of the second redox couple)
        `alpha1` (electron transfer coefficient of the first redox couple)
        `alpha2` (electron transfer coefficient of the second redox couple)
        `D` (diffusion coefficient, m^2.s^(-1))
        `k01` (heterogeneous electron transfer rate constant in the first charge transfer step, m.s^(-1))
        `k02` (heterogeneous electron transfer rate constant in the second charge transfer step, m.s^(-1))
        `nharms` (number of harmonics in the experiment)
        'dE` (alternating potenial magnitude, V)
        `freq` (alternating potential frequency, Hz)
        `Ru` (uncompensated cell resistance, Ohm)
        `q` (phase angle of the constant phase element responsible for double layer charging current; q=1 corresponds to an ideal capacitor (90 degrees), q=0 to an ideal resistor)
        `Qdl` (effective capacitcance of the constant phase element; if q=1, the units of Qdl are F/m^2)
    Output is:
        `I' (total faradaic current, uA, as an array with length 2*nharms + 1. See paper for details.)
    '''
    Qdl = np.ones_like(E_DC)*(Qdl + dQdE*(E_DC - E_DC[0]))
    dt = t[1]-t[0]
    om = 2*np.pi*freq
    beta1 = 1 - alpha1
    beta2 = 1 - alpha2
    nFRT1 = n1*F/(R*T)
    nFRT2 = n2*F/(R*T)
    nFAcD1 = n1*F*A*c0*D**0.5
    nFAcD2 = n2*F*A*c0*D**0.5
    N = len(t)
    Psi1_far = np.zeros((2*nharms+1,N), dtype = complex)
    Psi2_far = np.zeros((2*nharms+1,N), dtype = complex)
    I_dl = np.zeros((2*nharms+1,N), dtype = complex)
    I_meas = np.zeros((2*nharms+1,N), dtype = complex)
    E_array = np.zeros((3,N),dtype=complex)
    E_array[0,:] = np.ones(N)*dE/(2*1j)
    E_array[1,:] = E_DC
    E_array[2,:] = np.ones(N)*dE/(-2*1j)
    E_corr = E_array.copy()
    G_array = np.ones(N)
    for i in range(1,N):
        G_array[i] *= G_array[i-1]*(i-q-1)/i
    def Po1(harm, n):
        if harm == 0:
            return 2*np.ones(n)/(np.pi**0.5)*((t[n]-t[:n])**0.5 - (t[n]-t[1:n+1])**0.5)
        else:
            return np.ones(1)/(harm*1j*om)**0.5
    def vo1(n):
        return 1 + sum(Po1(0,n)[:n-1]*Psi1_far[nharms,1:n])
    def Pr1(harm, n):
        if harm == 0:
            return 2*np.ones(n)/(np.pi**0.5)*((t[n]-t[:n])**0.5 - (t[n]-t[1:n+1])**0.5)
        else:
            return np.ones(1)/(harm*1j*om)**0.5
    def vr1(n):
        return sum(Pr1(0,n)[:n-1]*Psi1_far[nharms,1:n])
    def Pr2(harm, n):
        if harm == 0:
            return 2*np.ones(n)/(np.pi**0.5)*((t[n]-t[:n])**0.5 - (t[n]-t[1:n+1])**0.5)
        else:
            return np.ones(1)/(harm*1j*om)**0.5
    def vr2(n):
        return sum(Pr2(0,n)[:n-1]*Psi2_far[nharms,1:n])
    def Pp2(harm, n):
        if harm == 0:
            return 2*np.ones(n)/(np.pi**0.5)*((t[n]-t[:n])**0.5 - (t[n]-t[1:n+1])**0.5)
        else:
            return np.ones(1)/(harm*1j*om)**0.5
    def vp2(n):
        return sum(Pp2(0,n)[:n-1]*Psi2_far[nharms,1:n])
    def H1element(mu,nu,n):
        return (iv(mu,-alpha1*d1)*np.exp(mu*1j*(theta - np.pi/2))*Po1(nu,n)[-1] + np.exp(j1)*iv(mu,beta1*d1)*np.exp(mu*1j*(theta - np.pi/2))*Pr1(nu,n)[-1]) / (1 + np.exp(j1)) 
    def G1element(mu,nu,n):
        return (iv(mu,-alpha1*d1)*np.exp(mu*1j*(theta - np.pi/2))*0 + np.exp(j1)*iv(mu,beta1*d1)*np.exp(mu*1j*(theta - np.pi/2))*(n2/n1)*Pr2(nu,n)[-1]) / (1 + np.exp(j1)) 
    def B1element(mu,n):
        return (iv(mu,-alpha1*d1)*np.exp(mu*1j*(theta - np.pi/2))*vo1(n) + np.exp(j1)*iv(mu,beta1*d1)*np.exp(mu*1j*(theta - np.pi/2))*(vr1(n)-(n2/n1)*vr2(n))) / (1 + np.exp(j1))
    def H2element(mu,nu,n):
        return (iv(mu,-alpha2*d2)*np.exp(mu*1j*(theta - np.pi/2))*Pr2(nu,n)[-1] + np.exp(j2)*iv(mu,beta2*d2)*np.exp(mu*1j*(theta - np.pi/2))*Pp2(nu,n)[-1]) / (1 + np.exp(j2))
    def G2element(mu,nu,n):
        return (iv(mu,-alpha2*d2)*np.exp(mu*1j*(theta - np.pi/2))*(n1/n2)*Pr1(nu,n)[-1] + np.exp(j2)*iv(mu,beta2*d2)*np.exp(mu*1j*(theta - np.pi/2))*0) / (1 + np.exp(j2)) 
    def B2element(mu,n):
        return (iv(mu,-alpha2*d2)*np.exp(mu*1j*(theta - np.pi/2))*(vr2(n) - (n1/n2)*vr1(n)) + np.exp(j2)*iv(mu,beta2*d2)*np.exp(mu*1j*(theta - np.pi/2))*vp2(n)) / (1 + np.exp(j2))
    H1mat = np.zeros((2*nharms+1, 2*nharms+1),dtype = complex)
    G1mat = np.zeros((2*nharms+1, 2*nharms+1),dtype = complex)
    B1mat = np.zeros(2*nharms+1, dtype = complex)
    H2mat = np.zeros((2*nharms+1, 2*nharms+1),dtype = complex)
    G2mat = np.zeros((2*nharms+1, 2*nharms+1),dtype = complex)
    B2mat = np.zeros(2*nharms+1, dtype = complex)
    for i in range(1,N):
        nu = (E_DC[i]-E_DC[i-1])/dt
        I_est = 2*I_meas[:,i-1] - I_meas[:,i-2]
        counter = 0
        while True:
            I_meas[:,i] = I_est
            E_corr[:,i] = E_array[:,i] - I_est[nharms-1:nharms+2]*Ru
            j1 = nFRT1*(np.real(E_corr[1,i]) - E01)
            j2 = nFRT2*(np.real(E_corr[1,i]) - E02)
            theta = np.angle(2j*E_corr[0,i])
            d1 = nFRT1*2*abs(E_corr[0,i])
            d2 = nFRT2*2*abs(E_corr[0,i])
            for x in range(2*nharms + 1):
                for y in range(int((x+1)/2),int(nharms+x/2)+1):
                    H1mat[x,y] = H1element(y-x,nharms-y,i)
                    G1mat[x,y] = G1element(y-x,nharms-y,i)
                    H2mat[x,y] = H2element(y-x,nharms-y,i)
                    G2mat[x,y] = G2element(y-x,nharms-y,i)
                B1mat[x] = B1element(nharms-x,i)
                B2mat[x] = B2element(nharms-x,i)
            eps1 = D**0.5 / (k01 * (np.exp(-alpha1*j1) + np.exp(beta1*j1)))
            eps2 = D**0.5 / (k02 * (np.exp(-alpha2*j2) + np.exp(beta2*j2)))
            M1 = (eps1*np.identity(2*nharms+1) + H1mat)
            M2 = (eps2*np.identity(2*nharms+1) + H2mat)
            Psi1_far[:,i] = np.linalg.solve(np.linalg.inv(M2)@G2mat - np.linalg.inv(G1mat)@M1, np.linalg.inv(M2)@B2mat + np.linalg.inv(G1mat)@B1mat)
            Psi2_far[:,i] = np.linalg.solve(np.linalg.inv(M1)@G1mat - np.linalg.inv(G2mat)@M2, np.linalg.inv(M1)@B1mat + np.linalg.inv(G2mat)@B2mat)
            if nharms > 0:
                I_dl[nharms-1,i] = Qdl[i]*A*(1j*om)**q*E_corr[0,i]
                I_dl[nharms+1,i] = Qdl[i]*A*(-1j*om)**q*E_corr[2,i]
            I_dl[nharms,i] = A*(Qdl[0]*nu*t[i]**(1-q)/(gamma(2-q)) - Ru/dt**q *(sum(G_array[:i]*I_meas[nharms,i:0:-1]*Qdl[i:0:-1])) + dQdE*nu**2*t[i]**(2-q)*gamma(3)/(gamma(3-q)))
            I_est = (I_est + nFAcD1*Psi1_far[:,i] + nFAcD2*Psi2_far[:,i] + I_dl[:,i]) / 2
            if sum(abs(I_est - I_meas[:,i])) < sum(abs(I_est))/1000:
                break
            counter += 1
            if counter > 30:
                print(f'error - not converging at t = {t[i]}')
                break
        I_meas[:,i] = I_est
    I_meas[:,0] = I_meas[:,1]
    return I_meas*1e6

def EX(t, E_DC, n1 = 1, n2 = 1, k1 = 1e3, K1 = 1, k2 = 1e3, K2 = 1, k3 = 1e3, K3 = 1, kcat = 1e-1, T = 298, A = 7.07e-6, c0 = 1, E01 = 0, E02 = -0.1, alpha1 = 0.5, alpha2 = 0.5, D = 1e-9, k01 = 1e-3, k02 = 1e-3, nharms = 3, dE = 0.035, freq = 20, Ru = 0, q = 1.0, Qdl = 0, dQdE = 0):
    '''
    Predicts the alternating current for a catalytic cycle with two step charge transfer, according to the scheme:
    A <-> B
    B + n1e- <-> C
    C <-> D
    D + n2e- <-> E
    E <-> F
    F -> A
    Required inputs are:
        `t` (array of time points, s)
        'E' (applied linear sweep potential, V)
    Optional inputs are:
        `n1` (number of electrons transferred per molecule in the first charge transfer step)
        `n2` (number of electrons transferred per molecule in the second charge transfer step)
        'k1' (kinetic rate constant for reaction A <-> B, sum of forward and reverse rate constants, s^(-1))
        'K1' (equilibrium constant for reaction A <-> B)
        'k2' (kinetic rate constant for reaction C <-> D, sum of forward and reverse rate constants, s^(-1))
        'K2' (equilibrium constant for reaction C <-> D)
        'k3' (kinetic rate constant for reaction E <-> F, sum of forward and reverse rate constants, s^(-1))
        'K3' (equilibrium constant for reaction E <-> F)
        'kcat' (catalytic rate constant for reaction F -> A, s^(-1)))
        `T` (temperature, K)
        `A` (electrode surface area, m^2)
        `c0` (bulk concentration of starting material, mol.m^(-3))
        `E01` (thermodynamic potential of the B/C redox couple)
        `E02` (thermodynamic potential of the D/E redox couple)
        `alpha1` (electron transfer coefficient of the B/C redox couple)
        `alpha2` (electron transfer coefficient of the D/E redox couple)
        `D` (diffusion coefficient, m^2.s^(-1))
        `k01` (heterogeneous electron transfer rate constant in the first charge transfer step, m.s^(-1))
        `k02` (heterogeneous electron transfer rate constant in the second charge transfer step, m.s^(-1))
        `nharms` (number of harmonics in the experiment)
        'dE` (alternating potenial magnitude, V)
        `freq` (alternating potential frequency, Hz)
        `Ru` (uncompensated cell resistance, Ohm)
        `q` (phase angle of the constant phase element responsible for double layer charging current; q=1 corresponds to an ideal capacitor (90 degrees), q=0 to an ideal resistor)
        `Qdl` (effective capacitcance of the constant phase element; if q=1, the units of Qdl are F/m^2)
    Output is:
        `I' (total faradaic current, uA, as an array with length 2*nharms + 1. See paper for details.)
    '''
    Qdl = np.ones_like(E_DC)*(Qdl + dQdE*(E_DC - E_DC[0]))
    c = k3 + kcat
    kEF = K3/(1+K3)*k3
    kap = 0.5 * (c + (c**2 - 4*kEF*kcat)**0.5)
    gam = 0.5 * (c - (c**2 - 4*kEF*kcat)**0.5)
    dt = t[1]-t[0]
    om = 2*np.pi*freq
    beta1 = 1 - alpha1
    beta2 = 1 - alpha2
    nFRT1 = n1*F/(R*T)
    nFRT2 = n2*F/(R*T)
    nFAcD1 = n1*F*A*c0*D**0.5
    nFAcD2 = n2*F*A*c0*D**0.5
    N = len(t)
    Psi1_far = np.zeros((2*nharms+1,N), dtype = complex)
    Psi2_far = np.zeros((2*nharms+1,N), dtype = complex)
    I_dl = np.zeros((2*nharms+1,N), dtype = complex)
    I_meas = np.zeros((2*nharms+1,N), dtype = complex)
    E_array = np.zeros((3,N),dtype=complex)
    E_array[0,:] = np.ones(N)*dE/(2*1j)
    E_array[1,:] = E_DC
    E_array[2,:] = np.ones(N)*dE/(-2*1j)
    E_corr = E_array.copy()
    G_array = np.ones(N)
    for i in range(1,N):
        G_array[i] *= G_array[i-1]*(i-q-1)/i
    def Pb1(harm, n):
        if harm == 0:
            return 2*np.ones(n)*((K1+np.exp(-k1*t[n-1::-1]))/(1+K1))/(np.pi**0.5)*((t[n]-t[:n])**0.5 - (t[n]-t[1:n+1])**0.5)
        else:
            s = harm*1j*om
            return np.ones(1)/s**0.5*(1/(1+K1)) * (K1 + 1/(1+k1/s)**0.5)
    def vb1(n):
        return K1/(1+K1) + sum(Pb1(0,n)[:n-1]*Psi1_far[nharms,1:n])
    def Pb2(harm, n):
        if harm == 0:
            return 2*np.ones(n)*K1/(1+K1)*(1-(k1*kap)/((k1-gam)*(kap-gam))*np.exp(-gam*t[n-1::-1]) - (k1*gam)/((k1-kap)*(gam-kap))*np.exp(-kap*t[n-1::-1]) - (kap*gam)/((kap-k1)*(gam-k1))*np.exp(-k1*t[n-1::-1]))/(np.pi**0.5)*((t[n]-t[:n])**0.5 - (t[n]-t[1:n+1])**0.5)
        else:
            s = harm*1j*om
            return np.ones(1)/s**0.5*K1/(1+K1)*(1 - kap*k1/((kap-gam)*(k1-gam)*(1+gam/s)**0.5) - gam*k1/((gam-kap)*(k1-kap)*(1+kap/s)**0.5) - kap*gam/((kap-k1)*(gam-k1)*(1+k1/s)**0.5))
    def vb2(n):
        return sum(Pb2(0,n)[:n-1]*Psi2_far[nharms,1:n])
    def Pc1(harm, n):
        if harm == 0:
            return 2*np.ones(n)*((1+K2*np.exp(-k2*t[n-1::-1]))/(1+K2))/(np.pi**0.5)*((t[n]-t[:n])**0.5 - (t[n]-t[1:n+1])**0.5)
        else:
            s = harm*1j*om
            return np.ones(1)/s**0.5*(1/(1+K2)) * (1 + K2/(1+k2/s)**0.5)
    def vc1(n):
        return sum(Pc1(0,n)[:n-1]*Psi1_far[nharms,1:n])
    def Pc2(harm, n):
        if harm == 0:
            return 2*np.ones(n)*((1-np.exp(-k2*t[n-1::-1]))/(1+K2))/(np.pi**0.5)*((t[n]-t[:n])**0.5 - (t[n]-t[1:n+1])**0.5)
        else:
            s = harm*1j*om
            return np.ones(1)/s**0.5*(1/(1+K2)) * (1 - 1/(1+k2/s)**0.5)
    def vc2(n):
        return sum(Pc2(0,n)[:n-1]*Psi2_far[nharms,1:n])
    def Pd1(harm, n):
        if harm == 0:
            return 2*np.ones(n)*((K2-K2*np.exp(-k2*t[n-1::-1]))/(1+K2))/(np.pi**0.5)*((t[n]-t[:n])**0.5 - (t[n]-t[1:n+1])**0.5)
        else:
            s = harm*1j*om
            return np.ones(1)/s**0.5*(K2/(1+K2)) * (1 - 1/(1+k2/s)**0.5)
    def vd1(n):
        return sum(Pd1(0,n)[:n-1]*Psi1_far[nharms,1:n])
    def Pd2(harm, n):
        if harm == 0:
            return 2*np.ones(n)*((K2+np.exp(-k2*t[n-1::-1]))/(1+K2))/(np.pi**0.5)*((t[n]-t[:n])**0.5 - (t[n]-t[1:n+1])**0.5)
        else:
            s = harm*1j*om
            return np.ones(1)/s**0.5*(1/(1+K2)) * (K2 + 1/(1+k2/s)**0.5)
    def vd2(n):
        return sum(Pd2(0,n)[:n-1]*Psi2_far[nharms,1:n])
    def Pe2(harm, n):
        if harm == 0:
            return 2*np.ones(n)*((kap-kEF)/(kap-gam)*np.exp(-gam*t[n-1::-1]) - (gam-kEF)/(kap-gam)*np.exp(-kap*t[n-1::-1]))/(np.pi**0.5)*((t[n]-t[:n])**0.5 - (t[n]-t[1:n+1])**0.5)
        else:
            s = harm*1j*om
            return np.ones(1)/s**0.5*((kap-kEF)/((kap-gam)*(1+gam/s)**0.5) - (gam-kEF)/((kap-gam)*(1+kap/s)**0.5))
    def ve2(n):
        return sum(Pe2(0,n)[:n-1]*Psi2_far[nharms,1:n])
    def H1element(mu,nu,n):
        return (iv(mu,-alpha1*d1)*np.exp(mu*1j*(theta - np.pi/2))*Pb1(nu,n)[-1] + np.exp(j1)*iv(mu,beta1*d1)*np.exp(mu*1j*(theta - np.pi/2))*Pc1(nu,n)[-1]) / (1 + np.exp(j1)) 
    def G1element(mu,nu,n):
        return (iv(mu,-alpha1*d1)*np.exp(mu*1j*(theta - np.pi/2))*(n2/n1)*Pb2(nu,n)[-1] + np.exp(j1)*iv(mu,beta1*d1)*np.exp(mu*1j*(theta - np.pi/2))*(n2/n1)*Pc2(nu,n)[-1]) / (1 + np.exp(j1)) 
    def B1element(mu,n):
        return (iv(mu,-alpha1*d1)*np.exp(mu*1j*(theta - np.pi/2))*(vb1(n)-(n2/n1)*vb2(n)) + np.exp(j1)*iv(mu,beta1*d1)*np.exp(mu*1j*(theta - np.pi/2))*(vc1(n)-(n2/n1)*vc2(n))) / (1 + np.exp(j1))
    def H2element(mu,nu,n):
        return (iv(mu,-alpha2*d2)*np.exp(mu*1j*(theta - np.pi/2))*Pd2(nu,n)[-1] + np.exp(j2)*iv(mu,beta2*d2)*np.exp(mu*1j*(theta - np.pi/2))*Pe2(nu,n)[-1]) / (1 + np.exp(j2))
    def G2element(mu,nu,n):
        return (iv(mu,-alpha2*d2)*np.exp(mu*1j*(theta - np.pi/2))*(n1/n2)*Pd1(nu,n)[-1] + np.exp(j2)*iv(mu,beta2*d2)*np.exp(mu*1j*(theta - np.pi/2))*0) / (1 + np.exp(j2)) 
    def B2element(mu,n):
        return (iv(mu,-alpha2*d2)*np.exp(mu*1j*(theta - np.pi/2))*(vd2(n) - (n1/n2)*vd1(n)) + np.exp(j2)*iv(mu,beta2*d2)*np.exp(mu*1j*(theta - np.pi/2))*ve2(n)) / (1 + np.exp(j2))
    H1mat = np.zeros((2*nharms+1, 2*nharms+1),dtype = complex)
    G1mat = np.zeros((2*nharms+1, 2*nharms+1),dtype = complex)
    B1mat = np.zeros(2*nharms+1, dtype = complex)
    H2mat = np.zeros((2*nharms+1, 2*nharms+1),dtype = complex)
    G2mat = np.zeros((2*nharms+1, 2*nharms+1),dtype = complex)
    B2mat = np.zeros(2*nharms+1, dtype = complex)
    for i in range(1,N):
        nu = (E_DC[i]-E_DC[i-1])/dt
        I_est = 2*I_meas[:,i-1] - I_meas[:,i-2]
        counter = 0
        while True:
            I_meas[:,i] = I_est
            E_corr[:,i] = E_array[:,i] - I_est[nharms-1:nharms+2]*Ru
            j1 = nFRT1*(np.real(E_corr[1,i]) - E01)
            j2 = nFRT2*(np.real(E_corr[1,i]) - E02)
            theta = np.angle(2j*E_corr[0,i])
            d1 = nFRT1*2*abs(E_corr[0,i])
            d2 = nFRT2*2*abs(E_corr[0,i])
            for x in range(2*nharms + 1):
                for y in range(int((x+1)/2),int(nharms+x/2)+1):
                    H1mat[x,y] = H1element(y-x,nharms-y,i)
                    G1mat[x,y] = G1element(y-x,nharms-y,i)
                    H2mat[x,y] = H2element(y-x,nharms-y,i)
                    G2mat[x,y] = G2element(y-x,nharms-y,i)
                B1mat[x] = B1element(nharms-x,i)
                B2mat[x] = B2element(nharms-x,i)
            eps1 = D**0.5 / (k01 * (np.exp(-alpha1*j1) + np.exp(beta1*j1)))
            eps2 = D**0.5 / (k02 * (np.exp(-alpha2*j2) + np.exp(beta2*j2)))
            M1 = (eps1*np.identity(2*nharms+1) + H1mat)
            M2 = (eps2*np.identity(2*nharms+1) + H2mat)
            Psi1_far[:,i] = np.linalg.solve(G1mat@np.linalg.inv(M2)@G2mat - M1, G1mat@np.linalg.inv(M2)@B2mat + B1mat)
            Psi2_far[:,i] = np.linalg.solve(G2mat@np.linalg.inv(M1)@G1mat - M2, G2mat@np.linalg.inv(M1)@B1mat + B2mat)
            if nharms > 0:
                I_dl[nharms-1,i] = Qdl[i]*A*(1j*om)**q*E_corr[0,i]
                I_dl[nharms+1,i] = Qdl[i]*A*(-1j*om)**q*E_corr[2,i]
            I_dl[nharms,i] = A*(Qdl[0]*nu*t[i]**(1-q)/(gamma(2-q)) - Ru/dt**q *(sum(G_array[:i]*I_meas[nharms,i:0:-1]*Qdl[i:0:-1])) + dQdE*nu**2*t[i]**(2-q)*gamma(3)/(gamma(3-q)))
            I_est = (I_est + nFAcD1*Psi1_far[:,i] + nFAcD2*Psi2_far[:,i] + I_dl[:,i]) / 2
            if sum(abs(I_est - I_meas[:,i])) < sum(abs(I_est))/1000:
                break
            counter += 1
            if counter > 30:
                print(f'error - not converging at t = {t[i]}')
                break
        I_meas[:,i] = I_est
    I_meas[:,0] = I_meas[:,1]
    return I_meas*1e6

def DIM(t, E_DC, kD = 1e3, n = 1, T = 298, A = 7.07E-6, c0 = 1, E0 = 0, alpha = 0.5, D = 1e-9, k0 = 1e-3, dE = 0.008, freq = 20, Ru = 0, q = 1.0, Qdl = 0, dQdE = 0):
    '''
    Predicts the fundamental harmonic alternating current for single-step, quasireversible charge transfer followed by irreversible dimerization, according to the scheme:
        A + ne- <-> B
        2B -> C
    Required inputs are:
        `t` (array of time points, s)
        'E' (applied linear sweep potential, V)
    Optional inputs are:
        'kD' (dimerization rate constant, m^3.mol^(-1).s^(-1))
        `n` (number of electrons transferred per molecule)
        `T` (temperature, K)
        `A` (electrode surface area, m^2)
        `c0` (bulk concentration of starting material, mol.m^(-3))
        `E0` (thermodynamic potential of redox couple)
        `alpha` (electron transfer coefficient)
        `D` (diffusion coefficient, m^2.s^(-1))
        `k0` (heterogeneous electron transfer rate constant, m.s^(-1))
        `nharms` (number of harmonics in the experiment)
        'dE` (alternating potenial magnitude, V)
        `freq` (alternating potential frequency, Hz)
        `Ru` (uncompensated cell resistance, Ohm)
        `q` (phase angle of the constant phase element responsible for double layer charging current; q=1 corresponds to an ideal capacitor (90 degrees), q=0 to an ideal resistor)
        `Qdl` (effective capacitcance of the constant phase element; if q=1, the units of Qdl are F/m^2)
    Output is:
        `I' (total faradaic current, uA, as an array with length 3. See paper for details.)
    '''
    Qdl = np.ones_like(E_DC)*(Qdl + dQdE*(E_DC - E_DC[0]))
    nharms = 1
    dt = t[1]-t[0]
    om = 2*np.pi*freq
    beta = 1 - alpha
    nFRT = n*F/(R*T)
    nFAcD = n*F*A*c0*D**0.5
    N = len(t)
    Psi_far = np.zeros((2*nharms+1,N), dtype = complex)
    I_dl = np.zeros((2*nharms+1,N), dtype = complex)
    I_meas = np.zeros((2*nharms+1,N), dtype = complex)
    E_array = np.zeros((3,N),dtype=complex)
    E_array[0,:] = np.ones(N)*dE/(2*1j)
    E_array[1,:] = E_DC
    E_array[2,:] = np.ones(N)*dE/(-2*1j)
    E_corr = E_array.copy()
    G_array = np.ones(N)
    for i in range(1,N):
        G_array[i] *= G_array[i-1]*(i-q-1)/i
    def Po(harm, n):
        if harm == 0:
            return 2*np.ones(n)/(np.pi**0.5)*((t[n]-t[:n])**0.5 - (t[n]-t[1:n+1])**0.5)
        else:
            return np.ones(1)/(harm*1j*om)**0.5
    def vo(n):
        return 1 + sum(Po(0,n)[:n-1]*Psi_far[nharms,1:n])
    def Pr(harm, n):
        if harm == 0:
            return -(3/(2*kD*c0))**(1/3)
        elif harm != 0:
            s = harm*1j*om
            L = -(12*D**2/(kD*c0*D**0.5*abs(Psi_far[nharms,n])))**(1/3)
            roe = L*(s/D)**0.5
            return np.ones(1)/(s)**0.5 * ((roe**4 - 6*roe**3 + 15*roe**2 - 15*roe) / (roe**4 - 6*roe**3 + 21*roe**2 - 45*roe + 45))
    def Helement(mu,nu,n):
        return (iv(mu,-alpha*d)*np.exp(mu*1j*(theta - np.pi/2))*Po(nu,n)[-1] + np.exp(j)*iv(mu,beta*d)*np.exp(mu*1j*(theta - np.pi/2))*Pr(nu,n)) / (1 + np.exp(j)) 
    def Belement(mu,n):
        return (iv(mu,-alpha*d)*np.exp(mu*1j*(theta - np.pi/2))*vo(n)) / (1 + np.exp(j))
    for i in range(1,N):
        nu = (E_DC[i]-E_DC[i-1])/dt
        I_est = 2*I_meas[:,i-1] - I_meas[:,i-2]
        counter = 0
        while True:
            I_meas[:,i] = I_est
            E_corr[:,i] = E_array[:,i] - I_est[nharms-1:nharms+2]*Ru
            j = nFRT*(np.real(E_corr[1,i]) - E0)
            theta = np.angle(2j*E_corr[0,i])
            d = nFRT*2*abs(E_corr[0,i])
            eps = D**0.5 / (k0 * (np.exp(-alpha*j) + np.exp(beta*j)))
            a = eps + iv(0,-alpha*d)*Po(0,i)[-1]/(1 + np.exp(j))
            b = np.exp(j)*iv(0,beta*d)*Pr(0,i)/(1 + np.exp(j))
            c = iv(0,-alpha*d)*vo(i)/(1+np.exp(j))
            Psi_far[nharms,i] = (2/3*b/a*(np.cos(1/3*np.arccos(-1 - 27*c*a**2/(2*b**3))) - 1/2))**3
            Psi_far[nharms-1,i] = ( -(iv(1,-alpha*d)*np.exp(1*1j*(theta - np.pi/2))*Po(0,i)[-1]*Psi_far[nharms,i] + np.exp(j)*iv(1,beta*d)*np.exp(1*1j*(theta - np.pi/2))*Pr(0,i)*abs(Psi_far[nharms,i])**(2/3))/(1+np.exp(j)) - Belement(1,i)) / (eps + Helement(0,1,i))
            Psi_far[nharms+1,i] = ( -(iv(-1,-alpha*d)*np.exp(-1*1j*(theta - np.pi/2))*Po(0,i)[-1]*Psi_far[nharms,i] + np.exp(j)*iv(-1,beta*d)*np.exp(-1*1j*(theta - np.pi/2))*Pr(0,i)*abs(Psi_far[nharms,i])**(2/3))/(1+np.exp(j)) - Belement(-1,i)) / (eps + Helement(0,-1,i))
            if nharms > 0:
                I_dl[nharms-1,i] = Qdl[i]*A*(1j*om)**q*E_corr[0,i]
                I_dl[nharms+1,i] = Qdl[i]*A*(-1j*om)**q*E_corr[2,i]
            I_dl[nharms,i] = A*(Qdl[0]*nu*t[i]**(1-q)/(gamma(2-q)) - Ru/dt**q *(sum(G_array[:i]*I_meas[nharms,i:0:-1]*Qdl[i:0:-1])) + dQdE*nu**2*t[i]**(2-q)*gamma(3)/(gamma(3-q)))
            I_est = (I_est + nFAcD*Psi_far[:,i] + I_dl[:,i]) / 2
            if sum(abs(I_est - I_meas[:,i])) < sum(abs(I_est))/1000:
                break
            counter += 1
            if counter > 30:
                print(f'error - not converging at t = {t[i]}')
                break
        I_meas[:,i] = I_est
    I_meas[:,0] = I_meas[:,1]
    return I_meas*1e6

def DISP(t, E_DC, kD = 1e3, n = 1, T = 298, A = 7.07e-6, c0 = 1, E0 = 0, alpha = 0.5, D = 1e-9, k0 = 1e-3, dE = 0.008, freq = 20, Ru = 0, q = 1.0, Qdl = 0, dQdE = 0):
    '''
    Predicts the fundamental harmonic alternating current for single-step, quasireversible charge transfer followed by irreversible disproportionation, according to the scheme:
        A + ne- <-> B
        2B -> C + A
    Required inputs are:
        `t` (array of time points, s)
        'E' (applied linear sweep potential, V)
    Optional inputs are:
        'kD' (dimerization rate constant, m^3.mol^(-1).s^(-1))
        `n` (number of electrons transferred per molecule)
        `T` (temperature, K)
        `A` (electrode surface area, m^2)
        `c0` (bulk concentration of starting material, mol.m^(-3))
        `E0` (thermodynamic potential of redox couple)
        `alpha` (electron transfer coefficient)
        `D` (diffusion coefficient, m^2.s^(-1))
        `k0` (heterogeneous electron transfer rate constant, m.s^(-1))
        `nharms` (number of harmonics in the experiment)
        'dE` (alternating potenial magnitude, V)
        `freq` (alternating potential frequency, Hz)
        `Ru` (uncompensated cell resistance, Ohm)
        `q` (phase angle of the constant phase element responsible for double layer charging current; q=1 corresponds to an ideal capacitor (90 degrees), q=0 to an ideal resistor)
        `Qdl` (effective capacitcance of the constant phase element; if q=1, the units of Qdl are F/m^2)
    Output is:
        `I' (total faradaic current, uA, as an array with length 3. See paper for details.)
    '''
    Qdl = np.ones_like(E_DC)*(Qdl + dQdE*(E_DC - E_DC[0]))
    nharms = 1
    dt = t[1]-t[0]
    om = 2*np.pi*freq
    beta = 1 - alpha
    nFRT = n*F/(R*T)
    nFAcD = n*F*A*c0*D**0.5
    N = len(t)
    Psi_far = np.zeros((2*nharms+1,N), dtype = complex)
    I_dl = np.zeros((2*nharms+1,N), dtype = complex)
    I_meas = np.zeros((2*nharms+1,N), dtype = complex)
    E_array = np.zeros((3,N),dtype=complex)
    E_array[0,:] = np.ones(N)*dE/(2*1j)
    E_array[1,:] = E_DC
    E_array[2,:] = np.ones(N)*dE/(-2*1j)
    E_corr = E_array.copy()
    G_array = np.ones(N)
    for i in range(1,N):
        G_array[i] *= G_array[i-1]*(i-q-1)/i
    def Po(harm, n):
        if harm == 0:
            return 2*np.ones(n)/(np.pi**0.5)*((t[n]-t[:n])**0.5 - (t[n]-t[1:n+1])**0.5)
        else:
            return np.ones(1)/(harm*1j*om)**0.5
    def vo(n):
        return 1 + 1/2*sum(Po(0,n)[:n-1]*Psi_far[nharms,1:n])
    def Pr(harm, n):
        if harm == 0:
            return -(3/(2*kD*c0))**(1/3)
        elif harm != 0:
            s = harm*1j*om
            L = -(12*D**2/(kD*c0*D**0.5*abs(Psi_far[1,n])))**(1/3)
            roe = L*(s/D)**0.5
            return np.ones(1)/(s)**0.5 * ((roe**4 - 6*roe**3 + 15*roe**2 - 15*roe) / (roe**4 - 6*roe**3 + 21*roe**2 - 45*roe + 45))
    def Helement(mu,nu,n):
        return (iv(mu,-alpha*d)*np.exp(mu*1j*(theta - np.pi/2))*Po(nu,n)[-1] + np.exp(j)*iv(mu,beta*d)*np.exp(mu*1j*(theta - np.pi/2))*Pr(nu,n)) / (1 + np.exp(j)) 
    def Belement(mu,n):
        return (iv(mu,-alpha*d)*np.exp(mu*1j*(theta - np.pi/2))*vo(n)) / (1 + np.exp(j))
    for i in range(1,N):
        nu = (E_DC[i]-E_DC[i-1])/dt
        I_est = 2*I_meas[:,i-1] - I_meas[:,i-2]
        counter = 0
        while True:
            I_meas[:,i] = I_est
            E_corr[:,i] = E_array[:,i] - I_est[nharms-1:nharms+2]*Ru
            j = nFRT*(np.real(E_corr[1,i]) - E0)
            theta = np.angle(2j*E_corr[0,i])
            d = nFRT*2*abs(E_corr[0,i])
            eps = D**0.5 / (k0 * (np.exp(-alpha*j) + np.exp(beta*j)))
            a = eps + iv(0,-alpha*d)*(1/2)*Po(0,i)[-1]/(1 + np.exp(j))
            b = (1/2*iv(0,-alpha*d) + np.exp(j)*iv(0,beta*d))*Pr(0,i)/(1 + np.exp(j))
            c = iv(0,-alpha*d)*vo(i)/(1+np.exp(j))
            Psi_far[nharms,i] = (2/3*b/a*(np.cos(1/3*np.arccos(-1 - 27*c*a**2/(2*b**3))) - 1/2))**3
            Psi_far[nharms-1,i] = ( -(iv(1,-alpha*d)/(1j)**(1)*(1/2)*Po(0,i)[-1]*Psi_far[nharms,i]  + (1/2*iv(1,-alpha*d) + np.exp(j)*iv(1,beta*d))/(1j)**(1)*Pr(0,i)*abs(Psi_far[nharms,i])**(2/3))/(1+np.exp(j)) - Belement(1,i)) / (eps + Helement(0,1,i))
            Psi_far[nharms+1,i] = ( -(iv(-1,-alpha*d)/(1j)**(-1)*(1/2)*Po(0,i)[-1]*Psi_far[nharms,i]  + (1/2*iv(-1,-alpha*d) + np.exp(j)*iv(-1,beta*d))/(1j)**(-1)*Pr(0,i)*abs(Psi_far[nharms,i])**(2/3))/(1+np.exp(j)) - Belement(-1,i)) / (eps + Helement(0,-1,i))
            if nharms > 0:
                I_dl[nharms-1,i] = Qdl[i]*A*(1j*om)**q*E_corr[0,i]
                I_dl[nharms+1,i] = Qdl[i]*A*(-1j*om)**q*E_corr[2,i]
            I_dl[nharms,i] = A*(Qdl[0]*nu*t[i]**(1-q)/(gamma(2-q)) - Ru/dt**q *(sum(G_array[:i]*I_meas[nharms,i:0:-1]*Qdl[i:0:-1])) + dQdE*nu**2*t[i]**(2-q)*gamma(3)/(gamma(3-q)))
            I_est = (I_est + nFAcD*Psi_far[:,i] + I_dl[:,i]) / 2
            if sum(abs(I_est - I_meas[:,i])) < sum(abs(I_est))/1000:
                break
            counter += 1
            if counter > 30:
                print(f'error - not converging at t = {t[i]}')
                break
        I_meas[:,i] = I_est
    I_meas[:,0] = I_meas[:,1]
    return I_meas*1e6

def cart(Psi):
    '''
    Converts the alternating current into cartesian form
    Input:
        `Psi` (output from any of the current generating functions above, with length 2N+1)
    Output:
        `x`,`y` (in-phase and out-of-phase components of alternating current, with length N)
    '''
    N = len(Psi)//2
    x = np.real(1j*(Psi[:N]-Psi[-1:-N-1:-1]))
    y = np.real(Psi[:N]+Psi[-1:-N-1:-1])
    return x[::-1], y[::-1]

def pol(Psi):
    '''
    Converts the alternating current into polar form
    Input:
        `Psi` (output from any of the current generating functions above, with length 2N+1)
    Output:
        `phi`,`dI` (phase angle and magnitude of alternating current, with length N)
    '''
    N = len(Psi)//2
    dI = 2*(Psi[:N]*Psi[-1:-N-1:-1])**0.5
    x = np.real(1j*(Psi[:N]-Psi[-1:-N-1:-1]))
    y = np.real(Psi[:N]+Psi[-1:-N-1:-1])
    phi = np.arctan2(y,x)
    return phi[::-1], dI[::-1]

def plotcartesian(Psi1,Psi2,Psi3):
    '''
    Compares complex-plane ACV spectra for each harmonic of three different current functions, `Psi1`, `Psi2`, and `Psi3` 
    '''
    x1,y1 = cart(Psi1)
    x2,y2 = cart(Psi2)
    x3,y3 = cart(Psi3)
    for k in range(len(Psi2)//2):
        plt.axhline(0, ls = ':', c='gray', lw=0.5)
        plt.axvline(0, ls = ':', c='gray', lw=0.5)
        plt.plot(x1[k],y1[k],'-',c='C0')
        plt.plot(x2[k],y2[k],'--',c='C1')
        plt.plot(x3[k],y3[k],'-.',c='C2')
        plt.xlabel(f'Re($I_{k+1}$) ($\mu A$)')
        plt.ylabel(f'Im($I_{k+1}$) ($\mu A$)')
        if k == 0:
            plt.gca().set_aspect('equal')
            plt.gca().set_box_aspect(1)
            plt.title('$1^{\mathrm{st}}$ Harmonic')
        elif k == 1:
            plt.gca().set_aspect('equal')
            plt.gca().set_box_aspect(1)
            plt.title('$2^{\mathrm{nd}}$ Harmonic')
        elif k == 2:
            plt.gca().set_aspect('equal')
            plt.gca().set_box_aspect(1)
            plt.title('$3^{\mathrm{rd}}$ Harmonic')
        plt.tight_layout()
        plt.show()

def plotmagnitude(Psi1,Psi2,Psi3,E_DC):
    '''
    Compares direct current and ACV magnitudes for each harmonic of three different current functions, `Psi1`, `Psi2`, and `Psi3`, as functions of direct potential `E_DC`
    '''
    di1 = pol(Psi1)[1]
    di2 = pol(Psi2)[1]
    di3 = pol(Psi3)[1]
    N = len(Psi2)//2
    plt.plot(E_DC,Psi1[len(Psi1)//2],'-',c='C0')
    plt.plot(E_DC,Psi2[len(Psi2)//2],'--',c='C1')
    plt.plot(E_DC,Psi3[len(Psi3)//2],'-.',c='C2')
    plt.xlabel('$E_{DC}$ ($V$)')
    plt.ylabel('$I_{DC}$ ($\mu A$)')
    plt.title('Direct Current')
    plt.tight_layout()
    plt.show()
    for k in range(N):
        plt.plot(E_DC,di1[k],'-',c='C0')
        plt.plot(E_DC,di2[k],'--',c='C1')
        plt.plot(E_DC,di3[k],'-.',c='C2')
        plt.xlabel('$E_{DC}$ ($V$)')
        plt.ylabel(f'$\Delta I_{k+1}$ ($\mu A$)')
        if k == 0:
            #plt.yticks(np.arange(0,80,10))
            plt.title('$1^{\mathrm{st}}$ Harmonic ')
        elif k == 1:
            #plt.yticks(np.arange(0,25,5))
            plt.title('$2^{\mathrm{nd}}$ Harmonic ')
        elif k == 2:
            #plt.yticks(np.arange(0,8,2))
            plt.title('$3^{\mathrm{rd}}$ Harmonic ')
        plt.tight_layout()
        plt.show()

def measuredcurrent(time, E_DC, Psi, freq, dE, FinerBy = 50):
    '''
    Increases the resolution of `Psi` so that each of the harmonics may be viewed as sinusoidal functions of time
    Required inputs:
        `time` (time array used to generate `Psi` in any of the current generating functions above, s)
        `pot` (DC potential array used to generate `Psi` in any of the current generating functions above, V)
        `Psi` (output from any of the current generating functions above, uA)
        `freq` (alternating potential frequency used to generate `Psi`, Hz)
        `dE' (alternating potential magnitude used to generate `Psi`, V)
    Optional input:
        'FinerBy' (proportional increase in temporal resolution)
    Outputs:
        `newtime` (finer time array, s)
        'newpot' (finer potential array, V. newpot[0] is direct potential, newpot[1] is alternating potential)
        `newI' (array of 'Psi' with increased resolution, uA. `newI[0]` is the DC component, `newI[1]` is the fundamental harmonic, etc.)
    '''
    L = FinerBy*len(time)
    om = 2*np.pi*freq
    newtime = np.linspace(0,time[-1],L+1)
    newEDC = np.interp(newtime,time,E_DC)
    newEalt = dE*np.sin(om*newtime)
    newE_DC = np.array((newEDC,newEalt))
    phi,di = pol(Psi)
    newphi = np.zeros((len(Psi)//2,L+1))
    newdi = np.zeros((len(Psi)//2,L+1))
    newI = np.zeros((len(Psi)//2+1,L+1))
    for k in range(len(Psi)//2):
        newphi[k] = np.interp(newtime,time,phi[k])
        newdi[k] = np.interp(newtime,time,di[k])
        newI[k+1] = newdi[k]*(np.sin((k+1)*om*newtime + newphi[k]))
    newI[0] = np.interp(newtime,time,Psi[len(Psi)//2])
    totalI = np.zeros_like(newtime)
    for k in range(len(newI)):
        totalI += newI[k]
    return newtime, newE_DC, newI

# Generate Figure 4 from the text
time, pot = LSV(nu = 0.05, Ei = 0.5, Ef = -0.5)
Psi1 = QR(time,pot,k0=np.inf)
Psi2 = QR(time,pot)
Psi3 = QR(time,pot,alpha=0.7)
plotcartesian(Psi1,Psi2,Psi3)
plotmagnitude(Psi1,Psi2,Psi3,pot)

# Generate Figure 1 (top) from the text
tt,ee,ii = measuredcurrent(time,pot,Psi2,freq=20,dE=0.035)
iit = np.zeros_like(ii[0])
plt.plot(tt,ee[0]+ee[1],lw=0.2,c='k',label='$E$')
plt.legend(loc=3)
plt.ylabel('$E$ ($V$)')
plt.xlabel('$t$ ($s$)')
plt.twinx()
for k in range(len(ii)):
    iit += ii[k]
plt.plot(tt,iit,lw=0.2,c='blue',label='$I$')
for k in range(len(ii)):
    plt.plot(tt,ii[k],lw=0.2,c=cm.cool((k+0.5)/len(ii)),label = f'$I_{k}$')
plt.legend(loc=1)
plt.ylabel('$I$ ($\mu A$)')
plt.tight_layout()
plt.show()


'''
fig,ax = plt.subplots(2,3,figsize=(12,7),dpi=100)
ax[0,0].plot(cart(Psi1)[0][0],cart(Psi1)[1][0],'x',c='C2',ms=2)
ax[0,0].set_xlim(0,1.1*np.max(cart(Psi1)[1][0]))
ax[0,0].set_ylim(0,1.1*np.max(cart(Psi1)[1][0]))
ax[0,0].grid()
ax[0,1].plot(cart(Psi1)[0][1]-0.12,cart(Psi1)[1][1]+0.24,'x',c='C2',ms=2)
ax[0,1].set_xlim(-1.1*np.max(cart(Psi1)[1][1]+0.24),1.1*np.max(cart(Psi1)[1][1]+0.24))
ax[0,1].set_ylim(-1.1*np.max(cart(Psi1)[1][1]+0.24),1.1*np.max(cart(Psi1)[1][1]+0.24))
ax[0,1].grid()
ax[0,2].plot(cart(Psi1)[0][2]-0.05,cart(Psi1)[1][2]-0.05,'x',c='C2',ms=2)
ax[0,2].set_xlim(-1.1*np.max(cart(Psi1)[0][2]-0.05),1.1*np.max(cart(Psi1)[0][2]-0.05))
ax[0,2].set_ylim(-1.1*np.max(cart(Psi1)[0][2]-0.05),1.1*np.max(cart(Psi1)[0][2]-0.05))
ax[0,2].grid()
ax[1,0].plot(pot,((cart(Psi1)[0][0])**2 + (cart(Psi1)[1][0])**2)**0.5,'x',c='C2')
ax[1,1].plot(pot,((cart(Psi1)[0][1]-0.12)**2 + (cart(Psi1)[1][1]+0.24)**2)**0.5,'x',c='C2')
ax[1,2].plot(pot,((cart(Psi1)[0][2]-0.05)**2 + (cart(Psi1)[1][2]-0.05)**2)**0.5,'x',c='C2')

ax[0,0].set_xlabel('Re($I_1$) / $\mu A$')
ax[0,0].set_ylabel('Im($I_1$) / $\mu A$')
ax[0,0].set_title('First Harmonic')
ax[0,1].set_xlabel('Re($I_2$) / $\mu A$')
ax[0,1].set_ylabel('Im($I_2$) / $\mu A$')
ax[0,1].set_title('Second Harmonic')
ax[0,2].set_xlabel('Re($I_3$) / $\mu A$')
ax[0,2].set_ylabel('Im($I_3$) / $\mu A$')
ax[0,2].set_title('Third Harmonic')
ax[1,0].set_xlabel('$E$ / $V$ $vs$ $Ag/AgNO_3$')
ax[1,0].set_ylabel('$|I_1|$ / $\mu A$')
ax[1,1].set_xlabel('$E$ / $V$ $vs$ $Ag/AgNO_3$')
ax[1,1].set_ylabel('$|I_2|$ / $\mu A$')
ax[1,2].set_xlabel('$E$ / $V$ $vs$ $Ag/AgNO_3$')
ax[1,2].set_ylabel('$|I_2|$ / $\mu A$')
plt.tight_layout()
plt.show()
'''