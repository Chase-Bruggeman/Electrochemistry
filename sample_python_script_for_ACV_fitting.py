import numpy as np
from fraccalc import diffint, bl
import matplotlib.pyplot as plt
from matplotlib import cm, rc
from to_precision import to_precision
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.fft import fft, ifft, fftfreq
import os
rc('xtick', labelsize=8)
rc('ytick', labelsize=8)
rc('font', size=8)

c0 = 2.00 # bulk concentration, mM or mol/m^3
A = 7.07e-6 # electrode surface area, m^2
n = 1 # number of electrons, mol_e-/mol
F = 96485 # Faraday's constant, C/mol_e-
R = 8.314 # Ideal Gas constant, J/(K*mol)
T = 295 # Temperature, K
dE = 0.005 # Alternating potential amplitude, V
f = 20 # Alternating potential frequency, Hz
om = 2*np.pi*f # Alternating potential frequency, radians/sec
nu = -0.05 # Linear potential sweep rate, V/s, 
Efc = 0.089 # Potential of Fc/Fc+ redox couple, V vs Ag/AgNO3
Ru = 93 # Internal cell resistance, Ohms
alpha = 0.5 # symmetry coefficient for electron transfer
beta = 1 - alpha
name = '1-All NA Br'

# load the ACV data
os.chdir('<file directory>')
ACV = np.loadtxt(f'sample_ACV_data.txt', skiprows = 1)

def wavefun(t,A,p):
    return A*np.sin(om*t + p)

def harmonic(n,array,t,f=f):
    # returns the nth harmonic of the array
    # n is the harmonic number (0 = direct current, 1 = fundamental, 2 = second harmonic)
    # array is the 1-d array of the measured current
    # t is the 1-d array of the corresponding time points
    # f is the alternating potential frequency in Hz
    W = fft(array)
    fq = np.arange(len(t))/t[-1]
    W_filt = np.zeros_like(W)
    ind1 = np.where(fq > (n-1/2)*f)[0][0]
    ind2 = np.where(fq > (n+1/2)*f)[0][0]
    W_filt[ind1:ind2] = W[ind1:ind2]
    if n == 0:
        W_filt[-ind2-1:] = W[-ind2-1:]
    else:
        W_filt[-ind2-1:-ind1-1] = W[-ind2-1:-ind1-1]
    return ifft(W_filt)

def fit(n,data,t,f=f):
    # returns a 2-d array; [:,0] is the alternating current magnitude, [:,1] is the phase angle
    # `n` is the harmonic (0 = DC)
    # `data` is the 1-d array of current values for the chosen harmonic
    # `t` is the time in seconds
    # f is the alternating potential frequency in Hz
    n_cycles = int(t[-1]/(1/f))
    if n == 0:
        ifit = np.zeros(n_cycles)
        for j in range(0,n_cycles):
            ifit[j] = np.average(data[np.where(t//(1/f)==j)[0]])
        return ifit
    elif n == 1:
        di_fit = np.zeros(n_cycles)
        phi_fit = np.zeros(n_cycles)
        for j in range(0,n_cycles):
            [di_fit[j], phi_fit[j]] = curve_fit(wavefun,t[np.where(t//(1/f)==j)[0]], data[np.where(t//(1/f)==j)[0]])[0]
        return [di_fit,phi_fit]


# experimental time
t1 = fit(0, ACV[:,5], ACV[:,5]) 

# experimental direct current
idc1 = fit(0, harmonic(0,ACV[:,8],ACV[:,5],20)/1e3, ACV[:,5]) 

# direct potential, corrected for Ru, referenced vs ferrocene
Edc1 = fit(0, ACV[:,7],ACV[:,5]) - idc1*Ru - Efc 

# the sinusoidal component of the measured current
acv1 = harmonic(1,ACV[:,8], ACV[:,5], 20)/1e3 

# the experimental AC magnitude [0] and phase [1]
di1, phi1 = fit(1, acv1, ACV[:,5], 20) 

# complex current, measured
i1 = di1*np.cos(phi1) + 1j*di1*np.sin(phi1) 

# complex potential, measured
v1 = dE*np.ones_like(i1, dtype = complex) 

# complex potential, corrected for Ru
v1c = v1-i1*Ru

# correction to phase angle due to Ru
theta = np.arctan2(np.imag(v1c),np.real(v1c))

# complex current, corrected for Ru
i1_ruc = di1*np.cos(phi1-theta) + 1j*di1*np.sin(phi1-theta)

# complex current, corrected for Ru and background (capacitive) current
i1c = i1_ruc - np.average(i1_ruc[1:int(0.1*len(i1_ruc))])

# Find the diffusion coefficient from convolution data
# Iterate D until convolution change in concentration equals bulk c0 (2 mM)
s400 = np.loadtxt(f'sample_CV_data.txt', skiprows=1)
D = 1.5e-9
si400 = diffint(s400[:,8]/1e3,s400[:,5]) # semiintegral of current
si400_2 = bl(si400,s400[:,7],int(0.08*len(si400)),int(0.18*len(si400))) # baseline the semiintegral
plt.plot(s400[:,7],si400/(F*A*D**0.5))
plt.plot(s400[:,7],si400_2/(F*A*D**0.5))
plt.show()

# Plot the convolution vs the CV
plt.figure(figsize = (3.35,2.9), dpi = 300)
plt.plot(s400[:,7][s400[:,9]==1], s400[:,8][s400[:,9]==1]*1e3)
plt.plot(0)
plt.legend(('Current','Convolution'), fontsize=9, loc=2)
plt.grid()
plt.xlabel('$E$ ($V$ $vs$ $SCE$)')
plt.ylabel('$I$ ($\mu A$)')
plt.ylim(-120,80)
plt.yticks(40*np.arange(-3,2))
plt.twinx()
plt.plot(s400[:,7][s400[:,9]==1], si400[s400[:,9]==1]/(F*A*D**0.5),c='C1')
plt.ylabel('$\Delta c_{{ox}}|_{{x=0}}$ ($mM$)')
plt.ylim(-6,4)
plt.xlim(-2.2,0.3)
plt.tight_layout()
# plt.savefig(f'{name} 400 mVs i vs convolution.png', transparent = True)
plt.show()

# Find c_ox from direct current data
si = diffint(idc1,t1)
j1 = int(0.4*len(si))
j2 = int(0.45*len(si))
si_cor = bl(si,Edc1,j1,j2) # baseline the start of the convolution integral
k1 = int(0.8*len(si))
k2 = int(0.9*len(si))
Ipsi = si_cor/(si_cor - bl(si_cor,Edc1,k1,k2)) # baseline the end of the convolution integral
cox = c0*(1-Ipsi)

# Baseline the measured current (will be used to find cr in `fund_harm()`)
idc = bl(idc1,Edc1,j1,j2)


def fund_harm(Edc1,E0,k0,logkD):
    '''
    Predict the fundamental harmonic alternating current from a given set of values for E0, k0, and kD
    Equation numbers match those in Chase Bruggeman's PhD dissertation, 
    Michigan State University, Spring 2024, "Electrochemistry of Nictotinamide Adenine Dinucleotide Mimetics
    '''
    kD = 10**logkD
    # Select which part of the data to process
    if abridge == True:
        # Process only part of the data
        # This is used for curve_fit, so that only the points associated with the redox event are considered
        domain = signal_indices
    elif abridge == False:
        # Process all of the data
        # This is used for generating polar plots across the entire range of experimental potentials
        domain = np.arange(len(idc))
    # Abridge the data as needed
    idc_ = idc[domain]
    cox_ = cox[domain]
    v1c_ = v1c[domain]
    Edc_ = Edc1[domain]
    # Eq S12
    j = n*F/(R*T) * (Edc_ - E0)
    # Eq. S53
    cr = (abs(idc_)/(n*F*A))**(2/3) * (3/(2*D*kD))**(1/3)
    b = np.zeros((len(idc_),4), dtype = complex)
    a = np.zeros((len(idc_),4), dtype = complex)
    # denominator of left-hand side of Eq S63
    poly = np.vstack([[np.ones(len(idc_))], [(6*kD*cr)**0.5], [(7/2)*kD*cr], [(15/24**0.5)*(kD*cr)**1.5], [(5/4)*(kD*cr)**2]])
    # solve Eq S68
    for p in range(len(poly[0,:])):
        # roots for denominators in right-hand side of Eq S63
        b[p,:] = -np.roots(poly[:,p]) 
        # matrix in left-hand side of Eq S68
        coeff_mat = np.vstack([[1, 1, 1, 1], 
        [b[p,1]+b[p,2]+b[p,3], b[p,0]+b[p,2]+b[p,3], b[p,0]+b[p,1]+b[p,3], b[p,0]+b[p,1]+b[p,2]], 
        [b[p,1]*b[p,2]+b[p,1]*b[p,3]+b[p,2]*b[p,3], b[p,0]*b[p,2]+b[p,0]*b[p,3]+b[p,2]*b[p,3], b[p,0]*b[p,1]+b[p,0]*b[p,3]+b[p,1]*b[p,3], b[p,0]*b[p,1]+b[p,0]*b[p,2]+b[p,1]*b[p,2]],
        [b[p,1]*b[p,2]*b[p,3], b[p,0]*b[p,2]*b[p,3], b[p,0]*b[p,1]*b[p,3], b[p,0]*b[p,1]*b[p,2]]])
        # matrix in right-hand side of Eq S68
        knowns_mat = np.array([1, (6*kD*cr[p])**0.5, (5/2)*kD*cr[p], (5/24**0.5)*(kD*cr[p])**1.5])
        # numerators in right-hand side of Eq S63
        a[p,:] = np.linalg.solve(coeff_mat, knowns_mat)
    Ls = np.zeros_like(idc_)
    Lc = np.zeros_like(idc_)
    for p in range(len(idc_)):
        # Eq S87
        Ls[p] = abs(sum(a[p]/(b[p]**4 + om**2) * (om**2 + b[p]*om*(b[p] - (2*om)**0.5))))
        # Eq S88
        Lc[p] = abs(sum(a[p]/(b[p]**4 + om**2) * (om**2 + b[p]**2*(b[p]*(2*om)**0.5 - om))))
    # Eq S97
    lam = (k0/D**0.5) * (np.exp(-alpha*j) + np.exp((1-alpha)*j))
    v = (2*om)**0.5/lam + 1/(1+np.exp(j)) * (1 + np.exp(j)*Lc)
    # Eq S98
    u = 1/(1+np.exp(j)) * (1 + np.exp(j)*Ls)
    # Eq S108
    phi = np.arctan2(u, v)
    # Eq S112
    Ft = (1 + np.exp(-j)) * (alpha*cox_/c0 + np.exp(j) * (1-alpha)*cr/c0)
    # Eq S113
    Gw = (2 / (v**2 + u**2) )**0.5
    # Eq S115
    Irev = (n*F)**2 * A*c0*abs(v1c_)*(om*D)**0.5 / (4*R*T*np.cosh(0.5*j)**2)
    # Eq S117
    Iwt = Irev*Ft*Gw
    return np.hstack((Iwt*np.cos(phi), Iwt*np.sin(phi)))

# Fit the data where the intensity of the signal is at least 20% of the maximum signal
abridge = True
signal_indices = np.where(abs(i1c)>0.2*max(abs(i1c)))[0][1:]
popt, pcov = curve_fit(fund_harm,Edc1,np.hstack((np.real(i1c[signal_indices]),np.imag(i1c[signal_indices]))), p0 = (-1.5,0.01,5))
# popt gives the best-fit parameters
# pcov gives the variance (square of standard deviation) along the diagonal
# off-diagonal terms of pcov are covariance values
# essentially the products of the standard deviations of the diagonal terms

# Create a plot comparing experimental and theoretical fits
abridge = False
plt.plot(np.real(i1c),np.imag(i1c),'o-')
plt.plot(fund_harm(Edc1,*popt)[:len(Edc1)], fund_harm(Edc1,*popt)[len(Edc1):], 'o', ms = 3)
plt.show()

# Figure to inspect the magnitude and phase of the current as functions of potential
x = fund_harm(Edc1,*popt)[:len(Edc1)]
y = fund_harm(Edc1,*popt)[len(Edc1):]
ie = np.where(di1==max(di1))[0][0]
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(12,6))
ax1.plot(Edc1,abs(i1c)*1e6,'o-',ms=3, c = 'C0')
ax1.plot(Edc1,(x**2+y**2)**0.5*1e6,'o-',ms=2, c = 'C1')
ax1.set_xlabel('$E_{DC}$ ($V$ $vs$ $Fc/Fc^+$)')
ax1.set_ylabel('$I$ ($\mu A$)')
ax1.set_xlim(-1.8,-1.2)
ax1.set_ylim(0,10)
ax1.legend(('Experiment', 'Theory'))
ax2.plot(Edc1,np.arctan2(np.imag(i1c),np.real(i1c))*180/np.pi,'--',c='C0')
ax2.plot(Edc1,np.arctan2(y,x)*180/np.pi,'--',c='C1')
ax2.set_xlim(-1.8,-1.2)
ax2.set_ylim(0,90)
ax2.set_ylabel('$\phi$ (degrees)')
ax2.legend(('Experiment', 'Theory'))
plt.show()

# Nice Figure of Experimental vs. Theoretical Fit
fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(3.35,3.35),dpi=300)
plt.xlim(0,np.pi/2)
plt.xticks([0,np.pi/4,np.pi/2])
plt.ylim(0,8)
ax.set_xlabel('$\Delta I$ ($\mu A$)', fontsize=8)
ax.xaxis.set_label_coords(0.5,-0.1)
plt.tight_layout()
c1 = cm.Greens(np.linspace(0,1,len(i1c)))
c2 = cm.Greys(np.linspace(0,1,len(i1c)))
# Points are lighter at more positive potentials and darker at more negative potentials
# Gray circles = experiment
# Green dots = theory
for i in range(len(c2)):
    ax.plot(np.arctan2(np.imag(i1c[i]),np.real(i1c[i])), abs(i1c[i])*1e6, marker = '$\mathrm{O}$', c=c2[i], mew=0.01, ms=3.5)

for i in range(len(c1)):
    ax.plot(np.arctan2(y[i],x[i]), (x[i]**2+y[i]**2)**0.5*1e6, 'o', c=c1[i], ms=0.6)

# Mark the point at peak signal/noise
ax.plot(np.arctan2(np.imag(i1c[ie]),np.real(i1c[ie])), abs(i1c[ie])*1e6, '+', ms = 6, c=c2[ie])
ax.plot(np.arctan2(y[ie],x[ie]), (x[ie]**2+y[ie]**2)**0.5*1e6, 'x', ms = 6, c = c1[ie])
# plt.savefig(f'{name} experiment vs theory.png', transparent=True)
plt.show()

'''
Generate contour plots of the error as a function of the unknown parameters
This code is commented because it takes ~15 minutes to run

# Function used to find the error
# Eqs S118/S119 in Chase's thesis
def sigma(E0=E0,k0=k0,kD=kD,alpha=alpha):
    x,y = fund_harm(E0,k0,kD,alpha)
    signal_indices = np.where(abs(i1c)>0.1*max(abs(i1c)))[0][2:]
    ierr = i1c[signal_indices] - y[signal_indices]*(np.cos(x[signal_indices]) + 1j*np.sin(x[signal_indices]))
    werr = ierr * abs(i1c[signal_indices])/max(abs(i1c[signal_indices]))
    return sum(abs(werr))/len(signal_indices)

# Array for the error values 
sigma_array = np.zeros((21,6,20,5))
# Parameter space to sweep through (adjust as needed)
E0_list = np.linspace(-1.549,-1.470,21)
k0_list = np.geomspace(1e-3,1e0,6)
kD_list = np.geomspace(1e3,1e8,20)
alpha_list = np.linspace(0.2,0.8,5)

# Calculate the error
for aa in range(21):
    print(aa) # monitor the progress of the `for` loop
    for bb in range(6):
        print(bb) # monitor the progress of the `for` loop
        for cc in range(20):
            for dd in range(5):
                sigma_array[aa,bb,cc,dd] = sigma(E0_list[aa],k0_list[bb],kD_list[cc],alpha_list[dd])

# define levels for the contour plot
lv = np.linspace(0.0,1,21)

# generate the plot
fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2, figsize=(6.8,6.7),dpi=300)
CS1 = ax1.contourf(kD_list*1e3, E0_list, sigma_array[:,3,:,1]*1e6, levels = lv)
CS2 = ax2.contourf(k0_list*1e2, E0_list, sigma_array[:,:,10,1]*1e6, levels = lv)
CS3 = ax3.contourf(alpha_list, E0_list, sigma_array[:,3,10,:]*1e6, levels = lv)

CS4 = ax4.contourf(alpha_list, k0_list*1e2, sigma_array[8,:,10,:]*1e6, levels = lv)
CS5 = ax5.contourf(kD_list*1e3, k0_list*1e2, sigma_array[8,:,:,1]*1e6, levels = lv)
CS6 = ax6.contourf(alpha_list, kD_list*1e3, sigma_array[8,3,:,:]*1e6, levels = lv)

ax1.set_ylabel('$E^0$ ($V$ $vs$ $Fc/Fc^+$)')
ax1.set_xlabel('$k_D$ ($M^{-1}$ $s^{-1}$)')
ax1.set_xscale('log')
ax1.set_xlim(1e6,1e10)
ax2.set_ylabel('$E^0$ ($V$ $vs$ $Fc/Fc^+$)')
ax2.set_xlabel('$k^0$ ($cm$ $s^{-1}$)')
ax2.set_xscale('log')
ax3.set_ylabel('$E^0$ ($V$ $vs$ $Fc/Fc^+$)')
ax3.set_xlabel(r'$\alpha$')

ax4.set_ylabel('$k^0$ ($cm$ $s^{-1}$)')
ax4.set_xlabel(r'$\alpha$')
ax4.set_yscale('log')
ax5.set_ylabel('$k^0$ ($cm$ $s^{-1}$)')
ax5.set_xlabel('$k_D$ ($M^{-1}$ $s^{-1}$)')
ax5.set_xscale('log')
ax5.set_yscale('log')
ax6.set_ylabel('$k_D$ ($M^{-1}$ $s^{-1}$)')
ax6.set_xlabel(r'$\alpha$')
ax6.set_yscale('log')

fig.colorbar(CS1)
fig.colorbar(CS2)
fig.colorbar(CS3)
fig.colorbar(CS4)
fig.colorbar(CS5)
fig.colorbar(CS6)
plt.tight_layout()
plt.savefig('Weighted Error Contour Plot.png')
plt.show()
'''