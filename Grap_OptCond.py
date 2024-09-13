import numpy as np
import mpmath as mp
from mpmath import log, exp
from mpmath import cosh, sinh
from mpmath import quad 
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from time import sleep
from tqdm import tqdm


## Universal Constant - SI (MKS)
ee = 1.60217663e-19 #elementary charge
pi = np.pi
eps0 = 8.8541878128e-12 # Vacuum permittivity, F/m
hh = 6.62607015e-34 # Planck constant, J. s
hheV = 4.135667696e-15 # Planck constant, eV. s
hbar = hh/2/pi      # Reduced Planck constant (J. s)
hbar_ = 1.05457182e-34 
hbareV = hheV/2/pi      # Reduced Planck constant (eV. s)
c = 299792458       # speed of light, m/s
kB = 1.380649e-23 # Boltzmann constant, J/K
kBeV = 8.617333262e-5 # Boltzmann constant, eV/K
vf = 1.1e6 # Fermi velocity of electron, m/s
X = hbar/ee 


## Unit adjustment
nm = 1e-9
um = 1e-6

## Parameter used here from Emani,Nanoletter,2012
#nd = np.array([3e11,1e12,1e13])
#nd_ = [6e16] # Carrier densities, m^-2
T = 300 
t_Gr = 1*nm # Graphene thickness

# intraband 
def sigma_intra(wavelength,omega_T,omega_F,tau,ee,hbar):
    omega = 2*np.pi*c/wavelength
    sig_intra = 2 * ee**2 * omega_T * 1j
    sig_intra /= (pi * hbar * (omega+1j/tau))
    sig_intra *= log(2*cosh(omega_F/2/omega_T))
    return complex(sig_intra)

# interband
def sigma_inter(wavelength,omega_T,omega_F,c,ee,hbar):
    omega = 2*np.pi*c/wavelength 
    omegaX = omega*X
    omega_TX = omega_T*X 
    omega_FX = omega_F*X
    Hw = lambda w: sinh(w/omega_TX)/(cosh(omega_FX/omega_TX)+cosh(w/omega_TX))
    
    intg = quad(lambda w: ( Hw(w/2)-Hw(omegaX/2) ) / (omegaX**2 - w**2), [0,np.inf])*X
    return complex(ee**2/4/hbar*(Hw(omegaX/2)+1j*2*omega/pi*intg))

def Grap_Cal(EF,TAU,freq):
    wvl = 299.792458/freq
    wavelength =wvl*um
    #Ef = hbar * vf * (pi * nd)**.5
    EfeV_ =EF# eV 
    EfeV =EfeV_ 
    Ef = EfeV*ee 
    #MU = .4#tau * ee * vf**2 / Ef
    tau = TAU*1e-12 
    MU = tau * ee * vf**2 / Ef
    tau = MU*Ef/ee/vf**2 
    omega_T = kB*T/hbar
    omega_F = EfeV/hbareV
    rat = omega_T/omega_F
    omega = 2*np.pi*c/wvl
    sig_intra = sigma_intra(wvl,omega_T,omega_F,tau,ee,hbar)
    sig_inter = sigma_inter(wvl,omega_T,omega_F,c,ee,hbar)
    sigma = sig_intra+sig_inter

    eps_Gr = 1j*sigma/omega/eps0/t_Gr
    return sigma



