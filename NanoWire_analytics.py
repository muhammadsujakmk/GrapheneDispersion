import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from Grap_OptCond import Grap_Cal
from mpmath import *


def spec_func(mu1,mu2,R,n=0):
    I = mp.besseli(n,mu1*R)
    Id = mp.besseli(n+1,mu1*R)+n/mu1/R*mp.besseli(n,mu1*R)
    K = mp.besselk(n,mu2*R)
    Kd = mp.besselk(n+1,mu2*R)+n/mu2/R*mp.besseli(n,mu2*R)
    return I,Id,K,Kd

def BuildMatx(beta): 
    ## Constant in SI
    mu0 = 1.25663706143e10-6
    eps0 = 8.8541878188e-12  #Vacuum permittivity, F/m
    eps1 = 3*eps0
    eps2 = eps0
    c = 299792458 # speed of light, m/s
    ## Wire Parameter
    R = 100 # nm
    omega = 10#np.arange(10,50,.001)
    k0 = 2*pi*omega/c
    sig = Grap_Cal(.5,.5,omega)
    n = 0 


    mu1 = (( beta**2 - omega**2 * eps1 * mu0 )**.5)
    mu2 = (( beta**2 - omega**2 * eps2 * mu0 )**.5)
    
    a11 = mp.besseli(n,mu1*R)
    a12 = 0
    a13 = -mp.besselk(n,mu2*R)
    a14 = 0

    a21 = -1j * n * beta * spec_func(mu1,mu2,R,n)[0] / mu1**2 / R
    a22 = omega * mu0 * spec_func(mu1,mu2,R,n)[1] /mu1
    a23 = -1j * n * beta * spec_func(mu1,mu2,R,n)[2] / mu2**2 / R
    a24 = -omega * mu0 * spec_func(mu1,mu2,R,n)[3] /mu2

    a31 = 1j * omega * eps1 * spec_func(mu1,mu2,R,n)[1] / mu1 - sig * spec_func(mu1,mu2,R,n)[0]  
    a32 = -n * beta / mu1**2 / R * spec_func(mu1,mu2,R,n)[0]
    a33 = 1j * omega * eps2 * spec_func(mu1,mu2,R,n)[3] / mu2
    a34 = -n * beta / mu2**2 / R * spec_func(mu1,mu2,R,n)[2]

    a41 = -sig * n * beta * spec_func(mu1,mu2,R,n)[0] / mu1**2 / R  
    a42 = spec_func(mu1,mu2,R,n)[0] - sig * 1j * omega * mu0 * spec_func(mu1,mu2,R,n)[1] / mu1
    a43 = 0
    a44 = -spec_func(mu1,mu2,R,n)[2]
    

    res = (a21-a23*a11/a13-a41*a24/a44)*(a32-a42*a34/a44)-(a22-a42*a24/a44)*(a31-a33*a11/a13-a41*a34/a44)
    return res,k0

def find_beta_root():
    beta_initial_guess = 2.0958e-7  # Adjust the initial guess based on your problem
    beta_root,k0 = mp.findroot(BuildMatx, beta_initial_guess)
    print(f"Root (beta): {beta_root}")
    print(f" Real(beta)/k0: {beta_root.real/k0}")
    return beta_root

find_beta_root()
"""
detMat = np.array([[a11,12,a13,a14],
                    [a21,a22,a23,a24],
                    [a31,a32,a33,a34],
                    [a41,a42,a43,a44]])

return np.linalg.det(detMat)
"""
