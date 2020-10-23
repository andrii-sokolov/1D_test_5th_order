'''
***********************
* This code implements
* TDSE solver of based
* on the 5-th order
* scheme
* Andrii Sokolov, Elena Blokhina
***********************
'''

import numpy as np

def H1rot(Wf,U,x,tau):
    h = x[1]-x[0]
    for i in range(int(len(U)/4)):
        P1 = Wf[4*i]
        P2 = Wf[4*i + 1]
        P3 = Wf[4*i + 2]
        P4 = Wf[4*i + 3]
        Wf[4*i] = P1*np.cos(tau/(12*h**2)) - 1j*P3*np.sin(tau/(12*h**2))
        Wf[4*i + 2] = P3*np.cos(tau/(12*h**2)) - 1j*P1*np.sin(tau/(12*h**2))
        Wf[4*i + 1] = P2*np.cos(tau/(12*h**2)) - 1j*P4*np.sin(tau/(12*h**2))
        Wf[4*i + 3] = P4*np.cos(tau/(12*h**2)) - 1j*P2*np.sin(tau/(12*h**2))

def H2rot(Wf,U,x,tau):
    h = x[1]-x[0]
    for i in range(int(len(U)/4)-1):
        P1 = Wf[4*i + 2]
        P2 = Wf[4*i + 3]
        P3 = Wf[4*i + 4]
        P4 = Wf[4*i + 5]
        Wf[4*i + 2] = P1*np.cos(tau/(12*h**2)) - 1j*P3*np.sin(tau/(12*h**2))
        Wf[4*i + 4] = P3*np.cos(tau/(12*h**2)) - 1j*P1*np.sin(tau/(12*h**2))
        Wf[4*i + 3] = P2*np.cos(tau/(12*h**2)) - 1j*P4*np.sin(tau/(12*h**2))
        Wf[4*i + 5] = P4*np.cos(tau/(12*h**2)) - 1j*P2*np.sin(tau/(12*h**2))

def H3rot(Wf,U,x,tau):
    h = x[1]-x[0]
    for i in range(int(len(U)/2)):
        P1 = Wf[2*i]
        P2 = Wf[2*i + 1]
        Wf[2*i] = P1*np.cos(4*tau/(3*h**2)) + 1j*P2*np.sin(4*tau/(3*h**2))
        Wf[2*i+1] = P2*np.cos(4*tau/(3*h**2)) + 1j*P1*np.sin(4*tau/(3*h**2))

def H4rot(Wf,U,x,tau):
    h = x[1]-x[0]
    for i in range(int(len(U)/2)-1):
        P1 = Wf[2*i + 1]
        P2 = Wf[2*i + 2]
        Wf[2*i + 1] = P1*np.cos(4*tau/(3*h**2)) + 1j*P2*np.sin(4*tau/(3*h**2))
        Wf[2*i + 2] = P2*np.cos(4*tau/(3*h**2)) + 1j*P1*np.sin(4*tau/(3*h**2))

def H5rot(Wf,U,x,tau):
    h = x[1]-x[0]
    for i in range(len(U)):
        Wf[i] = Wf[i]*np.exp(-1j*tau*(5/(2*h**2) + U[i]))

def TDSE_1D_Solver(inWf,U,x,tau,N):
    Wf = np.copy(inWf)
    for i in range(N):
        H1rot(Wf,U,x,tau)
        H2rot(Wf,U,x,tau)
        H3rot(Wf,U,x,tau)
        H4rot(Wf,U,x,tau)
        H5rot(Wf,U,x,tau)
    return(Wf)

def TDSE_split_operator(inWf,U,x,tau,N):
    Wf = np.copy(inWf)
    h = x[1]-x[0]
    for j in range(N):
        # multiply the Wf and 0.5 step in r-space:
        for i in range(len(Wf)):
            Wf[i] = Wf[i]*np.exp(-1j*U[i]*tau/(2.0*h))
        # going to the k-space:
        Wf = np.fft.fft(Wf)
        k = np.pi*np.fft.fftfreq(len(Wf),d=1/len(Wf))/x[-1]
        for i in range(len(Wf)):
            Wf[i] = Wf[i]*np.exp(-1j*k[i]**2*tau/(2.0*h))
        Wf = np.fft.ifft(Wf)
        # multiply the Wf and 0.5 step in r-space:
        for i in range(len(Wf)):
            Wf[i] = Wf[i]*np.exp(-1j*U[i]*tau/(2.0*h))
    return(Wf)
