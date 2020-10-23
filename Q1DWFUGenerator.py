'''
***********************
* This code implements
* Generates the 2-well potential profile,
* Gauss-like test wave-function,
* and the Static-solver to receive the
* Eigenfunctions on the ground state.
* Andrii Sokolov, Elena Blokhina
***********************
'''

import numpy as np

def Udouble(x,x1 = 0.5, x2 = 3.5, U1 = 10.0, U2 = 100.0):
    UU = np.where(abs(x)<=x1,U1,np.where(abs(x)>=x2,U2,0))
    return(UU)

def Gauss(x, x0 = -2, alpha= 0.5):
    GG = np.exp(-np.power(x-x0,2)/np.power(alpha,2))/(np.sqrt(np.pi)*alpha/np.sqrt(2)) + 1j*0
    return(GG)

def StaticSolver( x_array, p_array ):
    N = len(x_array)
    h = x_array[1]-x_array[0]

    T = np.zeros((N-2)**2).reshape(N-2,N-2)
    for i in range(N-2):
        for j in range(N-2):
            if i==j:
                T[i,j]= -2
            elif np.abs(i-j)==1:
                T[i,j]=1
            else:
                T[i,j]=0

    V = np.zeros((N-2)**2).reshape(N-2,N-2)
    for i in range(N-2):
        for j in range(N-2):
            if i==j:
                V[i,j]= p_array[i+1]
            else:
                V[i,j]=0

    H = -T/(2*h**2) + V

    val,vec = np.linalg.eig(H)

    Energy_indexes = np.argsort(val)
    Energies = (val[Energy_indexes])

    Wf = []
    for i in range(len(Energy_indexes)):
        Wf.append(vec[:,Energy_indexes[i]])

    return(Energies, Wf)
