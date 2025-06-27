# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 10:05:00 2021

CID: 01702088
"""

import numpy as np

'''
N: number of entrices in each bin, the real measurements
S: unoscillated flux in each bin, the simulated data
i: for the ith bin
lamb_i = P_i * S_i, Pi is related to a (mixing angle) and m (mass difference squared) and E_i
E_i = 0.025+0.05*i, is the (mid-point) energy of each bin, i ranges from 0 to 199
take partial derivatives of lamb_i w.r.t a and m
'''

N,S = np.loadtxt('data.csv',delimiter=',',unpack=1)

L = 295

E = []
for i in range(len(N)):
    E.append(0.025+0.05*i)

def dlamb_da(a,m,i):
    k = 1.267*L/E[i]
    return -2*S[i]*np.sin(4*a)*np.sin(k*m)**2
    
def dlamb_dm(a,m,i):
    k = 1.267*L/E[i]
    return -k*S[i]*np.sin(2*a)**2*np.sin(2*k*m)

def d2lamb_da2(a,m,i):
    k = 1.267*L/E[i]
    return -8*S[i]*np.cos(4*a)*np.sin(k*m)**2

def d2lamb_dm2(a,m,i):
    k = 1.267*L/E[i]
    return -2*k**2*S[i]*np.sin(2*a)**2*np.cos(2*k*m)

def d2lamb_dadm(a,m,i):
    k = 1.267*L/E[i]
    return -2*k*S[i]*np.sin(4*a)*np.sin(2*k*m)

'''
use Newton's method to minimize NLL(lamb) where lamb = lamb(a,m).
NLL = summation(f(lamb_i))
take partial detivatives of f w.r.t a and m and sum over all i is the partial derivatives of NLL
which will be used in the Hessian matrix and the gradient
'''

def df_dlamb(a,m,i):
    k = 1.267*L/E[i]
    lamb_i = (1-np.sin(2*a)**2*np.sin(k*m)**2)*S[i]
    return 1-N[i]/lamb_i

def d2f_dlamb2(a,m,i):
    k = 1.267*L/E[i]
    lamb_i = (1-np.sin(2*a)**2*np.sin(k*m)**2)*S[i]
    return N[i]/lamb_i**2

'''
chian rule is used to relate f (and hence NLL) with a and m
'''

def dNLL_da(a,m):
    summ = 0
    for i in range(len(N)):
        summ += df_dlamb(a,m,i)*dlamb_da(a,m,i)
    return summ

def dNLL_dm(a,m):
    summ = 0
    for i in range(len(N)):
        summ += df_dlamb(a,m,i)*dlamb_dm(a,m,i)
    return summ

def d2NLL_da2(a,m):
    summ = 0
    for i in range(len(N)):
        summ += d2f_dlamb2(a,m,i)*dlamb_da(a,m,i)**2+df_dlamb(a,m,i)*d2lamb_da2(a,m,i)
    return summ

def d2NLL_dm2(a,m):
    summ = 0
    for i in range(len(N)):
        summ += d2f_dlamb2(a,m,i)*dlamb_dm(a,m,i)**2+df_dlamb(a,m,i)*d2lamb_dm2(a,m,i)
    return summ

def d2NLL_dadm(a,m):
    summ = 0
    for i in range(len(N)):
        summ += d2f_dlamb2(a,m,i)*dlamb_dm(a,m,i)*dlamb_da(a,m,i)+df_dlamb(a,m,i)*d2lamb_dadm(a,m,i)
    return summ

'''
having all the derivatives ready, now can define the Hessian matrix and the gradient
'''

def H(a,m):
    Hessian_matrix = np.array([[d2NLL_da2(a,m),d2NLL_dadm(a,m)],
                               [d2NLL_dadm(a,m),d2NLL_dm2(a,m)]])
    return Hessian_matrix

def grad(a,m):
    gradient = np.array([dNLL_da(a,m),dNLL_dm(a,m)])
    return gradient

'''
iterate to simultaneously minimize NLL w.r.t both a and m
'''

def newton_method(a0,m0):   
    x0 = np.array([a0,m0])
    x1 = x0 - np.dot(np.linalg.inv(H(a0,m0)),grad(a0,m0))
    x0_modulus = np.sqrt(x0[0]**2+x0[1]**2)
    x1_modulus = np.sqrt(x1[0]**2+x1[1]**2)
    n = 0
    while abs(x1_modulus - x0_modulus) > 1e-10:
        n += 1
        print('iteration %s:'%(n))
        x0 = x1
        x1 = x0 - np.dot(np.linalg.inv(H(x0[0],x0[1])),grad(x0[0],x0[1]))
        x0_modulus = np.sqrt(x0[0]**2+x0[1]**2)
        x1_modulus = np.sqrt(x1[0]**2+x1[1]**2)
        print(x1,'\n')
    return x1[0],x1[1]

'''
use parameters obtained from previous univariate method
errors are estimated from the covariance matrix
'''

a,m = newton_method(0.71768,0.00233)
cov = np.linalg.inv(H(a,m))
err_a = np.sqrt(cov[0,0])
err_m = np.sqrt(cov[1,1])
print('theta =',a,'+-',err_a)
print('mdiff2 =',m,'+-',err_m)    

'''
calculate the value (minimum) of NLL at the optimised a and m
compare to that obtained from univariate method
'''

def get_nll(a,m):
    nll = 0
    for i in range(len(N)):
        E = 0.025+0.05*i
        lamb = (1-np.sin(2*a)**2*np.sin(1.267*m*L/E)**2)*S[i]
        if N[i] == 0:
            nll += lamb
        else:
            nll += lamb-N[i]+N[i]*np.log(N[i]/lamb)
    return nll
    
nll_min = get_nll(a,m)
print('minimum value of NLL =',nll_min)


    
    