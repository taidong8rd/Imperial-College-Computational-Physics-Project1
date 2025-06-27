# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 11:21:32 2021

CID: 01702088
"""

import numpy as np
import matplotlib.pyplot as plt

'''
N: number of entrices in each bin, the real measurements
S: unoscillated flux in each bin, the simulated data
i: for the ith bin
lamb_i = P_i*S_i*r*E_i, Pi is related to: 
a (mixing angle),
m (mass difference squared),
r (rate of increase of the cross-section with neutrino energy E_i) where
E_i = 0.025+0.05*i, is the (mid-point) energy of each bin, i ranges from 0 to 199
take partial derivatives of lamb_i w.r.t a, m and r
'''

N,S = np.loadtxt('data.csv',delimiter=',',unpack=1)

L = 295

E = []
for i in range(len(N)):
    E.append(0.025+0.05*i)

def dlamb_da(a,m,r,i):
    k = 1.267*L/E[i]
    return -2*S[i]*np.sin(4*a)*np.sin(k*m)**2*r*E[i]

def dlamb_dm(a,m,r,i):
    k = 1.267*L/E[i]
    return -k*S[i]*np.sin(2*a)**2*np.sin(2*k*m)*r*E[i]

def dlamb_dr(a,m,r,i):
    k = 1.267*L/E[i]
    return (1-np.sin(2*a)**2*np.sin(k*m)**2)*S[i]*E[i]

def d2lamb_da2(a,m,r,i):
    k = 1.267*L/E[i]
    return -8*S[i]*np.cos(4*a)*np.sin(k*m)**2*r*E[i]

def d2lamb_dm2(a,m,r,i):
    k = 1.267*L/E[i]
    return -2*k**2*S[i]*np.sin(2*a)**2*np.cos(2*k*m)*r*E[i]

def d2lamb_dr2(a,m,r,i):
    return 0

def d2lamb_dadm(a,m,r,i):
    k = 1.267*L/E[i]
    return -2*k*S[i]*np.sin(4*a)*np.sin(2*k*m)*r*E[i]

def d2lamb_drda(a,m,r,i):
    k = 1.267*L/E[i]
    return -2*S[i]*np.sin(4*a)*np.sin(k*m)**2*E[i]

def d2lamb_drdm(a,m,r,i):
    k = 1.267*L/E[i]
    return -k*S[i]*np.sin(2*a)**2*np.sin(2*k*m)*E[i]

'''
use Newton's method to minimize NLL(lamb) where lamb = lamb(a,m,r).
NLL = summation(f(lamb_i))
take partial detivatives of f w.r.t a, m and r and sum over all i is the partial derivatives of NLL
which will be used in the Hessian matrix and the gradient
'''

def df_dlamb(a,m,r,i):
    k = 1.267*L/E[i]
    lamb_i = (1-np.sin(2*a)**2*np.sin(k*m)**2)*S[i]*r*E[i]
    return 1-N[i]/lamb_i

def d2f_dlamb2(a,m,r,i):
    k = 1.267*L/E[i]
    lamb_i = (1-np.sin(2*a)**2*np.sin(k*m)**2)*S[i]*r*E[i]
    return N[i]/lamb_i**2

'''
chian rule is used to relate f (and hence NLL) with a, m and r
'''

def dNLL_da(a,m,r):
    summ = 0
    for i in range(len(N)):
        summ += df_dlamb(a,m,r,i)*dlamb_da(a,m,r,i)
    return summ

def dNLL_dm(a,m,r):
    summ = 0
    for i in range(len(N)):
        summ += df_dlamb(a,m,r,i)*dlamb_dm(a,m,r,i)
    return summ

def dNLL_dr(a,m,r):
    summ = 0
    for i in range(len(N)):
        summ += df_dlamb(a,m,r,i)*dlamb_dr(a,m,r,i)
    return summ

def d2NLL_da2(a,m,r):
    summ = 0
    for i in range(len(N)):
        summ += d2f_dlamb2(a,m,r,i)*dlamb_da(a,m,r,i)**2+df_dlamb(a,m,r,i)*d2lamb_da2(a,m,r,i)
    return summ
    
def d2NLL_dm2(a,m,r):
    summ = 0
    for i in range(len(N)):
        summ += d2f_dlamb2(a,m,r,i)*dlamb_dm(a,m,r,i)**2+df_dlamb(a,m,r,i)*d2lamb_dm2(a,m,r,i)
    return summ

def d2NLL_dr2(a,m,r):
    summ = 0
    for i in range(len(N)):
        summ += d2f_dlamb2(a,m,r,i)*dlamb_dr(a,m,r,i)**2+df_dlamb(a,m,r,i)*d2lamb_dr2(a,m,r,i)
    return summ
    
def d2NLL_dadm(a,m,r):
    summ = 0
    for i in range(len(N)):
        summ += d2f_dlamb2(a,m,r,i)*dlamb_dm(a,m,r,i)*dlamb_da(a,m,r,i)+df_dlamb(a,m,r,i)*d2lamb_dadm(a,m,r,i)
    return summ

def d2NLL_drda(a,m,r):
    summ = 0
    for i in range(len(N)):
        summ += d2f_dlamb2(a,m,r,i)*dlamb_dr(a,m,r,i)*dlamb_da(a,m,r,i)+df_dlamb(a,m,r,i)*d2lamb_drda(a,m,r,i)
    return summ

def d2NLL_drdm(a,m,r):
    summ = 0
    for i in range(len(N)):
        summ += d2f_dlamb2(a,m,r,i)*dlamb_dm(a,m,r,i)*dlamb_dr(a,m,r,i)+df_dlamb(a,m,r,i)*d2lamb_drdm(a,m,r,i)
    return summ

'''
having all the derivatives ready, now can define the Hessian matrix and the gradient
'''

def H(a,m,r):
    Hessian_matrix = np.array([[d2NLL_da2(a,m,r),d2NLL_dadm(a,m,r),d2NLL_drda(a,m,r)],
                               [d2NLL_dadm(a,m,r),d2NLL_dm2(a,m,r),d2NLL_drdm(a,m,r)],
                               [d2NLL_drda(a,m,r),d2NLL_drdm(a,m,r),d2NLL_dr2(a,m,r)]])        
    return Hessian_matrix

def grad(a,m,r):
    gradient = np.array([dNLL_da(a,m,r),dNLL_dm(a,m,r),dNLL_dr(a,m,r)])
    return gradient

'''
iterate to simultaneously minimize NLL w.r.t both a, m and r
'''

def newton_method(a0,m0,r0):   
    x0 = np.array([a0,m0,r0])
    x1 = x0 - np.dot(np.linalg.inv(H(a0,m0,r0)),grad(a0,m0,r0))
    x0_modulus = np.sqrt(x0[0]**2+x0[1]**2+x0[2]**2)
    x1_modulus = np.sqrt(x1[0]**2+x1[1]**2+x1[2]**2)
    n = 0
    while abs(x1_modulus - x0_modulus) > 1e-10:
        n += 1
        print('iteration %s:'%(n))
        x0 = x1
        x1 = x0 - np.dot(np.linalg.inv(H(x0[0],x0[1],x0[2])),grad(x0[0],x0[1],x0[2]))
        x0_modulus = np.sqrt(x0[0]**2+x0[1]**2+x0[2]**2)
        x1_modulus = np.sqrt(x1[0]**2+x1[1]**2+x1[2]**2)
        print(x1,'\n')
    return x1[0],x1[1],x1[2]

'''
use parameters obtained from previous methods
errors are estimated from the covariance matrix
'''

a,m,r = newton_method(0.71768,0.00233,1.02957)
cov = np.linalg.inv(H(a,m,r))
err_a = np.sqrt(cov[0,0])
err_m = np.sqrt(cov[1,1])
err_r = np.sqrt(cov[2,2])
print('theta =',a,'+-',err_a)
print('mdiff2 =',m,'+-',err_m)
print('alpha =',r,'+-',err_r)
#%%
'''
use parameters just obtanined from above to fit the data histogram
'''

def get_lamb(a,m,r,Ei,Si):
    val = (1-np.sin(2*a)**2*np.sin(1.267*m*L/Ei)**2)*Si*r*Ei
    return val

lamb_list = []
for i in range(len(N)):
    lamb_list.append(get_lamb(a,m,r,E[i],S[i]))
    
bins = np.arange(0,10,0.05)

plt.figure()
plt.grid()
plt.bar(bins,N,align='edge',width=0.05,label='simulated T2K data')
plt.plot(E,lamb_list,'.-',color='#ff7f0e',label='NLL fit')
plt.legend()
plt.xlabel('Energy (GeV)')
plt.ylabel('Number of events')
plt.title('Fit by $\\theta_{23} = 0.6732,\Delta m_{23}^{2}=2.665\\times10^{-3},\\alpha=0.9780$ \nextracted from minimising the NLL likelihood')
plt.show()
#%%
def get_nll(a,m,r):
    nll = 0
    for i in range(len(N)):
        E = 0.025+0.05*i
        lamb = get_lamb(a,m,r,E,S[i])
        if N[i] == 0:
            nll += lamb
        else:
            nll += lamb-N[i]+N[i]*np.log(N[i]/lamb)
    return nll

print('minimum value of NLL =',get_nll(a,m,r))