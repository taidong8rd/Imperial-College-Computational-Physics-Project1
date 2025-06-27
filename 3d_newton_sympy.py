# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 23:46:11 2021

CID: 01702088
"""

from sympy import *
import numpy as np
import matplotlib.pyplot as plt

N,S = np.loadtxt('data.csv',delimiter=',',unpack=1)

L = 295

E = []
for i in range(len(N)):
    E.append(0.025+0.05*i)
    
def lamb(a,m,r,Ei,Si):
    val = (1-sin(2*a)**2*sin(1.267*m*L/Ei)**2)*Si*r*Ei
    return val

def f(a,m,r,Ei,Si,Ni):
    val = lamb(a,m,r,Ei,Si)-Ni+Ni*ln(Ni/lamb(a,m,r,Ei,Si))
    return val

a,m,r,Ei,Si,Ni = symbols('a m r Ei Si Ni',real=True)

df_da = diff(f(a,m,r,Ei,Si,Ni),a)
df_dm = diff(f(a,m,r,Ei,Si,Ni),m)
df_dr = diff(f(a,m,r,Ei,Si,Ni),r)

d2f_da2 = diff(f(a,m,r,Ei,Si,Ni),a,2)
d2f_dm2 = diff(f(a,m,r,Ei,Si,Ni),m,2)
d2f_dr2 = diff(f(a,m,r,Ei,Si,Ni),r,2)

d2f_dadm = diff(df_da,m)
d2f_drda = diff(df_da,r)
d2f_drdm = diff(df_dr,m)

def dNLL_da(_a,_m,_r):
    summ = 0
    for i in range(len(N)):
        _Ei = E[i]
        _Si = S[i]
        _Ni = N[i]
        summ += df_da.subs({a:_a,m:_m,r:_r,Ei:_Ei,Si:_Si,Ni:_Ni})
    return float(summ)

def dNLL_dm(_a,_m,_r):
    summ = 0
    for i in range(len(N)):
        _Ei = E[i]
        _Si = S[i]
        _Ni = N[i]
        summ += df_dm.subs({a:_a,m:_m,r:_r,Ei:_Ei,Si:_Si,Ni:_Ni})
    return float(summ)

def dNLL_dr(_a,_m,_r):
    summ = 0
    for i in range(len(N)):
        _Ei = E[i]
        _Si = S[i]
        _Ni = N[i]
        summ += df_dr.subs({a:_a,m:_m,r:_r,Ei:_Ei,Si:_Si,Ni:_Ni})
    return float(summ)

def d2NLL_da2(_a,_m,_r):
    summ = 0
    for i in range(len(N)):
        _Ei = E[i]
        _Si = S[i]
        _Ni = N[i]
        summ += d2f_da2.subs({a:_a,m:_m,r:_r,Ei:_Ei,Si:_Si,Ni:_Ni})
    return float(summ)        
        
def d2NLL_dm2(_a,_m,_r):
    summ = 0
    for i in range(len(N)):
        _Ei = E[i]
        _Si = S[i]
        _Ni = N[i]
        summ += d2f_dm2.subs({a:_a,m:_m,r:_r,Ei:_Ei,Si:_Si,Ni:_Ni})
    return float(summ)

def d2NLL_dr2(_a,_m,_r):
    summ = 0
    for i in range(len(N)):
        _Ei = E[i]
        _Si = S[i]
        _Ni = N[i]
        summ += d2f_dr2.subs({a:_a,m:_m,r:_r,Ei:_Ei,Si:_Si,Ni:_Ni})
    return float(summ)

def d2NLL_dadm(_a,_m,_r):
    summ = 0
    for i in range(len(N)):
        _Ei = E[i]
        _Si = S[i]
        _Ni = N[i]
        summ += d2f_dadm.subs({a:_a,m:_m,r:_r,Ei:_Ei,Si:_Si,Ni:_Ni})
    return float(summ)

def d2NLL_drda(_a,_m,_r):
    summ = 0
    for i in range(len(N)):
        _Ei = E[i]
        _Si = S[i]
        _Ni = N[i]
        summ += d2f_drda.subs({a:_a,m:_m,r:_r,Ei:_Ei,Si:_Si,Ni:_Ni})
    return float(summ)

def d2NLL_drdm(_a,_m,_r):
    summ = 0
    for i in range(len(N)):
        _Ei = E[i]
        _Si = S[i]
        _Ni = N[i]
        summ += d2f_drdm.subs({a:_a,m:_m,r:_r,Ei:_Ei,Si:_Si,Ni:_Ni})
    return float(summ)

def H(_a,_m,_r):
    Hessian_matrix = np.array([[d2NLL_da2(_a,_m,_r),d2NLL_dadm(_a,_m,_r),d2NLL_drda(_a,_m,_r)],
                               [d2NLL_dadm(_a,_m,_r),d2NLL_dm2(_a,_m,_r),d2NLL_drdm(_a,_m,_r)],
                               [d2NLL_drda(_a,_m,_r),d2NLL_drdm(_a,_m,_r),d2NLL_dr2(_a,_m,_r)]])        
    return Hessian_matrix

def grad(_a,_m,_r):
    gradient = np.array([dNLL_da(_a,_m,_r),dNLL_dm(_a,_m,_r),dNLL_dr(_a,_m,_r)])
    return gradient

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

_a,_m,_r = newton_method(0.71768,0.00233,1.02957)
cov = np.linalg.inv(H(_a,_m,_r))
err_a = np.sqrt(cov[0,0])
err_m = np.sqrt(cov[1,1])
err_r = np.sqrt(cov[2,2])
print('theta =',_a,'+-',err_a)
print('mdiff2 =',_m,'+-',err_m)
print('alpha =',_r,'+-',err_r)
#%%
lamb_list = []
for i in range(len(N)):
    lamb_list.append(float(lamb(_a,_m,_r,E[i],S[i])))
    
bins = np.arange(0,10,0.05)

plt.figure()
plt.bar(bins,N,align='edge',width=0.05)
plt.plot(E,lamb_list,'.-',color='orange')
plt.show()
