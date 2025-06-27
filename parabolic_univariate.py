# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 12:55:58 2021

CID: 01702088
"""
import numpy as np
import matplotlib.pyplot as plt

theta = np.pi/4
mdiff2 = 2.4e-3
L = 295
def prob(E,theta=np.pi/4,mdiff2=2.4e-3,L=295):
    return 1-((np.sin(2*theta))**2)*((np.sin(1.267*mdiff2*L/E))**2)

def get_lamb(theta=np.pi/4,mdiff2=2.4e-3,alpha=None):
    lamb_list = []
    for i in range(len(simu_m_list)):
        E = 0.025+0.05*i
        if alpha == None:
            lamb = simu_m_list[i]*prob(E,theta,mdiff2,L)
        else:    
            lamb = simu_m_list[i]*prob(E,theta,mdiff2,L)*alpha*E
        lamb_list.append(lamb)
    return lamb_list

def NLL(theta_list=[np.pi/4],mdiff2_list=[2.4e-3],alpha_list=[None]):
    nll_list = []
    for theta in theta_list:
        for mdiff2 in mdiff2_list:
            for alpha in alpha_list:
                lamb_list = get_lamb(theta,mdiff2,alpha)
                nll = 0
                for i in range(len(lamb_list)):
                    if m_list[i] == 0:
                        nll += lamb_list[i]
                    else:
                        nll += lamb_list[i]-m_list[i]+m_list[i]*np.log(m_list[i]/lamb_list[i])
                nll_list.append(nll)
    if len(nll_list) == 1:
        return nll_list[0]
    else:
        return nll_list

def iterate(x,target,theta=np.pi/4,mdiff2=2.4e-3,alpha=None):
    if target == 'angle':
        y = NLL(x,[mdiff2],[alpha])
    if target == 'mass':
        y = NLL([theta],x,[alpha])
    if target == 'rate':
        y = NLL([theta],[mdiff2],x)
    numerator = (x[2]**2-x[1]**2)*y[0]+(x[0]**2-x[2]**2)*y[1]+(x[1]**2-x[0]**2)*y[2]
    denominator = (x[2]-x[1])*y[0]+(x[0]-x[2])*y[1]+(x[1]-x[0])*y[2]
    x3 = 0.5*(numerator/denominator)
    x.append(x3)
    if target == 'angle':
        y3 = NLL([x3],[mdiff2],[alpha])
        y.append(y3)
    if target == 'mass':
        y3 = NLL([theta],[x3],[alpha])
        y.append(y3)
    if target == 'rate':
        y3 = NLL([theta],[mdiff2],[x3])
        y.append(y3)
    index = np.argmax(y)
    del x[index]
    del y[index]
    x.sort()
    return x,x3,y,y3
    
def parabolic_minimiser(x,target,theta=np.pi/4,mdiff2=2.4e-3,alpha=None):
    x,temp,y,y3 = iterate(x,target,theta,mdiff2,alpha)
    x,x3,y,y3 = iterate(x,target,theta,mdiff2,alpha)
    while abs(x3-temp) >= 1e-10:
        temp = x3
        x,x3,y,y3 = iterate(x,target,theta,mdiff2,alpha)
    A = y[0]/((x[0]-x[1])*(x[0]-x[2])) + y[1]/((x[1]-x[0])*(x[1]-x[2])) + y[2]/((x[2]-x[0])*(x[2]-x[1]))
    B = ((-x[1]-x[2])*y[0])/((x[0]-x[1])*(x[0]-x[2])) + ((-x[0]-x[2])*y[1])/((x[1]-x[0])*(x[1]-x[2])) + ((-x[0]-x[1])*y[2])/((x[2]-x[0])*(x[2]-x[1]))
    C = (x[1]*x[2]*y[0])/((x[0]-x[1])*(x[0]-x[2])) + (x[0]*x[2]*y[1])/((x[1]-x[0])*(x[1]-x[2])) + (x[0]*x[1]*y[2])/((x[2]-x[0])*(x[2]-x[1]))
    xleft = (-B-np.sqrt(B**2-4*A*(C-y3-0.5)))/(2*A)
    xright = (-B+np.sqrt(B**2-4*A*(C-y3-0.5)))/(2*A)
    std = (xright-xleft)/2
    return x3,std,x,[A,B,C]

def univariate(thetax,mdiff2x):
    mdiff2_min,mdiff2x = parabolic_minimiser(mdiff2x,'mass')[::2]
    nll_mdiff2_min = NLL([theta],[mdiff2_min])
    theta_min,thetax = parabolic_minimiser(thetax,'angle',mdiff2=mdiff2_min)[::2]
    nll_theta_min = NLL([theta_min],[mdiff2_min])
    n = 0
    while abs(nll_mdiff2_min - nll_theta_min) > 1e-7:
        n += 1
        mdiff2_min,std_mdiff2,mdiff2x = parabolic_minimiser(mdiff2x,'mass',theta=theta_min,mdiff2=mdiff2_min)[:3]
        nll_mdiff2_min = NLL([theta_min],[mdiff2_min])
        theta_min,std_theta,thetax = parabolic_minimiser(thetax,'angle',theta=theta_min,mdiff2=mdiff2_min)[:3]
        nll_theta_min = NLL([theta_min],[mdiff2_min])
        print('iteration %s done'%(n))
    return theta_min,mdiff2_min,std_theta,std_mdiff2
#%%
m_list,simu_m_list = np.loadtxt('data.csv',delimiter=',',unpack=1)

bins = np.arange(0,10,0.05)
plt.figure(1)
plt.bar(bins,m_list,align='edge',width=0.05)

plt.figure(2)
plt.plot(bins+0.05,prob(bins+0.05),'.-')

lamb_list = get_lamb()
plt.figure(3)
plt.bar(bins,lamb_list,align='edge',width=0.05)
#%%
thetax = [0.6,0.7,0.77]
mdiff2x = [0.0020,0.0025,0.0030]
alphax = [0.8,1,1.2]

theta_list = np.arange(0,np.pi/2,np.pi/200)
nll_list = NLL(theta_list=theta_list)
plt.figure(4)
plt.grid()
plt.plot(theta_list,nll_list,label='NLL($\\theta _{23}$)')
theta_min,coeff_list = parabolic_minimiser(thetax,"angle")[::3]
def P2(x):
    return coeff_list[0]*x**2 + coeff_list[1]*x + coeff_list[2]
plt.plot(theta_list,P2(theta_list),label='last parabolic fit')
plt.scatter(theta_min,NLL(theta_list=[theta_min]),label='minimum',color='red',marker='x')
plt.xlim((0,np.pi/2))
plt.ylim((300,1000))
plt.legend()
plt.xlabel('$\\theta _{23}$')
plt.ylabel('NLL')
plt.title('NLL as a function of $\\theta_{23}$ at $\Delta m_{23}^{2}=2.4\\times10^{-3}$ \ncross-section $\sigma=1$')

mdiff2_list = np.arange(0,0.005,0.000025)
nll_list = NLL(mdiff2_list=mdiff2_list)
plt.figure(5)
plt.grid()
plt.plot(mdiff2_list,nll_list,label='NLL($\Delta m_{23}^{2}$)')
mdiff2_min,coeff_list = parabolic_minimiser(mdiff2x,'mass')[::3]
def P2(x):
    return coeff_list[0]*x**2 + coeff_list[1]*x + coeff_list[2]
plt.plot(mdiff2_list,P2(mdiff2_list),label='last parabolic fit')
plt.scatter(mdiff2_min,NLL(mdiff2_list=[mdiff2_min]),label='minimum',color='red',marker='x')
plt.xlim((0,0.005))
plt.ylim((300,800))
plt.legend()
plt.xlabel('$\Delta m_{23}^{2}$')
plt.ylabel('NLL')
plt.title('NLL as a function of $\Delta m_{23}^{2}$ at $\\theta_{23}=\pi/4$ \ncross-section $\sigma=1$')

alpha_list = np.arange(0.01,2.01,0.02)
nll_list = NLL(alpha_list=alpha_list)
plt.figure(6)
plt.grid()
plt.plot(alpha_list,nll_list,label='NLL($\\alpha$)')
alpha_min,coeff_list = parabolic_minimiser(alphax,'rate')[::3]
def P2(x):
    return coeff_list[0]*x**2 + coeff_list[1]*x + coeff_list[2]
plt.plot(alpha_list,P2(alpha_list),label='last parabolic fit')
plt.scatter(alpha_min,NLL(alpha_list=[alpha_min]),label='minimum',color='red',marker='x')
plt.xlim((0,2))
plt.ylim((100,800))
plt.legend()
plt.xlabel('$\\alpha$')
plt.ylabel('NLL')
plt.title('NLL as a function of $\\alpha$ at $\\theta_{23}=\pi/4$ and $\Delta m_{23}^{2}=2.4\\times10^{-3}$ \ncross-section $\sigma=\\alpha E_{\\nu}$')
#%%
thetax = [0.6,0.7,0.77]
mdiff2x = [0.0020,0.0025,0.0030]
alphax = [0.8,1,1.2]
print('1D parabolic minimisation:')
print('theta =',parabolic_minimiser(thetax,"angle")[:2])
print('mdiff2 =',parabolic_minimiser(mdiff2x,'mass')[:2])
print('alpha =',parabolic_minimiser(alphax,'rate')[:2])
#%%
thetax = [0.6,0.7,0.77]
mdiff2x = [0.0020,0.0025,0.0030]
print('2D univariate method:')
a,m,a_err,m_err = univariate(thetax,mdiff2x)
print('theta =',a,'+-',a_err)
print('mdiff2 =',m,'+-',m_err)
print('minimum value of NLL =',NLL([a],[m]))