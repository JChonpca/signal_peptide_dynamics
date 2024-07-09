# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 09:50:12 2024

@author: JChonpca_Huang
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 23:29:37 2024

@author: JChonpca_Huang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import r2_score

# scale = MinMaxScaler()

# scale = scale.fit(data_f_t)

# data_f_t = scale.transform(data_f_t)*2

# data_f_m = scale.transform(data_f_m)*2

# data_p = (data_f_t - data_f_m)/data_od

# data_pm = data_f_m

def sec(y, t, miumax, ks, xvmax, kd, kt, xtmax, kxd, yxvs, kpn , kds, ypxv, kpng, kp, 
        k1, kd1, k2, kd2, alpha, kd3, beta, kd4, gama, kd5, d1, d2, d3):
    
# def sec(y, t,  
#         k1, kd1, k2, kd2, alpha, kd3, beta, kd4, gama, kd5):

    
    # global ml_call
    
    
    
    xv, xd, s, p, lys, mrna, pc, pp, pl, pm  = y
    
    
    xv = max(0, xv)
    
    xd = max(0, xd)
    
    s = max(0, s)
    
    p = max(0, p)
    
    
    mrna = max(0, mrna)
    
    pc = max(0, pc)
    
    pp = max(0, pp)
    
    pl = max(0, pl)
    
    pm = max(0, pm)

    
    dxv = (miumax*s/(ks+s)*(1-xv/xvmax) - kd)*xv
    
    dxd = kt*xv*(1-(xv+xd)/xtmax) - (miumax*s/(ks+s)*(1-xv/xvmax) - kd)*xv - kxd*xd
    
    ds = - miumax*s/(ks+s)*(1-xv/xvmax)/yxvs*xv - kpn*xv - kds*s
        
    dp = miumax*s/(ks+s)*(1-xv/xvmax)/ypxv*xv + kpng*xv - kp*p
    
    dlys = kxd*xd
    
    dmrna = k1 - kd1*mrna
    
    dpc = k2*mrna - kd2*pc - alpha*pc
    
    dpp = alpha*pc - kd3*pp - beta*pp
    
    dpl = beta*pp - kd4*pl - gama*pl - miumax*s/(ks+s)*(1-xv/xvmax)*xv/(xv+xd)*pl- kxd*xd/(xv+xd)*pl
    
    dpm = gama*pl*xv*d1 + miumax*s/(ks+s)*(1-xv/xvmax)*xv*pl*d2 + kxd*xd*pl*d3

    

    return [dxv, dxd, ds, dp, dlys, dmrna, dpc, dpp, dpl, dpm]
    

simulate_t = np.linspace(0,10,100)

# miumax, ks, xvmax, kd, kt, xtmax, kxd, yxvs, kds, ypxv, kp = 0.139, 0.000142, 1.14, 0.029, 0.103, 1, 0.5 ,1.62e3, 7.58e-3, 0 , 0


test_index = 4

log_ph_start = 0

log_ph = 11


# res = odeint(ml, init, simulate_t, args=(miumax, ks, xvmax, kd, kt, xtmax, kxd, yxvs, kds, ypxv, kp))

# plt.plot(res)

# plt.show()


location = []

for i in t[log_ph_start:log_ph]:
    
    location.append(abs((simulate_t-i)).tolist().index(abs((simulate_t-i)).min()))

def goal(x):
    
    global simulate_t
    global location
    global test_index
    # global init
    global sec
    global res
    global shabi
    
    shabi = x.copy()
    
    init = [data_od[log_ph_start,test_index]*0.95, data_od[log_ph_start,test_index]*0.05, 10, data_f_t[log_ph_start,test_index], 0, 0, 0, 0, data_p[log_ph_start, test_index], data_pm[log_ph_start, test_index]]
    
    miumax, ks, xvmax, kd, kt, xtmax, kxd, yxvs, kpn , kds, ypxv, kpng, kp, k1, kd1, k2, kd2, alpha, kd3, beta, kd4, gama, kd5, d1, d2, d3 = x
    
    kpn = 0
    
    kpng = 0

    
    res = odeint(sec, init, simulate_t, args=(miumax, ks, xvmax, kd, kt, xtmax, kxd, yxvs, kpn , kds, ypxv, kpng, kp, k1, kd1, k2, kd2, alpha, kd3, beta, kd4, gama, kd5, d1, d2, d3))
    
    loss = 0
    
    loss += (((res[location][:,0] + res[location][:,1]) - data_od[log_ph_start:log_ph,test_index])**2).mean()
    
    loss += ((res[location][:,3] - data_f_t[log_ph_start:log_ph,test_index])**2).mean()
    
    loss += ((res[location][:,8]  - data_p[log_ph_start:log_ph,test_index])**2).mean()
    
    loss += ((res[location][:,9] - data_pm[log_ph_start:log_ph,test_index])**2).mean()

    if (x<0).sum():
        
        return (x<0).sum()*10**100

    elif np.isnan(res).sum():
        
        return np.isnan(res).sum()*10**50
    
    else:
        
        print(loss)
        
        return loss


lb_array = [0]*26

ub_array = [4]*23 + [1]*3

from sko.GA import GA

ga = GA(func=goal, n_dim=26, size_pop=200, max_iter=200, prob_mut=0.1, lb=lb_array, ub=ub_array, precision=1e-100)



import time

a = time.time()

best_x, best_y = ga.run()

b = time.time()

print('best_x:', best_x, '\n', 'best_y:', best_y)

print('time spend:', b-a, 's')



from scipy.optimize import minimize

bounds = [(0, 4), (0, 4), (0, 4), (0, 4), (0, 4), (0, 4), (0, 4), (0, 4), (0, 4), (0, 4), (0, 4), (0, 4), (0, 4), (0, 4), (0, 4), (0, 4), (0, 4), (0, 6), (0, 4), (0, 4), (0, 4), (0, 0.1), (0, 4), (0, 1), (0, 1), (0, 1)]

opt_res = minimize(goal, 
                  best_x,  
                  bounds=bounds, 
                  method='trust-constr', # 'L-BFGS-B' cannot handle constraints
                  options={"maxiter": 10000}, 
                  tol=1e-6
                  )

print(opt_res)









plt.plot(simulate_t,res[:,0]+res[:,1])
plt.scatter(t[0:log_ph],data_od[0:log_ph,test_index])
plt.show()

plt.plot(simulate_t,res[:,3])
plt.scatter(t[0:log_ph],data_f_t[0:log_ph,test_index])
plt.show()

plt.plot(simulate_t,res[:,8])
plt.scatter(t[0:log_ph],data_p[0:log_ph,test_index])
plt.show()

plt.plot(simulate_t,res[:,9])
plt.scatter(t[0:log_ph],data_pm[0:log_ph,test_index])
plt.show()



plt.plot(simulate_t,res[:,0]+res[:,1])
plt.scatter(t,data_od[:,test_index])
plt.show()

plt.plot(simulate_t,res[:,3])
plt.scatter(t,data_f_t[:,test_index])
plt.show()

plt.plot(simulate_t,res[:,8])
plt.scatter(t,data_p[:,test_index])
plt.show()

plt.plot(simulate_t,res[:,9])
plt.scatter(t,data_pm[:,test_index])
plt.show()


# plt.plot(simulate_t, res[:,0] + res[:,1])

# plt.scatter(t,data_od[:,test_index])

# plt.show()


# plt.plot(simulate_t, res[:,3])

# plt.scatter(t,data_f_t[:,test_index])

# plt.show()

# from scipy.optimize import fmin

# res=fmin(goal, best_x, maxiter=1e10, maxfun=1e500)

# print(res)

# goal(res)

# plt.plot(simulate_t, res[:,0] + res[:,1])

# plt.scatter(t,data_od[:,test_index])

# plt.show()


# plt.plot(simulate_t, res[:,3])

# plt.scatter(t,data_f_t[:,test_index])

# plt.show()