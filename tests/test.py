from integration import Rkf54
import math
import matplotlib.pyplot as plt
import numpy as np
import time

if __name__=="__main__":
    def eom_cs(t,y,mu,u=np.zeros(3)):
        dydt = np.zeros(6)
        r = np.linalg.norm(y[:3])
        p1 = -1.0*mu/r**3
        u = 1e-1 * y[:3]/r
        u = np.cross(np.array([0., 0., 1.]), u)
        dydt[0] = y[3]
        dydt[1] = y[4]
        dydt[2] = y[5]
        dydt[3] = p1*y[0]+u[0]
        dydt[4] = p1*y[1]+u[1]
        dydt[5] = p1*y[2]+u[2]
        return dydt

    def eom_mee(t,y,mu,u=np.array([0., 0., 0.])):
        dydt = np.zeros(6)
        u[1] = 1e-1
        p1 = math.cos(y[5])
        p2 = math.sin(y[5])
        w = 1.0+y[1]*p1+y[2]*p2
        s2 = 1.0+y[4]**2+y[5]**2
        p3 = math.sqrt(y[0]/mu)
        p4 = w+1
        p5 = y[3]*p2-y[4]*p1
        p6 = p5*y[2]*u[2]/w 
        p8 = p3*s2*u[2]/(2.0*w)
        dydt[0] = 2.0*y[0]*p3*u[1]/w
        dydt[1] = p3*( u[0]*p2+(p4*p1+y[1])*u[1]/w-p6)
        dydt[2] = p3*(-u[0]*p1+(p4*p2+y[2])*u[1]/w+p6)
        dydt[3] = p8*p1
        dydt[4] = p8*p2
        dydt[5] = np.sqrt(mu*y[0])*(w/y[0])**2+p3*p5*u[2]/w
        return dydt

    def mee2cs(y):
        n = np.shape(y)[1]
        ycs = np.zeros((2,n))
        for i in range(n):
            s2 = 1.0+y[3,i]**2+y[4,i]**2
            a2 = y[3,i]**2-y[4,i]**2
            p1 = math.cos(y[5,i])
            p2 = math.sin(y[5,i])
            w = 1.0+y[1,i]*p1+y[2,i]*p2 
            r = y[0,i]/w
            ycs[0,i] = (r/s2)*(p1+a2*p1+2*y[3,i]*y[4,i]*p2)
            ycs[1,i] = (r/s2)*(p2-a2*p2+2*y[3,i]*y[4,i]*p1)
        return ycs
            

    mu = 4*math.pi**2
    y0 = np.array([1.0, 0.0, 0.0, 0.0, 2*math.pi, 0.0])
    tspan = (0.0, 10)
    sol0 = Rkf54(eom_cs, tspan, y0, args=(mu,), tol=1e-8, verbose=1)
    t_start = time.time()
    sol0.integrate()
    t_end = time.time()
    print(f"CS Duration = {t_end-t_start}")
    y1 = sol0.y
    
    y0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    sol = Rkf54(eom_mee, tspan, y0, args=(mu,), tol=1e-8, verbose=2)
    t_start = time.time()
    sol.integrate()
    t_end = time.time()
    print(f"MEE Duration = {t_end-t_start}")

    fig, ax = plt.subplots(1)
    ycs = mee2cs(sol.y)
    ax.plot(y1[0,:], y1[1,:], 'b-')
    ax.plot(ycs[0,:], ycs[1,:], 'k--')
    plt.show()
