import numpy as np
from numpy.fft import fft, ifft, fftfreq
from tqdm import tqdm

def kutz(x, u0, tmax = 10, nu = 0.05, N = 2048, h = 0.01, saved_steps = 1000):

    # N is the spatial discretisation and h is the time step
    domain_length = x[-1] - x[0]
    
    v = fft(u0(x))

    # spatial grid and initial conditions
    k = 2 * np.pi / domain_length * (np.r_[np.arange(0, N/2), np.array([0]), np.arange(-N/2+1, 0)]).astype('float128')
    L = k**2 - nu*k**4
    
    exp1 = np.exp(h*L)
    exp2 = np.exp(h*L/2)
    
    # RK integration quadrature
    M = 16
    r = np.exp(1j*np.pi*(np.arange(1, M+1)-0.5) / M)
    LR = h*np.repeat([L], M, axis=0).T + np.repeat([r], N, axis=0)
    Q = h*np.real(np.mean((np.exp(LR/2)-1)/LR, axis=1))
    f1 = h*np.real(np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=1))
    f2 = h*np.real(np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=1))
    f3 = h*np.real(np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis=1))

    step_max = round(tmax/h)
    step_plt = int(tmax/(saved_steps*h))
    g = -0.5j*k
    uu = np.array([u0(x)])
    tt = 0

    for step in tqdm(range(1, step_max+1), 'Solving KS equation with nu={:.2f}'.format(nu)):
        t = step*h
        Nv = g*fft(np.real(ifft(v))**2)
        a = exp2*v + Q*Nv
        Na = g*fft(np.real(ifft(a))**2)
        b = exp2*v + Q*Na
        Nb = g*fft(np.real(ifft(b))**2)
        c = exp2*a + Q*(2*Nb - Nv)
        Nc = g*fft(np.real(ifft(c))**2)
        v = exp1*v + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3
        if step % step_plt == 0:
            u = np.real(ifft(v))
            uu = np.append(uu, np.array([u]), axis=0)
            tt = np.hstack((tt, t))
            
    return x, tt, uu


def renshaw(x, u0, tmax = 10, nu = 0.05, N = 2048, h = 0.01, saved_steps = 1000, verbose = False):

    # N is the spatial discretisation and h is the time step
    domain_length = x[-1] - x[0]

    v = np.fft.fft(u0(x))
    
    # scalars for ETDRK4
    k = 2 * np.pi / domain_length * np.transpose(np.conj(np.concatenate((np.arange(0, N/2), np.array([0]), np.arange(-N/2+1, 0))))) 
    L = k**2 - nu * k**4
    E = np.exp(h*L)
    E_2 = np.exp(h*L/2)
    
    M = 16
    r = np.exp(1j*np.pi*(np.arange(1, M+1)-0.5) / M)
    LR = h*np.transpose(np.repeat([L], M, axis=0)) + np.repeat([r], N, axis=0)
    Q = h*np.real(np.mean((np.exp(LR/2)-1)/LR, axis=1))
    f1 = h*np.real(np.mean((-4-LR+np.exp(LR)*(4-3*LR+LR**2))/LR**3, axis=1))
    f2 = h*np.real(np.mean((2+LR+np.exp(LR)*(-2+LR))/LR**3, axis=1))
    f3 = h*np.real(np.mean((-4-3*LR-LR**2+np.exp(LR)*(4-LR))/LR**3, axis=1))
    
    # main loop
    uu = np.array([u0(x)])
    tt = 0
    nmax = round(tmax/h)
    nplt = int((tmax/saved_steps)/h)
    g = -0.5j*k
    
    for n in tqdm(range(1, nmax+1), 'Solving KS equation with nu={:.6f}'.format(nu)) if verbose else range(1, nmax+1):
        t = n*h
        Nv = g*np.fft.fft(np.real(np.fft.ifft(v))**2)
        a = E_2*v + Q*Nv
        Na = g*np.fft.fft(np.real(np.fft.ifft(a))**2)
        b = E_2*v + Q*Na
        Nb = g*np.fft.fft(np.real(np.fft.ifft(b))**2)
        c = E_2*a + Q*(2*Nb-Nv)
        Nc = g*np.fft.fft(np.real(np.fft.ifft(c))**2)
        v = E*v + Nv*f1 + 2*(Na+Nb)*f2 + Nc*f3
        if n%nplt == 0:
            u = np.real(np.fft.ifft(v))
            uu = np.append(uu, np.array([u]), axis=0)
            tt = np.hstack((tt, t))
            # print(np.sum(np.abs(np.fft.fft(u))**2))
            
    return x, tt, uu