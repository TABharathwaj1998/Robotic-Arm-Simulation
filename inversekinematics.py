import numpy as np

def theta_calculation(a,b):
    ab_sqrt = np.sqrt(a**2+b**2)
    phi = np.arctan2(b,a)
    return phi, ab_sqrt

def id_solutions(px,py,pz):
    px = float(px)
    py = float(py)
    pz = float(pz)
    theta1 = np.arctan2(py,px)

    b = float(1.72965)
    a = float(0.39)
    c = px**2 + py**2 + pz**2 - (0.64 * ( (np.cos(theta1) * px) + (np.sin(theta1) * py) )) - 1.674994
    phi, ab_sqrt = theta_calculation(a,b)
    
    D = c/ab_sqrt
    D = np.clip(D, -1, 1)
    theta3 = float(phi + np.arccos(D))

    a = b = c = phi = ab_sqrt = 0
    COS = np.cos(theta3)
    SIN = np.sin(theta3)

    a = (0.2 * SIN) - (0.887 * COS)
    b = (0.887 * SIN) + (0.2 * COS) + 975
    c = pz

    phi, ab_sqrt = theta_calculation(a,b)

    D = c/ab_sqrt
    D = np.clip(D, -1, 1)
    theta2 = float(phi + np.arccos(D))

    q_sol = np.array([theta1, theta2, theta3])
    return q_sol 
