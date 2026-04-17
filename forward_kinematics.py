import numpy as np
from inversekinematics import id_solutions

import numpy as np

def forward_kinematics(L1, L2, L3, q):
    """
    Computes joint positions using FK.
    """

    q1, q2, q3 = q  # already radians

    # Base
    x0, y0, z0 = 0, 0, 0

    # Joint 1
    x1 = L2 * np.cos(q1) * np.cos(q2)
    y1 = L2 * np.sin(q1) * np.cos(q2)
    z1 = L1 + L2 * np.sin(q2)

    # Joint 2
    x2 = x1 + L3 * np.cos(q1) * np.cos(q2 + q3)
    y2 = y1 + L3 * np.sin(q1) * np.cos(q2 + q3)
    z2 = z1 + L3 * np.sin(q2 + q3)

    return np.array([x2, y2, z2]), np.array([x1, y1, z1]), np.array([x0, y0, z0])