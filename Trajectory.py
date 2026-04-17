import numpy as np

def cubic_trajectory(q_start, q_end, num_steps):
    """Generate cubic polynomial trajectory in joint space.

    Boundary conditions: zero velocity at start and end.
    q(t) = a0 + a1*t + a2*t^2 + a3*t^3, t in [0, 1]
    With q(0)=q_start, q(1)=q_end, qd(0)=0, qd(1)=0:
        a0 = q_start, a1 = 0, a2 = 3*(q_end-q_start), a3 = -2*(q_end-q_start) 0, tf + dt, dt
    """
    trajectory = []
    for i in range(num_steps):
        t = i / max(num_steps - 1, 1)
        # Cubic with zero-velocity endpoints
        s = 3 * t**2 - 2 * t**3
        q = q_start + s * (q_end - q_start)
        trajectory.append(q.copy())
    return trajectory