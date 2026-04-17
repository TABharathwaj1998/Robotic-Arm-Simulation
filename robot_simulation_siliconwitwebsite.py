import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

class Robot2R:
    """Complete 2R planar robot simulator.

    Combines FK (Lesson 1), IK (Lesson 2), Jacobian (Lesson 4),
    and trajectory planning (Lesson 5) into one simulation class.
    """

    def __init__(self, L1=1.0, L2=0.8, base=(0.0, 0.0),
                 joint_limits=((-np.pi, np.pi), (-np.pi + 0.1, np.pi - 0.1))):
        # Link parameters
        self.L1 = L1
        self.L2 = L2
        self.base = np.array(base)

        # Joint limits: [(theta1_min, theta1_max), (theta2_min, theta2_max)]
        self.joint_limits = joint_limits

        # Current state
        self.q = np.array([0.0, 0.0])   # joint angles [theta1, theta2]
        self.qd = np.array([0.0, 0.0])  # joint velocities

        # Workspace bounds
        self.r_outer = L1 + L2
        self.r_inner = abs(L1 - L2)

        # Trajectory storage
        self.trail_x = []
        self.trail_y = []

        # Singularity threshold
        self.singularity_threshold = 0.05

    # ---- Kinematics (Lessons 1 and 2) ----

    def forward_kinematics(self, q=None):
        """Compute end-effector position from joint angles."""
        if q is None:
            q = self.q
        t1, t2 = q
        # Elbow position
        x1 = self.base[0] + self.L1 * np.cos(t1)
        y1 = self.base[1] + self.L1 * np.sin(t1)
        # End-effector position
        x2 = x1 + self.L2 * np.cos(t1 + t2)
        y2 = y1 + self.L2 * np.sin(t1 + t2)
        return np.array([x2, y2]), np.array([x1, y1])

    def inverse_kinematics(self, target, elbow_up=True):
        """Solve IK for target (x, y). Returns joint angles or None."""
        px = target[0] - self.base[0]
        py = target[1] - self.base[1]
        r_sq = px**2 + py**2
        r = np.sqrt(r_sq)

        # Workspace check
        if r > self.r_outer - 0.01 or r < self.r_inner + 0.01:
            return None  # outside reachable workspace

        cos_t2 = (r_sq - self.L1**2 - self.L2**2) / (2 * self.L1 * self.L2)
        cos_t2 = np.clip(cos_t2, -1.0, 1.0)

        sin_t2 = np.sqrt(1 - cos_t2**2)
        if not elbow_up:
            sin_t2 = -sin_t2

        t2 = np.arctan2(sin_t2, cos_t2)
        t1 = np.arctan2(py, px) - np.arctan2(
            self.L2 * sin_t2, self.L1 + self.L2 * cos_t2
        )

        q_sol = np.array([t1, t2])

        # Joint limit check
        if not self._check_joint_limits(q_sol):
            return None

        return q_sol

    # ---- Jacobian (Lesson 4) ----

    def jacobian(self, q=None):
        """Compute 2x2 Jacobian matrix."""
        if q is None:
            q = self.q
        t1, t2 = q
        s1 = np.sin(t1)
        c1 = np.cos(t1)
        s12 = np.sin(t1 + t2)
        c12 = np.cos(t1 + t2)

        J = np.array([
            [-self.L1 * s1 - self.L2 * s12, -self.L2 * s12],
            [ self.L1 * c1 + self.L2 * c12,  self.L2 * c12]
        ])
        return J

    def manipulability(self, q=None):
        """Compute manipulability measure: |det(J)|."""
        J = self.jacobian(q)
        return abs(np.linalg.det(J))

    def is_near_singularity(self, q=None):
        """Check if configuration is near a singularity."""
        return self.manipulability(q) < self.singularity_threshold

    # ---- Trajectory Planning (Lesson 5) ----

    def cubic_trajectory(self, q_start, q_end, num_steps):
        """Generate cubic polynomial trajectory in joint space.

        Boundary conditions: zero velocity at start and end.
        q(t) = a0 + a1*t + a2*t^2 + a3*t^3, t in [0, 1]
        With q(0)=q_start, q(1)=q_end, qd(0)=0, qd(1)=0:
            a0 = q_start, a1 = 0, a2 = 3*(q_end-q_start), a3 = -2*(q_end-q_start)
        """
        trajectory = []
        for i in range(num_steps):
            t = i / max(num_steps - 1, 1)
            # Cubic with zero-velocity endpoints
            s = 3 * t**2 - 2 * t**3
            q = q_start + s * (q_end - q_start)
            trajectory.append(q.copy())
        return trajectory

    def plan_pick_and_place(self, pick_pos, place_pos, home_q=None,
                            steps_per_segment=30):
        """Plan a complete pick-and-place trajectory.

        Sequence: home -> pick -> (lift) -> place -> home
        """
        if home_q is None:
            home_q = np.array([np.pi / 4, np.pi / 4])
        
        q_pick = self.inverse_kinematics(pick_pos)
        q_place = self.inverse_kinematics(place_pos)

        if q_pick is None or q_place is None:
            return None  # unreachable target
        
        # Build trajectory segments
        traj = []
        #traj += self.cubic_trajectory(home_q, q_pick, steps_per_segment)
        traj += self.cubic_trajectory(q_pick, q_place, steps_per_segment)
        #traj += self.cubic_trajectory(q_place, home_q, steps_per_segment)

        print("Target Pick:", pick_position)
        print("Target Place:", place_position)
        print("Pick Position ", q_pick)
        print("Place Position ", q_place)

        return traj

    # ---- Safety and Constraints ----

    def _check_joint_limits(self, q):
        """Return True if all joints are within limits."""
        for i, (lo, hi) in enumerate(self.joint_limits):
            if q[i] < lo or q[i] > hi:
                return False
        return True

    def clamp_joints(self, q):
        """Clamp joint angles to their limits."""
        q_clamped = q.copy()
        for i, (lo, hi) in enumerate(self.joint_limits):
            q_clamped[i] = np.clip(q[i], lo, hi)
        return q_clamped

    def is_in_workspace(self, target):
        """Check if a target position is within the reachable workspace."""
        dx = target[0] - self.base[0]
        dy = target[1] - self.base[1]
        r = np.sqrt(dx**2 + dy**2)
        return self.r_inner < r < self.r_outer

    # ---- State Update ----

    def set_configuration(self, q):
        """Set joint angles (with limit enforcement)."""
        self.q = self.clamp_joints(q)
        ee, _ = self.forward_kinematics()
        self.trail_x.append(ee[0])
        self.trail_y.append(ee[1])

    def reset_trail(self):
        """Clear the trajectory trail."""
        self.trail_x = []
        self.trail_y = []

# Create robot
robot = Robot2R(L1=1.0, L2=0.8, base=(0.0, 0.0))

# Define pick-and-place task
pick_position = np.array([1.2, 0.8])
place_position = np.array([-0.5, 1.0])
home_angles = np.array([np.pi / 4, np.pi / 4])

# Plan trajectory
trajectory = robot.plan_pick_and_place(
    pick_position, place_position,
    home_q=home_angles, steps_per_segment=40
)

if trajectory is None:
    print("Error: one or more targets are unreachable.")
else:
    # Set up figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title('2R Robot Pick-and-Place Simulation')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    # Draw workspace boundary
    theta_ws = np.linspace(0, 2 * np.pi, 200)
    ax.plot(robot.r_outer * np.cos(theta_ws),
            robot.r_outer * np.sin(theta_ws),
            'g--', alpha=0.3, linewidth=1, label='Outer workspace')
    ax.plot(robot.r_inner * np.cos(theta_ws),
            robot.r_inner * np.sin(theta_ws),
            'r--', alpha=0.3, linewidth=1, label='Inner workspace')

    # Mark pick and place locations
    ax.plot(*pick_position, 'rv', markersize=12, label='Pick')
    ax.plot(*place_position, 'bs', markersize=12, label='Place')

    # Initialize plot elements
    arm_line, = ax.plot([], [], 'o-', color='#2c3e50', linewidth=3,
                        markersize=8, markerfacecolor='#e74c3c')
    trail_line, = ax.plot([], [], '-', color='#3498db', linewidth=1.5,
                          alpha=0.6)
    frame_text = ax.text(-2.3, 2.2, '', fontsize=10)

    ax.legend(loc='upper right')

    robot.reset_trail()
    robot.set_configuration(home_angles)

    def update(frame):
        q = trajectory[frame]
        robot.set_configuration(q)

        ee, elbow = robot.forward_kinematics()
        base = robot.base

        print("End-Effector ", ee)


        arm_line.set_data([base[0], elbow[0], ee[0]],
                          [base[1], elbow[1], ee[1]])
        trail_line.set_data(robot.trail_x, robot.trail_y)

        # Label the phase
        steps = 40
        if frame < steps:
            phase = 'Moving to Pick'
        elif frame < 2 * steps:
            phase = 'Moving to Place'
        else:
            phase = 'Returning Home'
        frame_text.set_text(f'Frame {frame}/{len(trajectory)-1}  |  {phase}')

        return [arm_line, trail_line, frame_text]

    anim = FuncAnimation(fig, update, frames=len(trajectory),
                         interval=1000, blit=True, repeat=True)
    plt.tight_layout()
    plt.show()