from inversekinematics import id_solutions
from Trajectory import cubic_trajectory
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from forward_kinematics import forward_kinematics

L1 = 1
L2 = 1
L3 = 1
pick_position = np.array([1, 1, 0])
place_position = np.array([-1, 1, 0])

def plan_pick_and_place(pick_position, place_position, home_q=None, steps_per_segment=30):
    """Plan a complete pick-and-place trajectory.

    Sequence: home -> pick -> (lift) -> place -> home
    """
    if home_q is None:
        home_q = np.array([0, 0, 0])

    q_pick = id_solutions(*pick_position)
    q_place = id_solutions(*place_position)

    if q_pick is None or q_place is None:
        return None  # unreachable target

    # Build trajectory segments
    traj = []
    traj += cubic_trajectory(home_q, q_pick, steps_per_segment)
    traj += cubic_trajectory(q_pick, q_place, steps_per_segment)
    
    print("Target Pick:", pick_position)
    print("Target Place:", place_position)
    print("Pick Position ", q_pick)
    print("Place Position ", q_place)

    return traj

home_angles = np.array([0, 0, 0])
# Plan trajectory
trajectory = plan_pick_and_place(pick_position, place_position, home_q=home_angles, steps_per_segment=40)



# Initialize plot elements
fig = plt.figure()
Axis = plt.axes(projection = "3d",
                xlim = (-5,5),
                ylim = (0,6),
                zlim = (0,6))
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("Robotic Arm Movement")
#Plots the intial point before performing the update of values using Funcanimation
line, = plt.plot(*pick_position, marker = 'o', mec = 'w', label = "Pick")
endposition, = plt.plot(*place_position, marker = '^', mec = 'w', label = "Place")
arm_line, = plt.plot([], [], 'o-', color="#2F00EA", linewidth=3, markersize=8, markerfacecolor="#4ae73c", label = "Robotic Arm")
plt.legend(loc='upper right')

def update(frame):
        q = trajectory[frame]

        ee, elbow, base = forward_kinematics(L1,L2,L3,q)

        print("End-Effector ", ee)

        arm_line.set_data_3d([base[0], elbow[0], ee[0]],[base[1], elbow[1], ee[1]],[base[2], elbow[2], ee[2]])

        return arm_line

anim = FuncAnimation(fig, update, frames=len(trajectory), interval=100, blit=False, repeat=True)
plt.tight_layout()

plt.show()
