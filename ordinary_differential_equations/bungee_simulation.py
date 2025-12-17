import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from runge_kutta_4 import rk4_method

# Define the Physics
def bungee_jumper_ode(t, state):
    """
    Computes derivatives [dx/dt, dv/dt] for a bungee jumper.
    State vector: [position (x), velocity (v)]
    Coordinate system: Down is Positive (x=0 at bridge)
    """
    x, v = state
    
    # Constants
    g = 9.81        # Gravity (m/s^2)
    m = 68.1        # Mass of jumper (kg)
    cd = 0.25       # Drag coefficient (kg/m)
    k = 40.0        # Spring constant of cord (N/m)
    L = 30.0        # Length of unstretched cord (m)
    
    # Derivative of position is velocity
    dxdt = v
    
    # Derivative of velocity is Acceleration (F/m)
    # Forces: Gravity (down/positive), Drag (opposes v), Spring (up/negative)
    
    # Drag Force: Always opposes motion direction
    # F_drag = -sign(v) * cd * v^2
    f_drag = -np.sign(v) * cd * (v**2)
    
    # Spring Force: Only acts if stretched (x > L)
    if x > L:
        f_spring = -k * (x - L)
    else:
        f_spring = 0.0 # Cord is slack
        
    # Total Acceleration = (F_grav + F_drag + F_spring) / m
    dvdt = g + (f_drag + f_spring) / m
    
    return np.array([dxdt, dvdt])

# Run the Simulation
# Initial State: x=0 (at bridge), v=0 (at rest)
y0 = [0.0, 0.0] 
t_span = (0, 50) # 50 seconds simulation
h = 0.1          # Time step

t_data, y_data = rk4_method(bungee_jumper_ode, y0, t_span, h)

# Extract position (x) and velocity (v) columns
position = y_data[:, 0]
velocity = y_data[:, 1]

# The "Wow" Animation
# Setup the figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), gridspec_kw={'width_ratios': [1, 2]})

# Left Plot: Visual representation of jumper
ax1.set_xlim(-1, 1)
ax1.set_ylim(max(position) + 10, -5) # Inverted Y-axis so "down" is down
ax1.set_title("Jumper View")
ax1.set_ylabel("Distance Fallen (m)")
bridge_line = ax1.axhline(0, color='black', linewidth=4, label='Bridge')
cord_line, = ax1.plot([], [], 'k-', lw=2) # The bungee cord
jumper_dot, = ax1.plot([], [], 'ro', markersize=10) # The person

# Right Plot: The Data Trace
ax2.set_xlim(0, 50)
ax2.set_ylim(min(position)-10, max(position)+10)
ax2.set_title("Position vs. Time")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Position (m)")
trace_line, = ax2.plot([], [], 'b-', lw=1.5) # The path drawn so far

def init():
    cord_line.set_data([], [])
    jumper_dot.set_data([], [])
    trace_line.set_data([], [])
    return cord_line, jumper_dot, trace_line

def update(frame):
    # Current x and t
    curr_x = position[frame]
    curr_t = t_data[frame]
    
    # Update Jumper (Left Plot)
    jumper_dot.set_data([0], [curr_x])
    cord_line.set_data([0, 0], [0, curr_x]) # Line from bridge (0) to jumper
    
    # Change cord color if stretched (visual feedback)
    if curr_x > 30: # L = 30
        cord_line.set_color('red') # Tension!
    else:
        cord_line.set_color('black') # Slack
        
    # Update Trace (Right Plot)
    trace_line.set_data(t_data[:frame], position[:frame])
    
    return cord_line, jumper_dot, trace_line

# Create Animation
# interval=30 means 30ms per frame
ani = FuncAnimation(fig, update, frames=len(t_data), init_func=init, blit=True, interval=30)

print("Simulation Complete. Showing Animation...")
plt.tight_layout()
plt.show()
