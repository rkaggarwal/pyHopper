from numpy import *;
from matplotlib import pyplot as plt;
from matplotlib import animation;
import matplotlib;
import control;
#matplotlib.use('Agg');
import slycot;

# Set up formatting for the movie files
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

# ===== INFORMATION =====
# 2D only for now (3DOF)


# ===== MODEL SETUP =====

# Physical Parameters
m = 10; # kg, vehicle mass
s = .30; # m, vehicle side length (cubic)
vol = s**3; # m^3, total vehicle volume
g = 9.81; # m/s^2, gravitational acceleration
F_max = 2 * m * g; # max accel is +/- 1g
d = s / 2; # distance from thrust origin to COM
I = 1 / 6* m * s**2; # kg-m^2, in-plane rotational inertia


# ===== LINEARIZED STATE-SPACE FORMULATION =====

# linearized approximation (upright hovering equilibrium)
# we have an augmented state var, with g.  This is an uncontrollable
# variable, so we exclude it from the controller design

A = array([[0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 1, 0],
           [0, 0, -g, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]]);

B = array([[0, 0],
          [0, 0],
          [0, 0],
          [0, -g],
          [1, 0],
          [0, -m*g*d/I],
          [0, 0]]);

C = eye(7);
D = zeros((7, 2));

controllability_rank = linalg.matrix_rank(control.ctrb(A, B)); # should be 6!
print(controllability_rank)

# ===== CONTROLLER DESIGN =====

# now let's wrap a simple controller around this to move from one point to another.

Q = array([[1, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0],
           [0, 0, 10000, 0, 0, 0],
           [0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 1]]);

R = array([[100, 0],
           [0, 1000000]]);


K, S, E = control.lqr(A[0:6, 0:6], B[0:6, :], Q, R);

# u = -K*(x_current - x_desired)

# now K is our gain matrix
# we have to append a 0 onto the end to account for our additional DC augmented var (gravity)

toAdd = array([[0], [0]]);


K = hstack((K, toAdd));


test = 1;


# ===== SIMULATION CONFIGURATION =====
# point to point movement

sim_time = 20; # seconds
dt = .010; # seconds
f = 1/dt; # Hz
sim_length = int(sim_time/dt);
time_vector = linspace(0, sim_time, int(sim_time/dt));

# Vehicle state denoted by x, y, theta, xdot, ydot, thetadot (m, rad)
# Vehicle control denoted by throttle (0-1), TVC angle, rad (assume instantaneous TVC tracking)

state_array = zeros((sim_length, 7));
control_array = zeros((sim_length, 2));
traj_array = zeros((sim_length, 7));

state_array[0, :] = array([0, 0, 0, 0, 0, 0, 0]);
control_array[0, :] = array([1, 0]);



# triangle-wave control thrust value
control_array[0:int(sim_length/4), 0] = F_max-m*g;
control_array[int(sim_length/4):2*int(sim_length/4), 0] = 0 - m*g;
control_array[2*int(sim_length/4):sim_length, 0] = m*g - m*g;

# TVC angle actuation
#control_array[:, 1] = deg2rad(random.randn()*.05); # rad
control_array[:, 1] = 0;


x1 = 0;
y1 = 0;
x2 = 100;
y2 = 100;

z_initial = array([[x1],
                   [y1],
                   [0],
                   [0],
                   [0],
                   [0],
                   [g]]);

z_final = array([[x2],
                 [y2],
                 [0],
                 [0],
                 [0],
                 [0],
                 [g]]);


state_array[0, :] = z_initial.T;

test = 1;

# ===== SIMULATION! =====
# linear and non-linear state evolution (simulation)

for i in range(1, sim_length):

    # hard-coded control efforts
    Ft = control_array[i-1, 0];
    phi = control_array[i-1, 1];

    state_prev = array(state_array[i-1, :]);
    state_prev = state_prev[newaxis, :];
    #theta = state_array[i-1, 2];

    # LQR control efforts
    if i < sim_length/2:
        traj_ref = 2*i/sim_length*(z_final - z_initial) + z_initial;
    else:
        traj_ref = z_final;


    u = -K @ (state_prev - z_final.T).T; # LQR finds 'optimal path' for point-to-point
    #u = -K @ (state_prev.T - traj_ref); # we impose the trajectory here, could be from a non-lin optimizer for ex.



    # Non-linear simulation
    # dz = array([ state_array[i-1, 3],
    #              state_array[i-1, 4],
    #              state_array[i-1, 5],
    #              (-Ft*sin(theta + phi))/m,
    #              (Ft*cos(theta + phi) - m*g)/m,
    #              -Ft*sin(phi)*d/I
    #             ]);
    #
    # deltaz = dz*dt;

    state_prev_t = state_prev.T;
    # Linear simulation, only valid about upright hovering equilbria (arbitrary (x,y) allowed).
    #dz = A @ state_array[i-1, :] + B @ control_array[i-1, :]; # manual control effort
    dz = A @ (state_prev_t) + B @ u; # LQR control effort
    deltaz = dz*dt;



    state_array[i, :] = state_prev + deltaz.T;
    traj_array[i, :] = traj_ref.T;
    control_array[i, :] = u.T;

    # for now the control array is OL

    # if state_array[i, 1] < 0:
    #     state_array[i, 1] = 0;


test = 2;


# ===== DIAGNOSTIC PLOTS =====

# Plot control effort
fig0 = plt.figure();

color = 'tab:blue'
ax0 = plt.axes();
ax0.plot(time_vector, (control_array[:, 0])/(m*g)+.5, color = color)
ax0.set_ylabel("Normalized Thrust (0-1)")
ax0.set_xlabel("Time [s]");
ax0.legend(["Thrust"], loc=2)
ax0.set_title("Control Effort");

color = 'tab:red'
ax1 = ax0.twinx();
ax1.plot(time_vector, rad2deg(control_array[:, 1]), color = color);
ax1.set_ylabel("Thrust Vector Angle [deg]");

ax1.legend(["TVC Angle"], loc=1)

# Plot errors
fig1 = plt.figure();

color = 'tab:blue'
ax0 = plt.axes();
ax0.plot(time_vector, (traj_array[:, 0] - state_array[:, 0]), color = color)
ax0.set_ylabel("X-Error [m]")
ax0.set_xlabel("Time [s]");
ax0.legend(["X"], loc=2)
ax0.set_title("Trajectory Errors");

color = 'tab:red'
ax1 = ax0.twinx();
ax1.plot(time_vector, (traj_array[:, 1] - state_array[:, 1]), color = color);
ax1.set_ylabel("Y-Error [m]");

ax1.legend(["Y"], loc=1)






# ===== TRAJECTORY ANIMATION =====

fig = plt.figure()

fig.set_dpi(100)
#fig.set_size_inches(6, 6)

ax = plt.axes(xlim=(-100, 100), ylim=(0, 100))
plt.axis('equal')
#patch = plt.Circle((5, -5), 0.75, fc='y')

patch_width = 5;
line_length = 5;
vehicle = plt.Rectangle((0, 0), patch_width, patch_width, 0.0, color='blue') # vehicle
thrust = plt.Line2D((0, 0), (0, -10), lw=3, color = 'red'); # thrust vector

def init():
    ax.clear();
    ax.add_patch(vehicle)
    ax.add_line(thrust)
    ax.plot(state_array[:, 0], state_array[:, 1], lw = .5, color='black');
    ax.plot(traj_array[:, 0], traj_array[:, 1], lw = .5, color = 'green');
    ax.set_xlabel("Range Position [m]");
    ax.set_ylabel("Altitude [m]");
    ax.set_title("Point-to-Point Tracking");

    return []

def animationController(i, vehicle, thrust):

    animatePatch(i, vehicle);
    animateLine(i, thrust);
    return [];



def animatePatch(i, patch):
    x = state_array[i, 0];
    y = state_array[i, 1];
    th = state_array[i, 2];


    patch.set_xy((x - patch_width/2*sqrt(2)*cos(pi/4+th), y - patch_width/2*sqrt(2)*sin(pi/4+th)));
    patch.angle = rad2deg(th);

    return patch,

def animateLine(i, line):
    throttle = control_array[i, 0] + m*g;
    phi = control_array[i, 1]; # 20x just to exaggerate thrust vector

    x = state_array[i, 0];
    y = state_array[i, 1];
    th = state_array[i, 2];

    line_length = throttle/(m*g) * 5;

    line.set_xdata([x + patch_width/2*sin(th), x + patch_width/2*sin(th) + line_length*sin(th+phi)]);
    line.set_ydata([y - patch_width/2*cos(th), y - patch_width/2*cos(th) - line_length*cos(th+phi)]);

    return line,

anim = animation.FuncAnimation(fig, animationController,
                               init_func=init,
                               frames=sim_length,
                               fargs=(vehicle, thrust,),
                               interval=2,
                               blit=True,
                               repeat=True)

plt.show()

# print(matplotlib.matplotlib_fname());
# print("saving...");
# anim.save('hover.mp4', writer=writer)
# #anim.save('~/PycharmProjects/2d_propulsive_control/animation.g`if`', writer='imagemagick', fps=30);
# print("saved!")

#print(state_array)
print("var explorer debug");