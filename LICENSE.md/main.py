"""
1) DEFINING PARAMETERS
In order to read results select the variable / functionality you want as
True, and then select the others as false.
"""
# Animation
Animation = True

# Generally conserved quantities
Kinetic_Energy = True
Momentum = False
Pressure = False

# Plots a histogram
Boltzmann_Distribution = False

# All the state diagrams
#PT_Plot = False

# Extra functionality
Repulsive_Forces = False
Gravity = False
Inelastic_Collisions = False

# Plotting styles
Individual_balls = False

# Defining Parameters
particle_number = 100
particle_radius = 0.01# 188E-12m radius of argon
particle_mass = 0.1 # 6.63E-26kg mass of argon molecule
time_simulation = 5.
container_width = 4. # approximately 100 times bigger than radius of particle
temperature = 298
boltzmann_constant = 1.38 # 1.38E-23

###########################################################

"""
2) IMPORTING FUNCTIONS
"""
import numpy as np
# a very useful function which calculates the distance between two points
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import matplotlib.animation as animation

###########################################################

"""
3) MAKING THE CLASS FOR PARTICLES IN THE BOX

All particles and their motions are defined by this class

init_state is an [N x 4] array, where N is the number of particles:
    [[x1, y1, vx1, vy1],
     [x2, y2, vx2, vy2],
     ...               ]

Bounds is the size of the box: [xmin, xmax, ymin, ymax]
"""
# define the container half width in order to define the limits
cr = container_width / 2

class ParticleBox:
    
    def __init__(self, init_state = [[1, 1, 1, 1],
                                     [1, 1, 1, 1]],
                 # define the edge of the box
                 limits = [-cr, cr, -cr, cr],
                 radius = particle_radius,
                 mass = particle_mass,
                 G = 9.8):
        # as array converts any input into an array
        self.init_state = np.asarray(init_state, dtype=float)
        # make a new array for the masses filled with ones (all the masses are the same)
        self.mass = mass * np.ones(self.init_state.shape[0])
        # the size is given
        self.radius = radius
        self.state = self.init_state.copy()
        self.time = 0
        # the maximum range in which the balls are allowed
        self.limits = limits
        # only necessary if we include gravity in the simulation
        self.G = G

    def move(self, dt):
        # make a single step by time dt
        self.time += dt
        
        # update positions
        # make the new position a product of the time step and the velocity
        # the velocity is in the last two columns of the array and does not change
        # the position is in the first two columns of the array and does change
        self.state[:, :2] += dt * self.state[:, 2:]

        # find pairs of particles undergoing a collision
        # pdist finds pairwise distances
        # returns a n x n matrix for n particles and 
        D = squareform(pdist(self.state[:, :2]))
        # D < 3 has been founc through trial and error to be best
        const = 3.5
        if Repulsive_Forces == True:
            const = 15
        ind1, ind2 = np.where(D < const * self.radius)
        unique = (ind1 < ind2)
        ind1 = ind1[unique]
        ind2 = ind2[unique]

        # update velocities of colliding pairs
        # zip function returns a list of tuples
        for b1, b2 in zip(ind1, ind2):
            # mass
            m1 = self.mass[b1]
            m2 = self.mass[b2]

            # location vector
            # note that position vector 2 is not included in the list
            r1 = self.state[b1, :2]
            r2 = self.state[b2, :2]

            # velocity vector
            # here position element 2 is included
            v1 = self.state[b1, 2:]
            v2 = self.state[b2, 2:]

            # relative postion and velocity vectors
            r_rel = r1 - r2
            v_rel = v1 - v2

            # momentum vector of the center of mass
            v_cm = (m1 * v1 + m2 * v2) / (m1 + m2)

            # collisions of spheres reflect v_rel over r_rel
            rr_rel = np.dot(r_rel, r_rel)
            vr_rel = np.dot(v_rel, r_rel)
            
            elasticity = 1.00
            if Inelastic_Collisions == True:
                elasticity = 0.995
            v_rel = (2 * r_rel * vr_rel / rr_rel - v_rel) * elasticity

            # assign new velocities
            self.state[b1, 2:] = v_cm + v_rel * m2 / (m1 + m2)
            self.state[b2, 2:] = v_cm - v_rel * m1 / (m1 + m2) 

        # check for crossing boundary
        # crossed_ will give a boolean value which tells you whether the particle has crossed
        # compare the position vector + ball radius with the boundary of the container
        cut_x1 = (self.state[:, 0] < self.limits[0] + self.radius)
        cut_x2 = (self.state[:, 0] > self.limits[1] - self.radius)
        cut_y1 = (self.state[:, 1] < self.limits[2] + self.radius)
        cut_y2 = (self.state[:, 1] > self.limits[3] - self.radius)
        
        #cut_x/y are boolean arrays of dimension (number of particles, 1)
        self.state[cut_x1, 0] = self.limits[0] + self.radius
        self.state[cut_x2, 0] = self.limits[1] - self.radius
        self.state[cut_y1, 1] = self.limits[2] + self.radius
        self.state[cut_y2, 1] = self.limits[3] - self.radius

        #
        velx = np.array([self.state[cut_x1 | cut_x2, 2]])
        vely = np.array([self.state[cut_y1 | cut_y2, 3]])
        
        # The vertical bar | (as opposed to / in italics) is a bitwise operation
        # Worth noting that the first line is equivalent to:
        # self.state[cut_x1, 2] *= -1
        # self.state[cut_x2, 2] *= -1
        self.state[cut_x1 | cut_x2, 2] *= -1
        self.state[cut_y1 | cut_y2, 3] *= -1
        
        
        # add gravity
        if Gravity == True:
            self.state[:, 2] -= self.mass * self.G * dt
        
        return velx, vely

    def Kinetic(self, dt):
        
        # It is very important that the move function is called otherwise
        # the KE returned will be for t = 0
        ParticleBox.move(self, dt)
        
        v_squared = (self.state[:, 2] ** 2) + (self.state[:, 3] ** 2)
        KE = 0.5 * self.mass * v_squared
        
        return KE

    def Momentum(self, dt):
        
        ParticleBox.move(self, dt)
        
        v = (self.state[:, 2]**2 + self.state[:, 3]**2)**0.5
        mom = self.mass * v
        
        return mom
    
        
        
    def Volume(self, dt, new_limits=[-cr, cr, -cr, cr]):
        
        self.limits = new_limits
        ParticleBox.move(dt)
        return
        
    def Pressure(self, dt):
        
        velx, vely = ParticleBox.move(self, dt)
        
        # note lowercase is momentum, uppercase P is pressure
        px = np.absolute(2**velx) * np.mean(self.mass)
        py = np.absolute(2**vely) * np.mean(self.mass)
        
        px_Total = np.nansum(px)
        py_Total = np.nansum(py)
        
        p_Total = px_Total + py_Total
        
        P = p_Total / (dt * 4 * container_width)
        
        return P
        
        
            

#################################################
"""
4) CREATING ALL THE BALLS AND DEFINING VELOCITIES AND POSITION
"""
# set up initial state
# this line seeds the generator, the number must vary between 0 and 2**32 - 1
# otherwise the number does not affect the calculation
np.random.seed(1)
# numbers from the random function vary between [0.0, 1.0] so with the correction
# [-0.5, 0.5] produced by the random number generator
# double brackets are needed to support the fact that random_sample() takes at most one positional argument
init_state = -0.5 + np.random.random((particle_number, 4))
# the size of the box is 2 x 2 so the random numbers must be multiplied
# by 4 to produce the numbers in the box
init_state[:, :2] *= (cr/0.5)
# multiply the velocities by a factor
init_state[:, 2:] *= 1

gas = ParticleBox(init_state, radius = particle_radius)

dt = 1. / 30 # 30fps
    

#################################################
"""
5) DEFINING THERMODYNAMIC QUANTITIES
"""
# Most of the graphs are plotted against time so make this a global variable
time = np.arange(0., time_simulation, dt)

# Using the Kinetic Energy output from the class
def Kinetic_Array():
    
    # initially defined at t = 0
    KE_all = gas.Kinetic(0)
    
    # loop over the entire time interval to append each time slice to the array
    # end up with an array containing the KE information for every particle at every time
    # (time_simulation - dt) is to ensure that the dimensions of time and KE_all are the same
    for t in np.arange(0., (time_simulation - dt), dt):
        KE2 = gas.Kinetic(t)
        KE_all = np.vstack((KE_all, KE2))
        
    # KE_all has shape (frames, number of particles) e.g. (150, 50)
    # for 5 seconds of 30 fps and 50 particles
    KE_total = np.nansum(KE_all, axis=1)

    return KE_total, KE_all
    
# The layout of this function is identical to Kinetic_Array with different variable names
def Momentum_Array():
    
    M_all = gas.Momentum(0)
    
    for t in np.arange(0., (time_simulation - dt), dt):
        M2 = gas.Momentum(t)
        M_all = np.vstack((M_all, M2))
    
    M_total = np.nansum(M_all, axis=1)
    
    return M_total, M_all

def Pressure_Array():
    
    P_total = gas.Pressure(0)
    
    for t in np.arange(0., (time_simulation - dt), dt):
        P2 = gas.Pressure(t)
        P_total = np.vstack((P_total, P2))
     
    return P_total
    

def Temperature():
    
    KE_total, KE_all = Kinetic_Array()
    
    Temp_K = np.nanmean(KE_all) / boltzmann_constant
    return Temp_K
    

#################################################
"""
6) DEFINING PLOTTING FUNCTIONS
"""

def Plotting():
    
    # Import the variables from the Kinetic Energy and Momentum function
    KE_total, KE_all = Kinetic_Array()
    M_total, M_all = Momentum_Array()
    
    if Kinetic_Energy == True:
        # Define the plotting parameters
        fig, ax = plt.subplots()
        
        # The functionality for the sum of the KE to be plotted or the indivdual balls
        # Both graphs should be a straight line
        if Individual_balls == True:
            for n in range(particle_number):
                ax.plot(time, KE_all[:, n])
        else:
            plt.plot(time, KE_total)
        
        # Creating the legend and labelling the axis
        plt.xlabel('Time (s)', fontsize=30)
        plt.ylabel('Kinetic Energy (J)', fontsize=30)
        plt.title('Kinetic Energy vs Time', fontsize=40)
        x = time_simulation / 2.
        y = KE_total[1]

    elif Momentum == True:
        M_total = M_total
        if Individual_balls == True:
            for n in range(particle_number):
                plt.plot(time, M_all[:, n])
        else:
            plt.plot(time, M_total)
        
        plt.xlabel('Time (s)', fontsize=30)
        plt.ylabel('Momentum (kgm/s)', fontsize=30)
        plt.title('Momentum vs Time', fontsize=40)
        x = time_simulation * 0.8
        #y = M_total[1]
        y = 15.1

        
    elif Pressure == True:
        # Defining pressure as P = (N * (internal energy)) / volume
        P_total = Pressure_Array()
        plt.plot(time, P_total)
        
        plt.xlabel('Time (s)', fontsize=30)
        plt.ylabel('Pressure (Pa)', fontsize=30)
        plt.title('Pressure vs Time', fontsize=40)
        x = time_simulation * 0.8
        y = P_total[1]
    
    
    plt.text(x, y, 'number of particles = %(number)d\n'
             'particle radius = %(radius).2E\n'
             'particle mass = %(mass).2E\n'
            'time for simulation = %(time).2f\n'
            'container width = %(width).2E' \
            % {"number":particle_number, "radius":particle_radius, \
               "mass":particle_mass, "time":time_simulation, \
               "width":container_width}, fontsize=15)
    
    plt.grid(True)
    plt.show()

    
    
def Boltzmann_Plot():
    
    KE_total, KE_all = Kinetic_Array()
    
    # this is the average value of energy for each particle
    KE_mean = np.nanmean(KE_all, axis=0)
    
    # Plotting the histogram of particle velocity
    # density = True normalises the histogram
    plt.title('Boltzmann Distribution', fontsize = 40)
    plt.xlabel('Mean speed (m/s)', fontsize = 30)
    plt.ylabel('Frequency', fontsize = 30)
    x = KE_mean.max()
    y = 50
    plt.text(x, y, 'number of particles = %(number)d\n'
             'particle radius = %(radius).2f\n'
             'particle mass = %(mass).2f\n'
            'time for simulation = %(time).2f\n'
            'container width = %(width).2f' \
            % {"number":particle_number, "radius":particle_radius, \
               "mass":particle_mass, "time":time_simulation, \
               "width":container_width})
    plt.hist(KE_mean, bins=100)
    #plt.
    plt.show()
    


#################################################
"""
7) PLOTTING THERMODYNAMIC QUANTITIES
"""
    
if Kinetic_Energy == True or Momentum == True or Pressure == True:
    Plotting()

if Boltzmann_Distribution == True:
    Boltzmann_Plot()
    


#################################################
"""
8) RUNNING THE ANIMATION
""" 

if Animation == True:
    
    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(-cr, cr), ylim=(-cr, cr))
    
    # particles holds the locations of the particles
    # 'bo' = blue circles
    # 'ys' = yellow squares
    # 'rd' = red diamonds
    # 'mh' = majenta hexagons
    # 'mp' = majenta pentagons
    # 'r' = red lines between balls
    particles, = ax.plot([], [], 'bo', ms=6)
    
    # rect is the box edge
    box = plt.Rectangle(gas.limits[::2],
                         gas.limits[1] - gas.limits[0],
                         gas.limits[3] - gas.limits[2],
                         ec='none', lw=1, fc='none')
    ax.add_patch(box)
    
    def init():
        #initialize animation
        particles.set_data([], [])
        #box.set_edgecolor('k')
        return particles, box
    
    def animate(i):
        #perform animation step
        gas.move(dt)
    
        # calculating the particle size based on the radius and the width of the figure
        size = int(fig.dpi * 2 * gas.radius * fig.get_figwidth()
                / np.diff(ax.get_xbound())[0])
        
        # update pieces of the animation
        box.set_edgecolor('k')
        # this defines what coordinates the particles are plotted on
        # it is 1 and then 0 as it 
        particles.set_data(gas.state[:, 1], gas.state[:, 0])
        particles.set_markersize(size)
        return particles, box
    
    # number of frames does not make a difference
    # interval changes how the windox size affects the speed of the particles
    ani = animation.FuncAnimation(fig, animate, frames=600,
                                  interval=10, blit=True, init_func=init)
    
    plt.show()
