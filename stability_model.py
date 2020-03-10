"""
Simple model of standing postural stability, consisting of foot and body segments,
and two muscles that create moments about the ankles, tibialis anterior and soleus.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from musculoskeletal import HillTypeMuscle, get_velocity, force_length_tendon
import math


def soleus_length(theta):
    """
    :param theta: body angle (up from prone horizontal)
    :return: soleus length
    """
    rotation = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    origin = np.dot(rotation, [.3, .03])
    insertion = [-.05, -.02]
    difference = origin - insertion
    return np.sqrt(difference[0]**2 + difference[1]**2)


def tibialis_length(theta):
    """
    :param theta: body angle (up from prone horizontal)
    :return: tibialis anterior length
    """
    rotation = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    origin = np.dot(rotation, [.3, -.03])
    insertion = [.06, -.03]
    difference = origin - insertion
    return np.sqrt(difference[0]**2 + difference[1]**2)


def gravity_moment(theta):
    """
    :param theta: angle of body segment (up from prone)
    :return: moment about ankle due to force of gravity on body
    """
    mass = 75 # body mass (kg; excluding feet)
    centre_of_mass_distance = 1 # distance from ankle to body segment centre of mass (m)
    g = 9.81 # acceleration of gravity
    return mass * g * centre_of_mass_distance * np.sin(theta - np.pi / 2)


def dynamics(x, soleus, tibialis, control):
    """
    :param x: state vector (ankle angle, angular velocity, soleus normalized CE length, TA normalized CE length)
    :param soleus: soleus muscle (HillTypeModel)
    :param tibialis: tibialis anterior muscle (HillTypeModel)
    :param control: True if balance should be controlled
    :return: derivative of state vector
    """
    
    # WRITE CODE HERE TO IMPLEMENT THE MODEL
    if control:
        activSMax = 1
        activeTAMax = 1
        if x[0] < np.pi/2 and x[1] < 0: 
            aS = activSMax
            aTA = 0
        elif x[0] > np.pi/2 and x[1] > 0:
            aS = 0
            aTA = activeTAMax
        elif x[0] < np.pi/2 and x[1] > 0:
            aS = (np.pi/2-x[0])
            aTA = (np.pi/2-x[0])
        elif x[0] > np.pi/2 and x[1] < 0:
            aS = -(np.pi/2-x[0])
            aTA = -(np.pi/2-x[0])
        else:
            aS = 0
            aTA = 0

        if aS > activSMax:
            aS = activSMax
        if aS < 0:
            aS = 0
        if aTA > activeTAMax:
            aTA = activeTAMax
        if aTA < 0:
            aTA = 0
        print(x[0], "      ", x[1])
    else:
        aS = 0.05
        aTA = 0.4
    
    tS = (soleus.f0M)*force_length_tendon(soleus.norm_tendon_length(soleus_length(x[0]), x[2]))*0.05
    tTA = (tibialis.f0M)*force_length_tendon(tibialis.norm_tendon_length(soleus_length(x[0]), x[3]))*0.03
    x_der = np.zeros(4)
    x_der[0] = x[1]
    x_der[1] = (tS - tTA + gravity_moment(x[0]))/90
    x_der[2] = get_velocity(aS, x[2], soleus.norm_tendon_length(soleus_length(x[0]), x[2]))
    x_der[3] = get_velocity(aTA, x[3], tibialis.norm_tendon_length(tibialis_length(x[0]), x[3]))
    return x_der

def simulate(control, T):
    """
    Runs a simulation of the model and plots results.
    :param control: True if balance should be controlled
    :param T: total time to simulate, in seconds
    """
    rest_length_soleus = soleus_length(np.pi/2)
    rest_length_tibialis = tibialis_length(np.pi/2)

    soleus = HillTypeMuscle(16000, .6*rest_length_soleus, .4*rest_length_soleus)
    tibialis = HillTypeMuscle(2000, .6*rest_length_tibialis, .4*rest_length_tibialis)

    def f(t, x):
        return dynamics(x, soleus, tibialis, control)

    sol = solve_ivp(f, [0, T], [np.pi/2 -0.001, 0, 1, 1], rtol=1e-5, atol=1e-8)
    time = sol.t
    theta = sol.y[0,:]
    soleus_norm_length_muscle = sol.y[2,:]
    tibialis_norm_length_muscle = sol.y[3,:]

    soleus_moment_arm = .05
    tibialis_moment_arm = .03
    soleus_moment = []
    tibialis_moment = []
    for th, ls, lt in zip(theta, soleus_norm_length_muscle, tibialis_norm_length_muscle):
        soleus_moment.append(soleus_moment_arm * soleus.get_force(soleus_length(th), ls))
        tibialis_moment.append(-tibialis_moment_arm * tibialis.get_force(tibialis_length(th), lt))

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(time, sol.y[0,:])
    plt.ylabel('Body angle (rad)')
    plt.subplot(2,1,2)
    plt.plot(time, soleus_moment, 'r')
    plt.plot(time, tibialis_moment, 'g')
    plt.plot(time, gravity_moment(sol.y[0,:]), 'k')
    plt.legend(('soleus', 'tibialis', 'gravity'))
    plt.xlabel('Time (s)')
    plt.ylabel('Torques (Nm)')
    plt.tight_layout()
    plt.show()

simulate(True, 10)
