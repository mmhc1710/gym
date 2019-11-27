import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.io import loadmat
import pandas as pd

# plt.style.use('fivethirtyeight')

class CarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.m = 5.
        self.L = 1.
        self.c = 0.
        self.force_mag = 10.0
        self.steer_mag= np.pi/6.
        self.dt = 0.05  # seconds between state updates
        self.timesteps = 1000
        self.kinematics_integrator = 'ode4'

        low = np.array([
            -np.finfo(np.float32).max,
            -np.pi/2.,
            -np.finfo(np.float32).max,
            -np.finfo(np.float32).max
            # -np.pi/6.
        ])

        high = np.array([
            np.finfo(np.float32).max,
            np.pi/2.,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max
            # np.pi/6.
            ])

        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Box(np.array([-1., -1.]), np.array([+1., +1.]), dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None
        self.isLearning = True
        self.state_dot = np.zeros(self.observation_space.shape[0])
        self.action_pre = np.zeros(self.action_space.shape[0])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        self.t += self.dt
        self.n += 1
        done = False
        costs = 0.0

        state = self.state
        action_pre = self.action_pre
        # if not len(args[0])==0: action_pre = args[0][0]
        # Vb, x, y, theta = state
        # x, y = self.positions[self.n-1, 1], self.positions[self.n-1, 2]
        # xTmp, yTmp = x, y
        # Vb, theta = state

        if action[0]==0. or np.sign(action[0])!=np.sign(action_pre[0]):
            action_pre[0] = 0.
        action[0] = action_pre[0] + self.dt*action[0]/4.
        action[0] = np.clip(action[0], -1., 1.)

        u = action*[self.force_mag, self.steer_mag]
        state_, state_dot = self.ode4(state, self.dt, u)
        # assert self.observation_space.contains(state_), "%r (%s) invalid" % (state_, type(state_))
        # if not self.observation_space.contains(state_):
        #     costs += 1000.
        #     done = True

        # if self.kinematics_integrator == 'euler':
        #     Vb += self.dt*Vb_dot
        #     # x += self.dt * x_dot
        #     # y += self.dt * y_dot
        #     theta += self.dt * theta_dot
        #     x_dot = Vb*np.cos(theta)
        #     y_dot = Vb*np.sin(theta)
        #     x += self.dt * x_dot
        #     y += self.dt * y_dot
        #     # theta = angle_normalize(theta)
        # else: # semi-implicit euler
        #     x_dot = x_dot + self.dt * xacc
        #     x  = x + self.dt * x_dot
        #     theta_dot = theta_dot + self.dt * thetaacc
        #     theta = theta + self.dt * theta_dot

        # self.state = state_
        # position = (x ,y)
        # self.state = (Vb, x, y, theta)
        if (self.n >= self.timesteps):# or (np.array_equal([Vb ,theta], self.goal)):
            if self.isLearning:
                done = True
        # done = np.array_equal([x, y], self.goal)
        # done = bool(done)

        Vb = state_[0]
        psi = state_[1] = angle_normalize(state_[1])
        Vbdot = state_dot[0]
        psi_dot = state_dot[1]
        # Vb_g = state_[2]
        # psi_g = state_[3] = angle_normalize(state_[3])

        # psi = psi_rad = angle_normalize(psi)
        # psi = np.rad2deg(psi)

        throttle = action[0]
        steer = action[1]
        throttle_pre = action_pre[0]
        steer_pre = action_pre[1]

        costs+=(Vb*np.sin(psi))**2
        # costs += 0.0001*(np.rad2deg(psi)) ** 2
        # costs += 0.001*Vb**2
        # costs += Vbdot**2
        # costs += psi_dot**2
        # costs += 0.1 * ((throttle-throttle_pre)/self.dt)**2
        # costs += 0.0001 * ((steer-steer_pre)/self.dt)**2
        # costs += 0.01 * throttle**2
        # costs += 1. * steer**2# + .1 * x_dot**2 + 1*self.out_of_bound(x) # + x**2 + .001 * (force ** 2)
        # costs = (x) ** 2 + (y) ** 2 + .0001 * (Vb) ** 2
        # costs = angle_normalize(theta - theta_ref[_r]) ** 2 + .1 * (Vb - Vb_ref[_r]) ** 2  # + .1 * x_dot**2 + 1*self.out_of_bound(x) # + x**2 + .001 * (force ** 2)

        # if ((theta>0.) and (theta_dot>0.) and (force<0.)) or (theta<0.) and (theta_dot<0.) and (force>0.):
        #         costs+=1.

        # costs = 1.
        # if np.abs(x)<1. and np.abs(y)<1.:#(np.array_equal([x, y], self.goal)):
        #     costs-=100
        #     done = True

        if (np.abs(Vb) < 1.) and (np.abs(psi) < np.deg2rad(2)):  # np.abs(Vb)<1. and  (np.array_equal([x, y], self.goal)):
            costs -= 2000
            if self.isLearning:
                done = True

        if np.abs(Vb) > 50.:# and np.abs(psi) < 0.1:  # np.abs(Vb)<1. and  (np.array_equal([x, y], self.goal)):
            costs += 1000
            if self.isLearning:
                done = True

        if np.abs(psi) > np.deg2rad(180):# and np.abs(psi) < 0.1:  # np.abs(Vb)<1. and  (np.array_equal([x, y], self.goal)):
            costs += 1000
            if self.isLearning:
                done = True

        if not done:
            reward = -costs  # 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = -costs  # 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = -1000000.

        # self.states = np.append(self.states, [np.append([self.t],np.array(self.state))], axis=0)
        # self,states = np.vstack([self.states, np.array(self.state)])

        # self.positions = np.append(self.positions, [np.append([self.t], np.array(position))], axis=0)

        # self.actions = np.append(self.actions, [np.append([self.t], np.array(action))], axis=0)

        self.state = state_
        self.action_pre = action

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.t = 0.
        self.n = 0
        self.state = np.asarray(self.state)
        self.state = self.observation_space.sample()
        # self.state[0] = -2.0
        # self.state[1] = -0.01
        self.state[0] = self.np_random.uniform(low=-10., high=10., size=(1,))
        self.state[1] = self.np_random.uniform(low=-np.pi/6, high=np.pi/6, size=(1,))
        self.state[2] = self.state[0]*np.cos(self.state[1])
        self.state[3] = self.state[0] * np.sin(self.state[1])
        # self.state[5] = self.np_random.uniform(low=-np.pi, high=np.pi, size=(1,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        # if self.toggle_plot:
        #     states = self.states
        #     t, Vb, theta = states[:, 0], states[:, 1], states[:, 2]
        #     # t, Vb, x, y, theta = states[:, 0], states[:, 1], states[:, 2], states[:, 3], states[:, 4]
        #     # Vb, x, y, theta = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
        #     self.ax[0].clear()
        #     # self.ax.plot(x, y)
        #     self.ax[0].plot(t, -Vb)
        #     if not self.isLearning: self.ax[0].plot(t_ref, Vb_ref)
        #     # self.ax[0].set_xlim([self.T,self.T])
        #     self.ax[0].set_ylim([-self.force_mag,self.force_mag])
        #
        #     self.ax[1].clear()
        #     self.ax[1].plot(t, angle_normalize(-theta))
        #     if not self.isLearning: self.ax[1].plot(t_ref, angle_normalize(theta_ref))
        #     # self.ax[1].set_xlim([self.T,self.T])
        #     self.ax[1].set_ylim([-np.pi/2,np.pi/2])
        #
        #     # self.ax[2].clear()
        #     # # self.ax[2].plot(x, y)
        #     # # self.ax[2].arrow(x[self.n],y[self.n],Vb[self.n]*np.cos(theta[self.n])/10.,Vb[self.n]*np.sin(theta[self.n])/10.)
        #     # self.ax[2].set_xlim([-self.force_mag,self.force_mag])
        #     # self.ax[2].set_ylim([-self.force_mag,self.force_mag])
        #
        #     actions = self.actions
        #     f, s = actions[:, 1], actions[:, 2]
        #     self.ax[2].clear()
        #     self.ax[2].plot(t, f, t, s)
        #     # self.ax[2].arrow(x[self.n],y[self.n],Vb[self.n]*np.cos(theta[self.n])/10.,Vb[self.n]*np.sin(theta[self.n])/10.)
        #     # self.ax[2].set_xlim([-self.force_mag, self.force_mag])
        #     # self.ax[2].set_ylim([-self.force_mag, self.force_mag])
        #
        #     x, y = self.positions[:, 1], self.positions[:, 2]
        #     self.ax[3].clear()
        #     self.ax[3].plot(x, y)
        #     if not self.isLearning: self.ax[3].plot(x_ref, y_ref)
        #     # self.ax[2].arrow(x[self.n],y[self.n],Vb[self.n]*np.cos(theta[self.n])/10.,Vb[self.n]*np.sin(theta[self.n])/10.)
        #     self.ax[3].set_xlim([-self.force_mag, self.force_mag])
        #     self.ax[3].set_ylim([-self.force_mag, self.force_mag])
        #
        #     plt.show(block=False)
        #     plt.tight_layout()
        #     plt.pause(0.01)

        return self.viewer

    def close(self):
        # if self.toggle_plot:
        #     plt.close()
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def out_of_bound(self, x):
        if (x < -self.x_threshold) or (x > self.x_threshold):
            return 1.0
        else:
            return -1.0

    def ode4(self, X_0, dt, control):
        dt2 = dt / 2
        dt6 = dt / 6

        dxdt = self.dynamics(X_0, control)  # , alphadot, control)
        x0t = X_0 + dt2 * dxdt

        dx0t = self.dynamics(x0t, control)  # , alphadot, control)
        x0t = X_0 + dt2 * dx0t

        dxmold = self.dynamics(x0t, control)  # , alphadot, control)
        x0t = X_0 + dt * dxmold

        dxm = dx0t + dxmold

        dx0t = self.dynamics(x0t, control)  # , alphadot, control)
        statesdot = dxdt + dx0t + 2 * dxm
        X = X_0 + dt6 * statesdot

        return X, (dt6 * statesdot)/self.dt

    def dynamics(self, x0, u):
        Vbdot = (u[0] - np.sign(x0[0])*self.c*x0[0]**2)/self.m
        Vx = x0[0] * np.cos(x0[1])
        Vy = x0[0] * np.sin(x0[1])
        # thetadot = x0[0]*np.tan(u[1])/self.L
        # thetadot = np.sign(x0[0])*x0[0]*(u[1])/self.L
        thetadot = x0[0]*np.tan((u[1]))/self.L
        # phidot = thetadot
        xdot = np.array([Vbdot, thetadot, Vx, Vy])
        return xdot


def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)

def nearest_point(A,p):
    d = np.array([(p[0]-A[1,:])**2, (p[1]-A[2,:])**2])
    d = np.sum(d, axis=0)
    min_idx = d.argmin()
    min_pt = [A[1,min_idx], A[2,min_idx]]
    s = np.sign(np.arctan2(min_pt[1]-p[1],min_pt[0]-p[0]))
    return min_idx, min_pt, s*np.sqrt(d[min_idx]), d

def next_target_point(A,p,v,l,dt):
    t = 1e-5
    Vb = 1e-5
    theta = 1e-5
    if v:
        t = l/v
    min_idx, min_pt, d, dist = nearest_point(A,p)
    # next_idx = np.int((t_ref[min_idx]+t)/dt)
    # next_pt = [A[1,next_idx],A[2,next_idx]]
#     l = np.sqrt((min_pt[0]-next_pt[0])**2 + (min_pt[1]-next_pt[1])**2)
    #     l = v*t
    h = np.sqrt(d**2 + l**2)
    if v:
        Vb = h/t
    theta = np.arctan2(d,l)
    return Vb, theta