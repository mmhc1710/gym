import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.io import loadmat
import pandas as pd
ref_data = loadmat('/home/mmhc/workspace/test_car/cmd_data.mat')
ref_data = ref_data['D']
t_ref, Vb_ref, x_ref, y_ref, theta_ref = ref_data[:, 0], ref_data[:, 1], ref_data[:, 2], ref_data[:, 3], ref_data[:, 4]

# plt.style.use('fivethirtyeight')

class CarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.m = 5.0
        self.L = 1.0
        self.force_mag = 50.0
        self.steer_mag= np.pi/6.
        self.dt = 0.01  # seconds between state updates
        self.goal = np.array([0., 0.])
        self.kinematics_integrator = 'euler'

        high = np.array([
            np.finfo(np.float32).max,
            # np.finfo(np.float32).max,
            # np.finfo(np.float32).max,
            np.pi])

        self.action_space = spaces.Box( np.array([0.,-1.]), np.array([+1.,+1.]), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = np.zeros(self.observation_space.shape)
        self.toggle_plot = False
        self.states = np.array([])
        self.actions = np.array([])
        self.positions = np.array([])
        # self.ax.set_xlim([-100,100])
        # self.ax.set_ylim([-100,100])
        # self.ani = animation.FuncAnimation(self.fig, self.render, interval=1000)
        self.t = 0
        self.T = t_ref[-1]
        self.n = 0

        self.steps_beyond_done = None
        self.isLearning = True
        self.finishedStartup = False

        self.xyData = np.array([t_ref,x_ref, y_ref])
        self.vTarget = 30.
        self.lTarget = 0.8
        self.xy0 = [0., x_ref[0], y_ref[0]]
        self.statesT = np.array([])

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        self.t += self.dt
        self.n += 1
        done = False

        state = self.state
        # Vb, x, y, theta = state
        x, y = self.positions[self.n-1, 1], self.positions[self.n-1, 2]
        # xTmp, yTmp = x, y
        Vb, theta = state

        Fb = self.force_mag * float(action[0]) - np.sign(Vb)*1.*Vb**2
        S = self.steer_mag * float(action[1])

        Vb_dot = Fb/self.m
        # x_dot = Vb*np.cos(theta)
        # y_dot = Vb*np.sin(theta)
        theta_dot = Vb*np.tan(S)/self.L

        if self.kinematics_integrator == 'euler':
            Vb += self.dt*Vb_dot
            # x += self.dt * x_dot
            # y += self.dt * y_dot
            theta += self.dt * theta_dot
            x_dot = Vb*np.cos(theta)
            y_dot = Vb*np.sin(theta)
            x += self.dt * x_dot
            y += self.dt * y_dot
            # theta = angle_normalize(theta)
        # else: # semi-implicit euler
        #     x_dot = x_dot + self.dt * xacc
        #     x  = x + self.dt * x_dot
        #     theta_dot = theta_dot + self.dt * thetaacc
        #     theta = theta + self.dt * theta_dot

        if (not self.isLearning):# and self.finishedStartup:
            # _r = np.int(self.t / self.dt)
            # Vb -= Vb_ref[_r]
            # theta -= theta_ref[_r]
            VbT, thetaT = self.statesT[self.n-1][1:]
            Vb+=VbT
            theta+=thetaT
            x_dot = Vb*np.cos(theta)
            y_dot = Vb*np.sin(theta)
            x, y = self.positions[self.n-1][1:]
            x += self.dt * x_dot
            y += self.dt * y_dot
            p = [x, y]
            VbT, thetaT, min_idx, d = next_target_point(self.xyData, p, self.vTarget, self.lTarget, self.dt)
            if (min_idx>=(t_ref.shape[0]-1)):
                if (d>=1.0): done=True
            Vb -= VbT
            theta -= thetaT
            Vb = np.clip(Vb, -self.force_mag, self.force_mag)
            theta = angle_normalize(theta)
            self.statesT = np.append(self.statesT, [np.append([self.t],np.array([VbT,thetaT]))], axis=0)

        # if (not self.isLearning):# and self.finishedStartup:
        #     # Vb += Vb_ref[_r]
        #     # theta += theta_ref[_r]
        #     Vb += VbT
        #     theta += thetaT
        #     x_dot = Vb*np.cos(theta)
        #     y_dot = Vb*np.sin(theta)
        #     x = xTmp + self.dt * x_dot
        #     y = yTmp + self.dt * y_dot

        self.state = (Vb, theta)
        position = (x ,y)
        # self.state = (Vb, x, y, theta)
        if (self.t >= self.T):# or (np.array_equal([Vb ,theta], self.goal)):
            # if self.isLearning:
            done = True
        # done = np.array_equal([x, y], self.goal)
        # done = bool(done)

        costs = angle_normalize(theta)**2# + .001 * (Vb)**2 + .0001 * (action[1])**2# + .1 * x_dot**2 + 1*self.out_of_bound(x) # + x**2 + .001 * (force ** 2)
        # costs = (x) ** 2 + (y) ** 2 + .0001 * (Vb) ** 2
        # costs = angle_normalize(theta - theta_ref[_r]) ** 2 + .1 * (Vb - Vb_ref[_r]) ** 2  # + .1 * x_dot**2 + 1*self.out_of_bound(x) # + x**2 + .001 * (force ** 2)

        # if ((theta>0.) and (theta_dot>0.) and (force<0.)) or (theta<0.) and (theta_dot<0.) and (force>0.):
        #         costs+=1.

        # costs = 1.
        # if np.abs(x)<1. and np.abs(y)<1.:#(np.array_equal([x, y], self.goal)):
        #     costs-=100
        #     done = True

        if np.abs(Vb)<1. and np.abs(angle_normalize(theta))<0.1:#np.abs(Vb)<1. and  (np.array_equal([x, y], self.goal)):
            costs-=100
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
            reward = 0.0

        self.states = np.append(self.states, [np.append([self.t],np.array(self.state))], axis=0)
        # self,states = np.vstack([self.states, np.array(self.state)])

        self.positions = np.append(self.positions, [np.append([self.t], np.array(position))], axis=0)

        self.actions = np.append(self.actions, [np.append([self.t], np.array(action))], axis=0)

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.t = 0.
        self.n = 0
        self.state = np.asarray(self.state)

        if self.isLearning:
            # self.state = self.np_random.uniform(low=-15.0, high=15.0, size=(2,))
            self.state[0] = self.np_random.uniform(low=-self.force_mag/5, high=self.force_mag/5.)
            # self.state[1] = self.np_random.uniform(low=-self.force_mag/10, high=self.force_mag/10)
            # self.state[2] = self.np_random.uniform(low=-self.force_mag/10, high=self.force_mag/10)
            self.state[1] = self.np_random.uniform(low=-self.steer_mag, high=self.steer_mag/1.)

            self.state[0] = -1.
            self.state[1] = -1.
        else:
            VbT, thetaT, _, _ = next_target_point(self.xyData, self.xy0, self.vTarget, self.lTarget, self.dt)
            self.state[0] = 0.-VbT
            # self.state[1] = self.np_random.uniform(low=-self.force_mag/10, high=self.force_mag/10)
            # self.state[2] = self.np_random.uniform(low=-self.force_mag/10, high=self.force_mag/10)
            self.state[1] = 0.-thetaT
            # self.t += self.dt
            # _r = np.int(self.t / self.dt)
            # self.state = np.array([Vb_ref[_r]-5., theta_ref[_r]])
            # self.positions = np.array([x_ref[_r], y_ref[_r]])
            # self.state = np.array([-Vb_ref[_r], -x_ref[_r], -y_ref[_r], -theta_ref[_r]])


        self.steps_beyond_done = None
        self.states = np.array(np.append([self.t],np.array(self.state)))[np.newaxis, :]
        self.statesT = np.array(np.append([self.t],np.array([VbT,thetaT])))[np.newaxis, :]
        self.actions = np.array(np.append([self.t], np.zeros(self.action_space.shape)))[np.newaxis, :]
        self.positions = np.array(np.append([self.t],np.zeros((2,))))[np.newaxis, :]
        self.positions[0] = self.xy0

        if self.toggle_plot:
            plt.close('all')
            self.fig = plt.figure()

            plt.grid()
            self.ax = []
            self.ax.append(self.fig.add_subplot(3,1,1))
            self.ax.append(self.fig.add_subplot(3,1,2))
            self.ax.append(self.fig.add_subplot(3,1,3))
            # self.ax.append(self.fig.add_subplot(4,1,4))
        return np.array(self.state)

    def render(self, mode='human'):
        if self.toggle_plot:
            states = self.states
            t, Vb, theta = states[:, 0], states[:, 1], states[:, 2]
            # t, Vb, x, y, theta = states[:, 0], states[:, 1], states[:, 2], states[:, 3], states[:, 4]
            # Vb, x, y, theta = states[:, 0], states[:, 1], states[:, 2], states[:, 3]
            self.ax[2].clear()
            # # self.ax.plot(x, y)
            self.ax[2].plot(t, -Vb+self.vTarget)
            # if not self.isLearning: self.ax[0].plot(t_ref, Vb_ref)
            # self.ax[0].set_xlim([self.T,self.T])
            self.ax[2].set_ylim([-self.force_mag,self.force_mag])
            plt.grid()
            self.ax[2].grid()
            plt.grid()

            # self.ax[1].clear()
            # self.ax[1].plot(t, angle_normalize(-theta))
            # if not self.isLearning: self.ax[1].plot(t_ref, angle_normalize(theta_ref))
            # # self.ax[1].set_xlim([self.T,self.T])
            # self.ax[1].set_ylim([-np.pi/2,np.pi/2])

            # self.ax[2].clear()
            # # self.ax[2].plot(x, y)
            # # self.ax[2].arrow(x[self.n],y[self.n],Vb[self.n]*np.cos(theta[self.n])/10.,Vb[self.n]*np.sin(theta[self.n])/10.)
            # self.ax[2].set_xlim([-self.force_mag,self.force_mag])
            # self.ax[2].set_ylim([-self.force_mag,self.force_mag])

            actions = self.actions
            f, s = actions[:, 1], actions[:, 2]
            self.ax[0].clear()
            self.ax[0].plot(t, f, t, s)
            plt.grid()
            self.ax[0].grid()
            plt.grid()
            # self.ax[2].arrow(x[self.n],y[self.n],Vb[self.n]*np.cos(theta[self.n])/10.,Vb[self.n]*np.sin(theta[self.n])/10.)
            # self.ax[2].set_xlim([-self.force_mag, self.force_mag])
            # self.ax[2].set_ylim([-self.force_mag, self.force_mag])

            x, y = self.positions[:, 1], self.positions[:, 2]
            self.ax[1].clear()
            self.ax[1].plot(x, y, 'b>', markersize=1)
            if not self.isLearning:
                self.ax[1].plot(x_ref, y_ref, 'g')
                self.ax[1].plot(x_ref+(2.*np.sin(theta_ref)), y_ref-(2.*np.cos(theta_ref)), 'y--')
                self.ax[1].plot(x_ref-(2.*np.sin(theta_ref)), y_ref+(2.*np.cos(theta_ref)), 'y--')
            # self.ax[2].arrow(x[self.n],y[self.n],Vb[self.n]*np.cos(theta[self.n])/10.,Vb[self.n]*np.sin(theta[self.n])/10.)
            self.ax[1].set_xlim([-self.force_mag, self.force_mag*3])
            self.ax[1].set_ylim([-self.force_mag, self.force_mag*3])
            plt.grid()

            self.ax[1].grid()
            plt.grid()

            plt.show(block=False)
            plt.tight_layout()
            plt.pause(0.001)

        return self.viewer

    def close(self):
        if self.toggle_plot:
            plt.close()
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def out_of_bound(self, x):
        if (x < -self.x_threshold) or (x > self.x_threshold):
            return 1.0
        else:
            return -1.0


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
    return Vb, theta, min_idx, d