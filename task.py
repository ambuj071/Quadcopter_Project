import numpy as np
from physics_sim import PhysicsSim
import random

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        #self.state_size = self.action_repeat * 2
        self.state_size = 1
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = -.3*(abs(self.sim.pose[:3] - self.target_pos[:3])).sum()+50.*self.sim.v[2]
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        pos_prev=self.sim.pose[0:6]
        vel_prev=self.sim.v[0:3]
        ang_vel_prev=self.sim.angular_v[0:3]
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
        pose_all=self.sim.pose[0:6] 
        reward+=max(-0.2,min(0.1,(self.sim.v[2]-vel_prev[2])/100))
        reward+=max(-0.2,min(self.sim.v[2]/100,0.5))
        reward-=min(0.3,(abs(self.sim.angular_v[0]-ang_vel_prev[0])+abs(self.sim.angular_v[1]-ang_vel_prev[1])+abs(self.sim.angular_v[2]-ang_vel_prev[2]))/100)
        next_state=np.array([self.sim.v[2]]).reshape(-1,1)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state=np.array([self.sim.v[2]]).reshape(-1,1)
        return state