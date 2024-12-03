import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import inv_kine.inv_kine as ik
import time

class QuadrupedRobotEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, use_GUI=True):
        super(QuadrupedRobotEnv, self).__init__()
        self.use_GUI = use_GUI
        if self.use_GUI:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        self.position_text_id = None
        state = self._load_environment()
        self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-60, high=60, shape=(60,), dtype=np.float32)
        # Set simulation parameters
        self.time_step = 1.0 / 240.0
        p.setTimeStep(self.time_step)

    def _load_environment(self):
        self.num_step = 0
        # p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf", [0, 0, 0], [0, 0, 0, 1])
        self.robot_id = p.loadURDF("/urdf/testudog.urdf", [0, 0, 0.25], [0, 0, 0, 1])
        focus,_ = p.getBasePositionAndOrientation(self.robot_id)
        p.resetDebugVisualizerCamera(cameraDistance=1,cameraYaw=-90,cameraPitch=0,cameraTargetPosition=focus)
        # self.robot_id = p.loadURDF("quadruped/spirit40newer.urdf", [0, 0, 0.25], [0, 0, 0, 1])
        self.goal = [0, -5, 0.2]
        GOAL_COLOR = [0, 0, 1]
        p.addUserDebugPoints([self.goal], [GOAL_COLOR], 5)
        p.addUserDebugText("Goal", self.goal, GOAL_COLOR, 5)

        observation = self._get_observation()
        return observation

    def _get_observation(self):
        robot_position = p.getLinkState(self.robot_id, 0)[0]
        robot_rot = p.getLinkState(self.robot_id, 0)[1]
        robot_orientation = p.getEulerFromQuaternion(robot_rot)
        # Get robot line and angular velocity in world coordinate system
        robot_line_velocity = p.getLinkState(self.robot_id,0,computeLinkVelocity=1)[6]
        robot_angular_velocity = p.getLinkState(self.robot_id,0,computeLinkVelocity=1)[7]
        # Get joint position, velocity and torque in joint coordinate system
        joint_indice = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        joint_states = p.getJointStates(self.robot_id, joint_indice)
        joint_position = [state[0] for state in joint_states]
        joint_velocity = [state[1] for state in joint_states]
        joint_torque = [state[3] for state in joint_states]
        joint_power = []
        for i in range(12):
            joint_power.append(p.getJointState(self.robot_id, i)[1] * p.getJointState(self.robot_id, i)[3])

        # 3 + 3 + 3 + 3 + 12 + 12 + 12 + 12
        observation = np.concatenate([robot_position, robot_line_velocity, robot_orientation, \
                              robot_angular_velocity, joint_position, joint_velocity, joint_torque, joint_power]).astype(np.float32)
        return observation

    def reset(self, seed=None):

        p.resetSimulation()
        observation = self._load_environment()
        return observation, {}

    def step(self, action):
        # Codes below comes from https://github.com/varithpu/Quadruped_Robot_Reinforcement_Learning
        action_legpos = np.array([[(action[0] * 0) + 0.1373, (action[0] * 0) - 0.1373, (action[3] * 0) + 0.1373, (action[3] * 0) - 0.1373],
                                  [(action[1] * 0.15) - 0.102, (action[1] * 0.15) - 0.102, (action[4] * 0.15) + 0.252, (action[4] * 0.15) + 0.252],
                                  [(action[2] * 0.05) + 0.05, (action[2] * 0.05) + 0.05, (action[5] * 0.05) + 0.05, (action[5] * 0.05) + 0.05]])
        joint_angle = ik.inv_kine(ik.global2local_legpos(action_legpos, 0, 0, 0.15, 0, 0, 0))
        joint_angle = np.reshape(np.transpose(joint_angle), [1, 12])[0]

        vel1 = action[6:9]
        vel2 = action[9:12]
        p.setJointMotorControlArray(self.robot_id, list(range(12)), p.POSITION_CONTROL, \
                                    targetPositions=joint_angle, targetVelocities=np.block([vel1, vel1, vel2, vel2]),
                                    positionGains=4 * [0.02, 0.02, 0.02], velocityGains=4 * [0.1, 0.1, 0.1])
        # Codes above comes from https://github.com/varithpu/Quadruped_Robot_Reinforcement_Learning
        p.stepSimulation()

        observation = self._get_observation()
        self.num_step += 1
        reward = self._compute_reward(observation)
        done, info, reward_termination = self._is_done()
        reward += reward_termination
        truncated = False
        if info == "the robot does not reach the goal within the time limitation":
            truncated = True
        return observation, reward, done, truncated, {"reason":info}

    def _compute_reward(self, observation):
        """Reward based on forward movement and stability."""
        robot_position = observation[0: 3]
        robot_line_velocity = observation[3: 6]
        robot_orientation = observation[6: 9]
        angular_velocity = observation[9: 12]
        joint_position = observation[12: 24]
        joint_velocity = observation[24: 36]
        joint_torque = observation[36: 48]
        joint_power = observation[48:60]

        # Reward the robot for reaching goal
        reward_reaching_goal = 2 * (10 - np.sqrt(np.square(self.goal[0] - robot_position[0]) + np.square(self.goal[1] - robot_position[1])))

        punishment_power_efficiency = -0.2 * sum(np.abs(joint_power)) * self.time_step

        reward_velocity = -0.5 * abs(robot_position[0]) - 0.5 * robot_line_velocity[1] - 2 * abs((math.pi / 2) - robot_orientation[1])

        punishment_z_position = -0.2 * abs(robot_position[2] - 0.18)

        reward = reward_reaching_goal + punishment_power_efficiency + reward_velocity + punishment_z_position

        return reward


    def _is_done(self):
        """Check if the episode is done by checking contact with the ground or if the robot is upside down."""
        # Get the orientation of the robot
        robot_position = p.getLinkState(self.robot_id,0)[0]
        robot_rot = p.getLinkState(self.robot_id,0)[1]
        robot_orientation = p.getEulerFromQuaternion(robot_rot)

        # terminal fail condition eg robot fall
        if (robot_orientation[1] < 0):
            return True, "the robot has fall over", -20

        # Reach the goal successfully, the xy distance of the origin of the body to the goal is less than 1e-6
        if (robot_position[0] - self.goal[0]) ** 2 + (robot_position[1] - self.goal[1]) ** 2 < 1e-3:
            # print(position)
            return True, "the robot has reached the goal successfully", 100

        return False, "", 0  # Continue otherwise

    def render(self):
        """Render the environment, drawing the robot's body frame axes."""
        # Get the robot's base position and orientation (quaternion)
        base_position, orientation = p.getBasePositionAndOrientation(self.robot_id)

        # Update or create the text showing the robot's position
        rounded_position = [round(coord, 2) for coord in base_position]
        text_position = [rounded_position[0], rounded_position[1], rounded_position[2] + 2]
        if self.position_text_id is None:
            self.position_text_id = p.addUserDebugText(f"{rounded_position}", text_position, textColorRGB=[1, 1, 0],textSize=1.5)
        else:
            p.removeUserDebugItem(self.position_text_id)
            self.position_text_id = p.addUserDebugText(f"{rounded_position}", text_position, textColorRGB=[1, 1, 0],textSize=1.5)

    def close(self):
        p.disconnect()


if __name__ == "__main__":
    env = QuadrupedRobotEnv()

    env.render()

    observation, _ = env.reset()
    joint_position = observation[18: 26]
    for i in range(1000000):
        time.sleep(1. / 60.)  # Rendering at 60 FPS
        action = env.action_space.sample()  # Sample random actions

        obs, reward, done, _, info = env.step(action)
        env.render()
        if done:
            print(f"Episode finished because {info['reason']}. Reward: {reward}")
            env.reset()
