
import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
import time

class QuadrupedRobotEnv(gym.Env):
    def __init__(self, use_GUI=True, verbose=True):
        # Connect to PyBullet
        if use_GUI:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        if verbose:
            self.verbose = True
        else:
            self.verbose = False
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Load URDF data path
        
        # Set the goal and visualize it, only consider x,y coordinates for termination, the goal is slightly above the ground for better visualization
        self.goal = [0,5,0.25]
        # Set the maximum time allowed
        self.max_time = 50
        # Initialize the set of movable joints and their limit
        self.joint_indice = [] # movable joint indices
        self.lower_position_limit = []
        self.upper_position_limit = []
        self.torque_limit = []

        self.start_pos = [0, 0, 0.20]  # Initial position above the ground
        self.start_orn = [0,0,0,1]

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(43,), dtype=np.float32)
        self.reward_dict = {}
        
        self.x_axis_id = None
        self.y_axis_id = None
        self.z_axis_id = None
        self.position_text_id = None  
    
        # Load plane and quadruped robot only once
        self.inited = False
        self.reset()      
        
    
    def reset(self, **kwargs):
        """Reset the environment to an initial state without reloading the URDF."""
        self.num_step = 0
        p.resetSimulation()
        p.setRealTimeSimulation(0)
        p.setGravity(0, 0, -9.8)  # Set gravity
        # Set simulation parameters
        self.time_step = 1.0 / 240.0
        p.setTimeStep(self.time_step)
        GOAL_COLOR = [0,0,1]
        p.addUserDebugPoints([self.goal],[GOAL_COLOR],5)
        p.addUserDebugText("Goal",self.goal,GOAL_COLOR,5)
        self.plane_id = p.loadURDF("plane.urdf", [0, 0, 0])

        #self.robot_id = p.loadURDF("quadruped/spirit40newer.urdf", self.start_pos, self.start_orn, useFixedBase=False)
        self.robot_id = p.loadURDF("urdf/testudog.urdf",self.start_pos,self.start_orn)
        
        if not self.inited:
            self.num_joints = p.getNumJoints(self.robot_id)
            for i in range(self.num_joints):
                joint_state = p.getJointInfo(self.robot_id, i)
                if joint_state[2] == 0 and joint_state[0] % 3 != 0: # if joint is rotate joint and not hip
                    self.joint_indice.append(joint_state[0])
                    self.lower_position_limit.append(joint_state[8])
                    self.upper_position_limit.append(joint_state[9])
                    self.torque_limit.append(joint_state[10])
            self.num_movable_joints = len(self.joint_indice)
            self.inited = True

        state = self._get_observation()
        return state, {}

    def _get_observation(self):
        """Return observation from the environment (joint angles, velocities)."""
        # Get robot position and orientation in world coordinate system (x,y,z) (q1,q2,q3,q4)
        robot_position, robot_quaternion = p.getBasePositionAndOrientation(self.robot_id)
        # 3by3
        robot_orientation = p.getMatrixFromQuaternion(robot_quaternion)

        # Get robot line and angular velocity in world coordinate system
        # [vx,vy,vz] [wx,wy,wz]
        robot_line_velocity, robot_angular_velocity = p.getBaseVelocity(self.robot_id)

        # Get joint position, velocity and torque in joint coordinate system
        joint_states = p.getJointStates(self.robot_id, self.joint_indice)
        
        joint_position = [state[0] for state in joint_states]
        normalized_joint_positions = []
        angle_limit = [[a, b] for a, b in zip(self.lower_position_limit, self.upper_position_limit)]
        for i in range(len(angle_limit)): # normalize joint angle to [-1, 1]
            angle = joint_position[i]
            low = angle_limit[i][0]
            high = angle_limit[i][1]
            angle_after_normalize = 2 * (angle-low) / (high-low) - 1
            normalized_joint_positions.append(angle_after_normalize)
            
        joint_velocity = [state[1] for state in joint_states]
        #scale knee velocity
        for i in range(len(joint_velocity)):
            joint_velocity[i] /= 1e2
        
        joint_torque = [state[3] for state in joint_states]

        for i in range(len(self.torque_limit)): # normalize joint toruqe to [0, 1]
            joint_torque[i] = joint_torque[i] / self.torque_limit[i]

        max_time_step = self.max_time / self.time_step

        return np.concatenate([robot_position, robot_orientation, robot_line_velocity, robot_angular_velocity, \
                               normalized_joint_positions, joint_velocity, joint_torque, [self.num_step/max_time_step]]).astype(np.float32)
    
    def step(self, action):
        """Take an action and return the new state, reward, done, and info."""
        
        np.clip(action,[-1,-1,-1,-1],[1,1,1,1])
        scaled_action = (action + 1) / 2  # Map from (-1, 1) to (0, 1)
        lower = self.lower_position_limit[:4]
        upper = self.upper_position_limit[:4]
        angles = (1 - scaled_action) * lower + scaled_action * upper
        target_positions = np.array([angles[0],angles[1],-angles[0],angles[1],angles[2],angles[3],-angles[2],angles[3]])

        # Apply action to each joint
        p.setJointMotorControlArray(self.robot_id, self.joint_indice, p.POSITION_CONTROL,\
                                    targetPositions=target_positions, forces=self.torque_limit)
        
        # Step simulation
        p.stepSimulation()
        
        # Get updated observation
        observation = self._get_observation()

        # Compute reward and check if the episode is done
        self._compute_reward(observation)
        reward = self.reward_dict["Total"]
        if self.verbose:
            print("=====================================")
            for key, value in self.reward_dict.items():
                if isinstance(value, float):
                    print(f"{key}: {round(value, 12)}")
                else:
                    print(f"{key}: None")
        self.num_step += 1
        done, info, reward_termination = self._is_done()
        reward += reward_termination
        truncated = False
        if info == "the robot does not reach the goal within the time limitation":
            truncated = True
        return observation, reward/100, done, truncated, {"reason":info}

    
    def _compute_reward(self, observation):
        """Reward based on forward movement and stability."""
        robot_position = observation[0: 3]
        robot_orientation = observation[3: 12]
        robot_line_velocity = observation[12: 15]
        angular_velocity = observation[15: 18]
        joint_position = observation[18: 26]
        joint_velocity = observation[26: 34]
        joint_torque = observation[34: 42]

        # Reward the robot for reaching goal
        current_pos = [robot_position[0], robot_position[1]]

        # Moving in y direction
        reward_y = float(robot_position[1])
        
        #pusnishment x
        punishment_x = float(-abs(robot_position[0]))
        # z within a range
        z_lower = 0.15
        z_upper = 0.30
        reward_z = -(robot_position[2] - z_lower)*(robot_position[2] - z_upper)
        punishment_angular_velocity = -angular_velocity[0]*angular_velocity[0]-angular_velocity[1]*angular_velocity[1]-angular_velocity[2]*angular_velocity[2]

        # Reward the robot for maintaining a high linear velocity, within a specified threshold, in the desired direction.
        # Penalize movement in the wrong direction or at very low speeds.
        desired_orientation = np.array([self.goal[0],self.goal[1]]) - np.array(current_pos)
        unit_desired_orientation_vector = desired_orientation / np.linalg.norm(desired_orientation)
        current_velocity_xy = np.array([robot_line_velocity[0], robot_line_velocity[1]])
        reward_velocity = np.dot(current_velocity_xy, unit_desired_orientation_vector)

        # Penalize excessive changes in the z-axis position. 
        current_velocity_z = robot_line_velocity[2]
        punishment_z_velocity = -np.square(current_velocity_z)

        # Penalize extreme joint angles to ensure a natural walking posture.
        punishment_extrme_joint_angle = 0
        angle_threshold = 0.5*0.5
        for angle in joint_position:
            punishment_extrme_joint_angle +=  angle_threshold - angle*angle   
        
        # Penalize large joint angle changes and large joint velocity to promote energy-efficient movement.
        reward_large_joint_velocity = -sum(x*x for x in joint_velocity)

        reward_large_torque = 0
        torque_threshold = 0.5*0.5
        for torque in joint_torque:
             reward_large_torque -= torque*torque-torque_threshold 

        # Penalize non-flat body orientation to maintain stability.
        z_axis = robot_orientation[6:9]
        direction_alignment = np.dot(z_axis, np.array([0,0,1]))
        reward_punish_threshold = 0.9
        reward_flat_body = (direction_alignment - reward_punish_threshold)**3

        reward_y = 1e-2*reward_y
        punishment_x = 1e-2*punishment_x
        reward_z = 1e-1*reward_z
        reward_velocity = 1*reward_velocity
        punishment_angular_velocity = 1e-4*punishment_angular_velocity
        punishment_z_velocity = 1e-4*punishment_z_velocity
        punishment_extrme_joint_angle = 1e-4*punishment_extrme_joint_angle
        punishment_large_velocity = 1e-4*reward_large_joint_velocity
        punishment_large_torque = 1e-4*reward_large_torque
        reward_flat_body = 1e-3*reward_flat_body
        reward_survival = 0.1
        
        
        
        reward = punishment_x + reward_y + reward_z + reward_velocity + punishment_angular_velocity + punishment_z_velocity + punishment_extrme_joint_angle \
                 + punishment_large_velocity + punishment_large_torque + reward_flat_body + reward_survival


        self.reward_dict["X"] = punishment_x
        self.reward_dict["Y"] = reward_y
        self.reward_dict["Z"] = reward_z
        self.reward_dict["Velocity"] = reward_velocity
        self.reward_dict["Angular-velocity"] = punishment_angular_velocity
        self.reward_dict["Z-punish"] = punishment_z_velocity
        self.reward_dict["Joint-angle"] = punishment_extrme_joint_angle
        self.reward_dict["Joint-velocity"] = punishment_large_velocity
        self.reward_dict["Torque"] = punishment_large_torque
        self.reward_dict["Flat-body"] = reward_flat_body
        self.reward_dict["Survival"] = reward_survival
        self.reward_dict["Total"] = reward

    
    def _is_done(self):
        """Check if the episode is done by checking contact with the ground or if the robot is upside down."""
        # Get the orientation of the robot
        position, orientation = p.getBasePositionAndOrientation(self.robot_id)

        # Case 1: The body has large tilt, indicating it has fallen
        rotation_matrix = p.getMatrixFromQuaternion(orientation)
        # The z-axis direction vector (third column of the rotation matrix)
        z_axis = np.array(rotation_matrix[6:9])  # Indexes 6 to 8 represent the z-axis direction 
        # Check if the z-axis is pointing significantly upwards
        if z_axis[2] < 0:  # body frame Z-axis pointing up
            return True, "the body has large tilt, indicating it has fallen", -1

        # Case 2: The base is in contact with the ground
        contact_points = p.getContactPoints(self.robot_id, self.plane_id)
        if len(contact_points) > 0:  # Check for any contact
            for contact in contact_points:
                if contact[3] == -1:  # Contact point for base link 
                    return True, "the base is in contact with the ground", -1
        
        # Case 3: The robot does not reach the goal within the time limitation
        if self.num_step*self.time_step >= self.max_time:
            return True, "the robot does not reach the goal within the time limitation", 0

        # Case 4: Reach the goal successfully, the xy distance of the origin of the body to the goal is less than 1e-6 
        if (position[0]-self.goal[0])**2 + (position[1]-self.goal[1])**2 < 1e-2 and position[1] > self.goal[1]:
            return True, "the robot has reached the goal successfully", 1e6/self.num_step
    
        return False, "", 0  # Continue otherwise
    
    def render(self):
        """Render the environment, drawing the robot's body frame axes."""
        # Get the robot's base position and orientation
        if not self.inited:
            self.reset()
        base_position, orientation = p.getBasePositionAndOrientation(self.robot_id)
        
        # Convert the quaternion to a rotation matrix
        rotation_matrix = p.getMatrixFromQuaternion(orientation)
        # focus,_ = p.getBasePositionAndOrientation(self.robot_id)
        # p.resetDebugVisualizerCamera(cameraDistance=3,cameraYaw=30,cameraPitch=-30,cameraTargetPosition=focus)
        # Define colors for the axes
        colors = {
            'x': [1, 0, 0],  # Red
            'y': [0, 1, 0],  # Green
            'z': [0, 0, 1]   # Blue
        }
        
        # Origin of the axes is the robot's current position
        origin = np.array(base_position)

        # Calculate the end points for each axis
        x_axis_end = origin + 1 * np.array(rotation_matrix[0:3])
        y_axis_end = origin + 1 * np.array(rotation_matrix[3:6])
        z_axis_end = origin + 1 * np.array(rotation_matrix[6:9])

        # Remove previous lines if they exist
        if self.x_axis_id is not None:
            p.removeUserDebugItem(self.x_axis_id)
        if self.y_axis_id is not None:
            p.removeUserDebugItem(self.y_axis_id)
        if self.z_axis_id is not None:
            p.removeUserDebugItem(self.z_axis_id)

        self.x_axis_id = p.addUserDebugLine(origin, x_axis_end, lineColorRGB=colors['x'], lineWidth=1)
        self.y_axis_id = p.addUserDebugLine(origin, y_axis_end, lineColorRGB=colors['y'], lineWidth=1)
        self.z_axis_id = p.addUserDebugLine(origin, z_axis_end, lineColorRGB=colors['z'], lineWidth=1)

        # Update or create the text showing the robot's position
        rounded_position = [round(coord, 2) for coord in base_position]
        text_position = [rounded_position[0],rounded_position[1],rounded_position[2]+2]
        if self.position_text_id is None:
            self.position_text_id = p.addUserDebugText(f"{rounded_position}", text_position, textColorRGB=[1, 1, 0], textSize=1.5)
        else:
            p.removeUserDebugItem(self.position_text_id)
            self.position_text_id = p.addUserDebugText(f"{rounded_position}", text_position, textColorRGB=[1, 1, 0], textSize=1.5)

    def close(self):
        p.disconnect()

if __name__ == "__main__":
    env = QuadrupedRobotEnv(verbose=False)
    
    env.render()

    observation, _ = env.reset()

    joint_position = observation[18: 26]
    for i in range(1000000):
        angle1 = 0.9*np.sin(240*i)
        angle2 = np.sin(240*i)
        angle3 = 0.9*np.sin(240*i)
        angle4 = -0.7*np.sin(240*i)
        # time.sleep(1./60.)  # Rendering at 60 FPS
        
        action = np.array([angle1,angle2,angle3,angle4])

        obs, reward, done, _, info = env.step(action)
        print(obs)
        env.render()
        if done:
            reason = info['reason']
            print(f"Episode finished because {reason}. Reward: {reward}")
            env.reset()
