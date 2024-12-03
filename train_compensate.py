import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import pybullet as p
import torch
from quadruped_robot_env_compensate import QuadrupedRobotEnv  # Ensure your custom environment is in the correct path

# Environment creation function
def make_env():
    return QuadrupedRobotEnv(use_GUI=False, verbose=False)

# from stable_baselines3.common.env_checker import check_env
# env = QuadrupedRobotEnv(use_GUI=False, verbose=False)
# check_env(env)

# Parallelized environments for vectorized processing
vec_env = make_vec_env(make_env, n_envs=4)

# Define the policy architecture
policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(vf=[256, 512, 512, 512,512], pi=[256, 512, 512, 512, 512],))

# Define the model save path
model_path = "ppo_quadruped_robot_compensate.zip" 
# Check if model exists and load it
if os.path.exists(model_path):
    print("Loading existing model...")
    model = PPO.load(model_path, env=vec_env, learning_rate=1e-6,normalize_advantage=True,
        clip_range=0.2)
else:
    print("Creating a new model...")
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        policy_kwargs=policy_kwargs, 
        verbose=1, 
        learning_rate=1e-6,
        tensorboard_log="./ppo_quadruped_robot/",
        normalize_advantage=True
    )


# Train the model
model.learn(total_timesteps=40000)

# Save the final model
model.save(model_path)

# Disconnect PyBullet to clean up resources
p.disconnect()
