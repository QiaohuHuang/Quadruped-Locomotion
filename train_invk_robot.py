import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import pybullet as p
from quadruped_robot_env_invk import QuadrupedRobotEnv  # Ensure your custom environment is in the correct path

# Environment creation function
def make_env():
    return QuadrupedRobotEnv(use_GUI=False)

# Parallelized environments for vectorized processing
vec_env = make_vec_env(make_env, n_envs=4)

# Define the policy architecture
policy_kwargs = dict(net_arch=dict(vf=[256, 512, 256], pi=[256, 512, 256]))

# Define the model save path
model_path = "models/ppo_invk_quadruped_robot.zip"  # Ensure the .zip extension is present when saving/loading

# Check if model exists and load it
if os.path.exists(model_path):
    print("Loading existing model...")
    model = PPO.load(model_path, env=vec_env, learning_rate=1e-4,
        clip_range=0.2)
else:
    print("Creating a new model...")
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        policy_kwargs=policy_kwargs, 
        verbose=1, 
        learning_rate=1e-4,
        clip_range=0.2,
        tensorboard_log="logs/ppo_quadruped_robot/"
    )

# Set up evaluation callback
eval_env = make_env()


# Train the model
model.learn(total_timesteps=400000)

# Save the final model
model.save(model_path)

# Disconnect PyBullet to clean up resources
p.disconnect()
