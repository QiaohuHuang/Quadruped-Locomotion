import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from quadruped_robot_env_invk import QuadrupedRobotEnv  # Ensure your custom environment is in the correct path
import pybullet as p
import time


# Function to initialize the environment with GUI
def make_env_with_gui():
    # Create and return a vectorized version of the environment
    def _init():
        return QuadrupedRobotEnv(use_GUI=True)  # Enable GUI for visualization

    return DummyVecEnv([_init])


# Load the trained model
model_path = "models/ppo_invk_quadruped_robot"

if not os.path.exists(model_path + ".zip"):
    raise FileNotFoundError(f"Model not found at {model_path}.zip")

print("Loading trained model...")
env = make_env_with_gui()  # Create the environment with GUI
model = PPO.load(model_path, env=env)  # Load the trained model and associate it with the environment

# Run the trained model in the environment
obs = env.reset()  # VecEnv reset only returns observations
print("Running the model in GUI...")

# record the result
# log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "inv_kinematics_close.mp4")
while True:
    # Predict action using the trained model
    action, _ = model.predict(obs, deterministic=True)
    print(action)

    # Step in the environment
    # print(action)
    obs, rewards, dones, infos = env.step(action)

    # Reset the environment if the episode is done
    if dones[0]:  # VecEnv returns `dones` as an array
        obs = env.reset()
    time.sleep(0.01)

# p.stopStateLogging(log_id)
# Note: In practice, this loop may run indefinitely unless interrupted manually.
# Consider adding a condition to break the loop or limit the number of steps for testing.

# Close the PyBullet connection when done
env.envs[0].close()  # Close the underlying environment explicitly
