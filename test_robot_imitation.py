import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from quadruped_robot_env import QuadrupedRobotEnv  # Ensure custom environment is in the correct path
import pybullet as p

# Function to initialize the environment with GUI
def make_env_with_gui():
    # Create and return a vectorized version of the environment
    def _init():
        return QuadrupedRobotEnv(use_GUI=True, verbose=True)  # Enable GUI for visualization
    return DummyVecEnv([_init])

if __name__ == "__main__":
    model_name = "ppo_quadruped_robot_with_pretrained_policy"
    model_path = os.path.abspath(model_name + ".zip")

    # Check if the model exists and can be read
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Ensure the model has been saved correctly and that you have proper read permissions.")

    print(f"Loading trained model from {model_path}...")
    # Create the environment with GUI
    env = make_env_with_gui()

    try:
        # Load the trained model and associate it with the environment
        model = PPO.load(model_path, env=env)
    except PermissionError:
        raise PermissionError(f"Permission denied when accessing the model at {model_path}. Please check your file permissions.")

    # Run the trained model in the environment
    obs = env.reset()
    print("Running the model in GUI...")

    while True:
        # Predict action using the trained model
        action, _ = model.predict(obs, deterministic=True)
        
        # Step in the environment
        print(action)
        obs, rewards, dones, infos = env.step(action)
        
        if dones[0]:  # VecEnv returns dones as an array
            break

    # Close the PyBullet connection
    env.envs[0].close()  

