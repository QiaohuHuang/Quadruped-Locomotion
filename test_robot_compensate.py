import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from quadruped_robot_env_compensate import QuadrupedRobotEnv  # Ensure your custom environment is in the correct path
import pybullet as p

# Function to initialize the environment with GUI
def make_env_with_gui():
    # Create and return a vectorized version of the environment
    def _init():
        return QuadrupedRobotEnv(use_GUI=True, verbose=True)  # Enable GUI for visualization
    return DummyVecEnv([_init])

if __name__ == "__main__":
    model_name = "ppo_quadruped_robot_compensate"
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
    obs = env.reset()  # VecEnv reset only returns observations
    print("Running the model in GUI...")

    # Simulation parameters
    fps = 240
    total_time = 50  # Total simulation time in seconds
    i = 0
    total_timestep = fps * total_time
    total_reward = 0
    while i < total_timestep:
        # Predict action using the trained model
        action, _ = model.predict(obs, deterministic=True)
        
        # Step in the environment
        print(action)
        obs, rewards, dones, infos = env.step(action)
        total_reward += rewards[0]
        i += 1
        # End the environment if the episode is done
        if dones[0]:
            print("\n")
            print(f"Finish episode in {i} steps")
            reason = infos[0]["reason"]
            print(f"The episode ended because {reason}")
            #scale back by multiply 100
            print(f"Total reward is : {total_reward*100}")
            print(f"Average reward per step is : {total_reward*100/i}")
            break

    env.close()

