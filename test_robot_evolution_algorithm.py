import os
import json
import numpy as np
from quadruped_robot_env import QuadrupedRobotEnv  

def sin_wave(amplitude, frequency, phase, t):
    """
    Generate a sine wave value.
    :param amplitude: Amplitude of the sine wave.
    :param frequency: Frequency of the sine wave.
    :param phase: Phase shift of the sine wave.
    :param t: Current time step.
    :return: Sine wave value.
    """
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)

# Load the parameters from the JSON file
def load_parameters(file_path="last_generation.json"):
    """
    Load previously saved parameters from a JSON file.
    :param file_path: Path to the JSON file.
    :return: Parameters dictionary.
    """
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"File {file_path} does not exist.")

if __name__ == "__main__":
    # Initialize the environment
    env = QuadrupedRobotEnv(use_GUI=True, verbose=True)
    env.reset()

    # Load parameters from the JSON file
    parameters = load_parameters()

    # Simulation parameters
    fps = 240
    total_time = 50  # Total simulation time in seconds

    i = 0
    total_timestep = fps * total_time
    total_reward = 0
    while i < total_timestep:
        t = i / fps  # Current time

        # Generate actions dynamically from loaded parameters
        action = np.array([
            sin_wave(params["amplitude"], params["frequency"], params["phase"], t)
            for params in parameters.values()
        ])
        
        print("Action:", action)
        
        # Step in the environment
        obs, rewards, done, _, infos = env.step(action)
        total_reward += rewards
        i += 1
        # End the environment if the episode is done
        if done:
            print("\n")
            print(f"Finish episode in {i} steps")
            reason = infos["reason"]
            print(f"The episode ended because {reason}")
            #scale back by multiply 100
            print(f"Total reward is : {total_reward*100}\n")
            print(f"Average per step reward is : {total_reward*100/i}")
            break
    env.close()
