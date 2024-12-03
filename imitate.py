import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import json
import gymnasium as gym
from quadruped_robot_env import QuadrupedRobotEnv  # Import your custom environment


# Define the PPO model as per your structure
class MlpExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super(MlpExtractor, self).__init__()

        # Policy network (actor)
        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Tanh()
        )

        # Value network (critic)
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Tanh()
        )

    def forward(self, x):
        # Extract features for both actor and critic
        policy_features = self.policy_net(x)
        value_features = self.value_net(x)
        return policy_features, value_features


class PPOModel(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super(PPOModel, self).__init__()

        # MLP Extractor for policy and value networks, as required by stable-baselines3
        self.mlp_extractor = MlpExtractor(input_dim, hidden_dim)

        # Action network (policy)
        self.action_net = nn.Linear(hidden_dim * 2, action_dim)

        # Value network (critic)
        self.value_net = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        policy_features, value_features = self.mlp_extractor(x)

        action_probs = self.action_net(policy_features)  
        state_value = self.value_net(value_features)  

        return action_probs, state_value


# Define the optimizer
def get_optimizer(model, learning_rate=1e-6):
    return optim.Adam(model.parameters(), lr=learning_rate)


# Generate expert trajectories (observations and actions)
def generate_expert_data(parameters, total_time=50, fps=240):
    expert_obs = []
    expert_actions = []

    env = QuadrupedRobotEnv(use_GUI=False, verbose=False)
    obs, _ = env.reset()
    expert_obs.append(obs)
    for i in range(fps * total_time):
        t = i / fps  # Current time
        action = np.array([  # Generate action using sine wave
            np.sin(2 * np.pi * params["frequency"] * t + params["phase"]) * params["amplitude"]
            for params in parameters.values()
        ])

        obs, _, done, _, _ = env.step(action)
        expert_obs.append(obs)
        expert_actions.append(action)
        if done:
            break
    expert_obs.pop()
    env.close()
    return np.array(expert_obs), np.array(expert_actions)


# Load previously saved locomotion parameters (from evolutionary algorithm)
def load_parameters(file_path="last_generation.json"):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"File {file_path} does not exist.")


# Imitation learning training loop (Policy Network Only)
def imitation_learning_policy_only(model, epochs=100, batch_size=16):
    optimizer = get_optimizer(model.action_net)  # Only optimize the policy network

    parameters = load_parameters()

    # Generate expert data
    expert_obs, expert_actions = generate_expert_data(parameters)

    # Convert expert data to tensors
    expert_states = torch.tensor(expert_obs, dtype=torch.float32)
    expert_actions = torch.tensor(expert_actions, dtype=torch.float32)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Shuffle the expert data for each epoch
        indices = torch.randperm(expert_states.size(0))
        expert_states = expert_states[indices]
        expert_actions = expert_actions[indices]

        # Batch processing
        for i in range(0, len(expert_states), batch_size):
            states_batch = expert_states[i:i+batch_size]
            actions_batch = expert_actions[i:i+batch_size]

            # Forward pass through the model's policy network
            policy_features, _ = model.mlp_extractor(states_batch)  # Ignore value net output
            action_probs = model.action_net(policy_features)

            # Calculate action loss (MSE for continuous actions)
            action_loss = F.mse_loss(action_probs, actions_batch)

            total_loss += action_loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            action_loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Action Loss: {total_loss / len(expert_states)}")


if __name__ == "__main__":
    # Initialize the PPO model
    input_dim = 43  
    action_dim = 4  
    model = PPOModel(input_dim, action_dim)

    # Check if the pretrained model exists
    model_path = "ppo_model_policy_pretrained.pth"
    if os.path.exists(model_path):
        print(f"Loading pretrained model from {model_path}...")
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully.")
    else:
        print("No pretrained model found. Starting from scratch.")

    # Train the model using imitation learning (Policy Only)
    imitation_learning_policy_only(model, epochs=1000)

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

