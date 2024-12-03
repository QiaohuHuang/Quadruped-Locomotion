import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.optim as optim
from imitate import PPOModel

# Load the pretrained PPO model weights (state_dict)
pretrained_weights = torch.load("ppo_model_policy_pretrained.pth")

# Define the custom PPO model (with new structure)
custom_policy = PPOModel(input_dim=43, action_dim=4)

# Ensure the custom policy is compatible with SB3 (especially the optimizer)
# Manually initialize an optimizer for the custom policy
optimizer = optim.Adam(custom_policy.parameters(), lr=3e-4)

# Attach the optimizer to the custom policy
custom_policy.optimizer = optimizer

# Get the current state_dict of the custom model
custom_policy_state_dict = custom_policy.state_dict()

# Forcefully update the state_dict with pretrained weights
for key in pretrained_weights:
    # Ensure we are only updating weights that are in the custom policy's state_dict
    if key in custom_policy_state_dict:
        print(f"Replacing weights for {key}...")
        custom_policy_state_dict[key] = pretrained_weights[key]
    else:
        print(f"Skipping {key}: Key not found in custom policy.")
    
# Load the custom policy's state_dict into the custom policy model
try:
    custom_policy.load_state_dict(custom_policy_state_dict, strict=False)
    print("Successfully loaded pretrained weights into custom policy.")
except Exception as e:
    print(f"Error while loading pretrained weights: {e}")

# Load the PPO model from StableBaselines3 (this is where SB3's PPO is used)
model2 = PPO.load("ppo_quadruped_robot_with_pretrained_policy.zip")
print("Loaded StableBaselines3 PPO model.")

# Overwrite the policy network in model2 with custom policy
print(model2.policy)
print(custom_policy)
model2.policy.mlp_extractor.policy_net = custom_policy.mlp_extractor.policy_net
model2.policy.action_net= custom_policy.action_net

# Save the updated model with the new policy network
model2.save("ppo_quadruped_robot_with_pretrained_policy.zip")
print("Updated PPO model with forced pretrained policy saved.")
