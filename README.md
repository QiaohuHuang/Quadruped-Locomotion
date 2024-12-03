# Quadruped Robot Locomotion

## Project Overview
This project focuses on developing locomotion policies for a quadruped robot using reinforcement learning. An evolutionary algorithm is used to generate a basic periodic movement. Based on this basic periodic movement, a policy that adds compensatory adjustments to the movement is learned using reinforcement learning. The policy is tested and trained in custom OpenAI Gymnasium environments using the PyBullet simulator.

---

## Project Structure

### Environments
- **`quadruped_robot_env.py`**  
  Initial custom OpenAI Gymnasium environment for the quadruped robot, using target joint angles as actions.  
- **`quadruped_robot_env_invk.py`**  
 Custom OpenAI Gymnasium environment for the quadruped robot with inverse kinematics, using target joint position as actions.  
- **`quadruped_robot_env_compensate.py`**  
  Final modified environment using compensatory adjustments as actions.  

### Evolutionary Algorithm
- **`evolution_algorithm.py`**  
  Generates four reference sine waves for joint positions using an evolutionary algorithm.  
- **`last_generation.json`**  
  The best candidate policy generated from `evolution_algorithm.py`.  
- **`test_robot_evolution_algorithm.py`**  
  Evaluates the performance of `last_generation.json` in `quadruped_robot_env.py`.  

### Imitation Learning
- **`imitate.py`**  
  Trains an actor network using behavior cloning based on the policy from the evolutionary algorithm.  
- **`ppo_model_policy_pretrained.pth`**  
  Saved actor network trained by `imitate.py`.  
- **`behaviour_clone.py`**  
  Overwrites the actor network in `ppo_quadruped_robot_with_pretrained_policy.zip` with `ppo_model_policy_pretrained.pth`.  

### Proximal Policy Optimization (PPO)
- **`train_invk_robot.py`**  
  Trains a policy starting with `ppo_invk_quadruped_robot.zip`. If this file does not exist, a new one is generated.  
- **`test_robot_invk.py`**  
  Tests the performance of `ppo_invk_quadruped_robot.zip` in `quadruped_robot_env_invk.py`.  
- **`train_imitation.py`**  
  Trains a policy starting with `ppo_quadruped_robot_with_pretrained_policy.zip`. If this file does not exist, a new one is generated.  
- **`test_robot_imitation.py`**  
  Tests the performance of `ppo_quadruped_robot_with_pretrained_policy.zip` in `quadruped_robot_env.py`.  
- **`train_robot_compensate.py`**  
  Trains a policy that adds compensatory adjustments to the movements defined in `last_generation.json`.  
- **`test_compensate.py`**  
  Tests the performance of `ppo_quadruped_robot_compensate.zip` in `quadruped_robot_env_compensate.py`.  

### Saved Models
- **`ppo_invk_quadruped_robot.zip`**  
  Contains the saved actor network and parameters for training with `train_invk_robot.py`.  
- **`ppo_quadruped_robot_with_pretrained_policy.zip`**  
  Contains the saved actor network and parameters for training with `train_imitation.py`.  
- **`ppo_quadruped_robot_compensate.zip`**  
  Contains the saved actor network and parameters for training with `train_robot_compensate.py`.  

---

## Supporting Files and Directories
- **`models/`**  
  Backup directory containing:  
  - `last_generation.json`  
  - `ppo_invk_quadruped_robot.zip`  
  - `ppo_model_policy_pretrained.pth`  
  - `ppo_quadruped_robot_with_pretrained_policy.zip`  
  - `ppo_quadruped_robot_compensate.zip`  
  The files with same names in the project root can be overwritten during training.  
- **`videos/`**  
  Directory containing performance demonstration videos.  
  - `inv_kinematics_close.mp4` : Performance of policy with inverse kinematics from a close perspective
  - `inv_kinematics_far.mp4` :Performance of policy with inverse kinematics from a far perspective
  - `compensation_close.mp4` : Performance of our final policy from a close perspective
  - `compensation_far.mp4` : Performance of our final policy from a far perspective
  - `evolution_algorithm_close.mp4` : Performance of the basic movement from evolutionary algorithm from a close perspective
  - `evolution_algorithm_far.mp4` : Performance of the basic movement from evolutionary algorithm from a far perspective
  - `imitation_close.mp4` : Performance of the learnt policy using behaviour cloning from a close perspective
  - `imitation_far.mp4` : Performance of the learnt policy using behaviour cloning from a far perspective

- **`urdf/`**  
  Directory containing URDF files defining the quadruped robot structure.  


Used Libraries
---------
Pybullet 3.2.5<br>
Pytorch 2.5.1<br>
Stable baseline3 stable release<br>

These libraries can all be easily implemented by following the instruction below<br>


Setup Gym Environment
---------

For Ubuntu:<br>
To run our example progamme, first,<br>
```python
conda env create -f environment.yml
```

then,<br>
```python
conda activate quadruped_group26
```

finally,<br>
```python
python quadruped_robot_env.py
```

if you have such error:<br>
```python
libGL error: MESA-LOADER: failed to open iris: /usr/lib/dri/iris_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)<br>
libGL error: failed to load driver: irislibGL error: MESA-LOADER: failed to open iris: /usr/lib/dri/iris_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)<br>
libGL error: failed to load driver: irislibGL error: MESA-LOADER: failed to open swrast: /usr/lib/dri/swrast_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)<br>
libGL error: failed to load driver: swrastFailed to create an OpenGL context
```

you can try this command in the environment,<br>
```python
conda install -c conda-forge libstdcxx-ng
```

For Windows:<br>
To run our example progamme, first,<br>
```python
conda create -n quadruped_group26 python==3.11
```

then,<br>
```python
conda activate quadruped_group26
```

next,<br>
```python
pip install -r requirements.txt
```

finally,<br>
```python
python quadruped_robot_env.py
```
```python
python quadruped_robot_env_compensate.py
```
if you have such error:<br>
```python
libGL error: MESA-LOADER: failed to open swrast: /usr/lib/dri/swrast_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)<br>
libGL error: failed to load driver: swrast Failed to create an OpenGL context 
```
you can try,<br>
```python
conda install -c conda-forge gcc
```
> **⚠️ IMPORTANT:**  
> If the installation of **PyBullet** fails, please ensure that the necessary C++ runtime libraries are installed on your system.
### 🐧 Linux
You may need to install the following dependencies:
```bash
sudo apt update
sudo apt install build-essential
sudo apt install gcc-12 g++-12
```

### 🪟 Windows
See the guide on installing PyBullet:  
[How to Install PyBullet Physics Simulation in Windows](https://deepakjogi.medium.com/how-to-install-pybullet-physics-simulation-in-windows-e1f16baa26f6)


> **⚠️ IMPORTANT:**  
> PyTorch and Stable Baseline3 need to be installed manually.

The pytorch version depends on your cuda version. For our group, the cude version is 12.1/12.2, to install right pytorch version
```python
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

To install stable baseline3, run
```python
pip install stable-baselines3[extra]
```
Above is all about how to build the environment of our porject.

## To Test
1. **Ensure required files are in the project root directory**  
   Confirm that the following files exist in the project root directory:  
   - `last_generation.json`  
   - `ppo_invk_quadruped_robot.zip`  
   - `ppo_model_policy_pretrained.pth`  
   - `ppo_quadruped_robot_with_pretrained_policy.zip`  
   - `ppo_quadruped_robot_compensate.zip`  

2. **Copy missing files if necessary**  
   If any of the files mentioned above are missing, copy the corresponding files from the `models/` folder to the project root directory.  

3. **Test the final model**  
   Run the following command to test the performance of the final model with compensatory adjustments:  
   ```bash
   python test_robot_compensate.py
4. **Test the basic movement**  
   Run the following command to evaluate the performance of the basic movements generated by the evolutionary algorithm:
   ```bash
   python test_robot_evolution_algorithm.py
5. **Test the imitation learning model**  
   Run the following command to test the performance of the imitation learning model:  
   ```bash
   python test_robot_imitation.py
6. **Test the inverse kinematics policy model**  
   Run the following command to test the performance of the inverse kinematics policy model:  
   ```bash
   python test_robot_invk.py

## To Train
1. **Ensure required files are in the project root directory**  
   Confirm that the following files exist in the project root directory. If you want to train from scratch, delete them (not recommended):  
   - `last_generation.json`  
   - `ppo_invk_quadruped_robot.zip`  
   - `ppo_model_policy_pretrained.pth`  
   - `ppo_quadruped_robot_with_pretrained_policy.zip`  
   - `ppo_quadruped_robot_compensate.zip`  

2. **Copy missing files if necessary**  
   If any of the files mentioned above are missing, copy the corresponding files from the `models/` folder to the project root directory.  

3. **Use evolutionary algorithm to generate a basic movement**  
   Run the following command. Adjust the number of generations, mutation rate, population size where proper:  
   ```bash
   python evolution_algorithm.py
4. **Train a actor network with behaviour cloning (Optional, only for imitation learning)**  
   Run the following command. Adjust learning rate, number of epochs where proper:
   ```bash
   python imitate.py
5. **Overwrite the PPO actor network with the network from previous step (Optional, only for imitation learning)**  
   Run the following command. Note that this script requires the `ppo_quadruped_robot_with_pretrained_policy.zip` exist under project root:  
   ```bash
   python behavior_clone.py
6. **Train a policy with the pretrained weight (Optional, not recommended, only for imitation learning)**  
   Run the following command. This is likely to degrade the model performance:  
   ```bash
   python train_imitation.py
7. **Train a policy that outputs compensary adjustment**  
   Run the following command:  
   ```bash
   python train_compensate.py
8. **Train a policy with inverse kinematics**  
   Run the following command:  
   ```bash
   python train_invk_robot.py
