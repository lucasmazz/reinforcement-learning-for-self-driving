# Deep Reinforcement Learning Implementation for a Simple Self-Driving Car Simulation

This project focuses on the development of a self-driving car simulation system utilizing Deep Q-Learning to train a robotic vehicle for autonomous navigation. A dedicated monitor system was implemented to function as an external observer, equipped with multiple cameras to track an ArUco marker placed on top of the vehicle. This system is used to evaluate the vehicle's trajectory by providing feedback, which is then used to calculate rewards for the Deep Q-Learning algorithm.

![simulation](https://github.com/user-attachments/assets/8912034a-e4f1-4aa4-852d-c656d60b67bf)

The project utilized [CoppeliaSim](https://www.coppeliarobotics.com/) for simulation, with [Python](https://www.python.org/) as the programming language, employing [PyTorch](https://pytorch.org/) for model training and [OpenCV](https://opencv.org/) for computer vision tasks, including ArUco marker tracking.

## How it Works

This project implements a Deep Q-Learning algorithm to train an autonomous vehicle within a simulated environment. The agent learns to control the vehicle's steering to keep the vehicle on track while maintaining a constant speed.

To simplify this learning process, a monitoring system observes the vehicle's movements and feeds this data back to the agent as rewards. Specifically, it uses computer vision to track the vehicle's position and orientation, comparing it to the center of the track. An ArUco marker on the vehicle assists in this process. Consequently, the system calculates distance and angle errors relative to the track, providing real-time reward feedback on the vehicle's trajectory.

![monitor system](https://github.com/user-attachments/assets/ed24828b-5f0d-4583-92ea-431f58e1c221)

In response to the feedback, the agent selects actions based on the current state, using a policy derived from a Convolutional Neural Network (the Q-network). To balance exploration and exploitation, the agent employs an Epsilon Greedy strategy, where it sometimes selects random actions (exploration) and at other times follows the policy's recommendations (exploitation).

Furthermore, to improve learning, the agent stores past experiences (state, action, reward, next state) in a replay memory and samples them randomly during training. This approach enhances the agent's ability to learn from a diverse set of situations, as it helps break correlations between experiences.

Additionally, the agent's policy network is updated to maximize future rewards. To stabilize training, a target network is maintained, which is updated less frequently. This dual-network approach prevents the training from becoming unstable due to rapidly changing targets.

In conclusion, the neural network, applied with Reinforcement Learning, processes images from the car's front camera to determine the appropriate steering angle (left, neutral, or right) and adjusts the vehicle's steering accordingly, ensuring it stays on track.

![deep-q-learning](https://github.com/user-attachments/assets/afacefb5-4efa-4e6b-a8ed-2b5f0413598e)

## Usage

### Requirements

- Python 3.7+
- PyTorch
- OpenCV
- CoppeliaSim (for the simulation environment)
- TensorBoard (for logging and visualization)

### Installation

Clone the repository:

```bash
git clone https://github.com/lucasmazz/reinforcement-learning-for-self-driving
cd reinforcement-learning-for-self-driving
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Make sure [CoppeliaSim](https://www.coppeliarobotics.com/) is installed and properly configured on your system. 

Next, locate the CoppeliaSim remote API library file, typically found within the CoppeliaSim installation directory under `CoppeliaSim/programming/remoteApiBindings/lib/lib`. In newer versions of the simulator, the remote API library file may instead be located at `CoppeliaSim/programming/legacyRemoteApi/remoteApiBindings/lib/lib`.

This directory contains the library files for the supported operating systems available for the remote API. Please select the appropriate file based on your operating system:

- Linux: Copy the `remoteApi.so` file.
- Windows: Copy the `remoteApi.dll` file.
- MacOS: Copy the `remoteApi.dylib` file.

Copy the remote API file into the simulator directory of this repository `reinforcement-learning-for-self-driving/simulator`.

For instance, with Ubuntu 20 Linux:

```bash
cp CoppeliaSim/programming/legacyRemoteApi/remoteApiBindings/lib/lib/Ubuntu20_04/remoteApi.so reinforcement-learning-for-self-driving/simulator
```

Next, open the simulation scene file, simulation.ttt, located in the root folder of this repository, using the CoppeliaSim simulator. Afterward, run the simulation to train and test the model.

### Train
To train the model, run the `train.py` script with the desired command-line arguments. 

Example command to start training:

```bash
python train.py --gpu --debug --save ./checkpoints
```

#### Saving and Loading Checkpoints

The script automatically saves checkpoints and replay memory data at regular intervals specified by `--save_frequency`. You can resume training from a checkpoint using the `--checkpoint` argument and load previous memory data with the `--memory` argument.

Example:

```bash
python train.py --checkpoint ./checkpoints/checkpoint_1000.pt --memory ./checkpoints/memory_replay_1000.data --gpu
```

#### Arguments and Hyperparameters
`--debug`: If set, this flag enables debug mode, which will show additional information and visualizations during training.

`--speed`: Set the car's speed level.

`--outputs`: Number of outputs for the neural network. This argument also corresponds to the number of discrete angle leves that the vehicle can take.

`--batch`: Sets the batch size used for model training. 

`--gamma`: Discount factor for future rewards. This parameter determines how much future rewards are valued compared to immediate rewards.

`--epsilon_start`: The initial value of epsilon used in the epsilon-greedy strategy. This represents the starting exploration rate, where a higher value means more exploration.

`--epsilon_end`: The final value of epsilon in the epsilon-greedy strategy. This is the exploration rate after it has decayed over time, where a lower value means less exploration.

`--epsilon_decay`: The decay rate of epsilon over time, determining how quickly the exploration rate decreases from epsilon_start to epsilon_end.

`--target_update`: Specifies how often (in terms of episodes) the target network's weights are updated to match the policy network's weights.

`--memory_size`: Max size of the replay memory.

`--max_step`: The maximum number of steps allowed per episode before the environment is reset.

`--training_stop`: The threshold for the moving average of the last episodes durations that determines when to stop training.

`--n_episodes`: Total number of episodes for training.

`--n_last_episodes`: The number of recent episodes considered when computing the moving average for stopping training.

`--frames`:  The number of last frames that are combined to represent the state input to the neural network.

`--resize_x`: The width (in pixels) to which the input image is resized for processing by the neural network.

`--resize_y`: The height (in pixels) to which the input image is resized for processing by the neural network.

`--lr`: The learning rate for the optimizer used in training.

`--checkpoint`: Path to a checkpoint file that can be used to resume training from a saved state.

`--memory`: Path to a memory file that contains saved replay memory data, allowing training to resume with previous experiences.

`--save`: Directory to save checkpoints and replay memory data.

`--save_frequency`: Specifies the number of episodes between each save operation for checkpoints and memory data.

`--gpu`: Enable GPU for training.



### Evaluate

To run the simulation using a pre-trained model, execute the `run.py` script with the appropriate command-line arguments.

Example:

```bash
python run.py --checkpoint ./checkpoints/checkpoint_1000.pt --gpu
```

You can [download here](https://drive.google.com/file/d/1365pWQ_PdOLLXAxhdDyvpywVPfYVf9hP/view?usp=sharing) a pre-trained model weight to evaluate the model without training. This model was trained with the default arguments and hyperparameters.

#### Arguments

`--debug`: Enables debug mode, which shows additional information and the processed camera frames.

`--speed`: Sets the car's speed level.

`--outputs`: Specifies the number of outputs for the neural network, corresponding to the range of discrete steering angles.

`--frames`: Defines the number of last frames that are combined to represent the state input to the neural network. 

`--resize_x`: Width (in pixels) to which the input image is resized.

`--resize_y`: Height (in pixels) to which the input image is resized.

`--checkpoint`: Path to the checkpoint file that contains the pre-trained model weights. This is required to run the simulation.

`--gpu`: Enables GPU support for running the simulation if a compatible GPU is available.
