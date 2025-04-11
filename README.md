# Earth Lander Simulation

## Contact: Kushal Pokhrel (k.pokhrel@aih.edu.au)

### Australian Institute of Higher Education

### Project is under development - feel free to use the code for experimental and research purposes

A custom OpenAI Gym environment simulating a drone landing on Earth with realistic wind effects and dynamic landing targets.

## Features

- 🌍 Earth-like environment with beach, ocean, and sky visuals
- 🚁 Drone-style lander with realistic physics
- 💨 Constant 10 m/s wind with random direction changes
- 🎯 Randomly generated landing targets each episode
- 🏗️ Custom-built using Pygame for rendering
- 🤖 Compatible with reinforcement learning algorithms

## Installation

1. Clone the repository:
   git clone https://github.com/kushalpokhrel/earth_lander_sim.git
   cd earth_lander
   Install dependencies:

pip install -r requirements.txt
Install the package in development mode:

pip install -e .
Usage
Training the Agent
To train a reinforcement learning agent:

python -m earth_lander.training.train
Training progress will be saved in:

training/metrics/ - Training logs and metrics

models/ - Saved model checkpoints

Evaluating a Trained Model
To evaluate a trained model:

python -m earth_lander.training.evaluate --model_path models/earth_lander_final.h5
Manual Control (Development)
For testing the environment manually:

import gymnasium as gym
import earth_lander

env = gym.make('EarthLander-v2', render*mode="human")
observation, * = env.reset()

for \_ in range(1000):
action = env.action_space.sample() # Random actions
observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, _ = env.reset()

env.close()
Environment Details
Action Space
The action space is a Box(3,) representing:

[0]: Left thruster power (0 to 1)

[1]: Right thruster power (0 to 1)

[2]: Bottom thruster power (0 to 1)

Observation Space
The observation space is a Box(9,) representing:

[0]: X position

[1]: Y position

[2]: X velocity

[3]: Y velocity

[4]: Angle "radians"

[5]: Angular velocity

[6]: Left ground contact (boolean)

[7]: Right ground contact (boolean)

[8]: Remaining fuel

Rewards
The reward function considers:

Successful landing on pad: +100

Crash: -100

Distance to landing pad

Velocity penalties

Angle penalties

Fuel remaining bonus

Customization
You can modify environment parameters by creating a custom instance:

from earth_lander.envs.earth_lander_env import EarthLanderEnv

env = EarthLanderEnv(
gravity=9.81, # m/s²
wind_speed=10.0, # m/s
initial_fuel=100.0, # units
max_thrust=15.0, # N
render_mode="human" # None, "human", or "rgb_array"
)
Project Structure

# Earth Lander Project - File Structure

earth_lander/
├── earth_lander/
│ ├── **init**.py
│ ├── envs/
│ │ ├── **init**.py
│ │ └── earth_lander_env.py
│ ├── utils/
│ │ ├── **init**.py
│ │ └── helpers.py
│ └── training/
│ ├── **init**.py
│ ├── train.py # Updated with evaluation fix
│ └── evaluate.py # Evaluation script
├── tests/
│ ├── **init**.py
│ └── test_environment.py
├── docs/
│ ├── README.md
│ └── earth_lander_demo.gif
├── models/ # Model checkpoints
│ ├── earth_lander_episode_100.keras
│ └── earth_lander_final.keras
├── training/
│ └── metrics/
│ └── EarthLander_Training.log
├── manual_play.py # Manual testing script
├── requirements.txt
├── setup.py
└── .gitignore

Contributing
Contributions are welcome! Please open an issue or submit a pull request.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Copy

### Additional Recommendations:

1. Add a `LICENSE` file (MIT License recommended for open source projects)

2. The README provides:
   - Clear installation instructions
   - Usage examples
   - Environment documentation
   - Project structure overview
