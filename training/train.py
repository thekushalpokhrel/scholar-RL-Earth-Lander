import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import tensorflow as tf
from collections import deque
import random
from earth_lander.envs.earth_lander_env import EarthLanderEnv
from dotenv import load_dotenv
load_dotenv()

# Create directories if they don't exist
os.makedirs("training/metrics", exist_ok=True)
os.makedirs("models", exist_ok=True)


def create_q_model(env):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(env.observation_space.shape[0],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(env.action_space.shape[0], activation='sigmoid')
    ])
    return model


def evaluate_model(model, env, num_episodes=5):
    total_reward = 0
    successful_landings = 0

    for episode in range(num_episodes):
        state = env.reset()[0]
        episode_reward = 0
        done = False

        while not done:
            action = model.predict(np.array([state]), verbose=0)[0]
            state, reward, done, _, info = env.step(action)
            episode_reward += reward

            if done and info.get("landed", False):
                successful_landings += 1

        total_reward += episode_reward

    avg_reward = total_reward / num_episodes
    success_rate = successful_landings / num_episodes * 100

    return avg_reward, success_rate


def train_step(model, target_model, batch, optimizer, loss_fn, gamma):
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

    state_batch = np.array(state_batch)
    action_batch = np.array(action_batch)
    next_state_batch = np.array(next_state_batch)
    reward_batch = np.array(reward_batch)
    done_batch = np.array(done_batch)

    # Predict Q values for next state using target model
    next_q_values = target_model.predict(next_state_batch, verbose=0)
    max_next_q_values = np.max(next_q_values, axis=1)

    # Calculate target Q values
    target_q_values = reward_batch + gamma * max_next_q_values * (1 - done_batch)

    # Train the model
    with tf.GradientTape() as tape:
        q_values = model(state_batch)
        q_action = tf.reduce_sum(q_values * action_batch, axis=1)
        loss = loss_fn(target_q_values, q_action)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def train_model():
    # Hyperparameters
    learning_rate = 0.001
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    batch_size = 64
    target_update_freq = 100
    num_episodes = 1000
    evaluation_frequency = 100

    # Initialize environment
    env = EarthLanderEnv(render_mode="human")

    # Initialize models
    model = create_q_model(env)
    target_model = create_q_model(env)
    target_model.set_weights(model.get_weights())

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_fn = tf.keras.losses.Huber()

    # Experience replay buffer
    replay_buffer = deque(maxlen=100000)

    # Log file
    log_filename = "training/metrics/EarthLander_Training.log"

    # Training loop
    step_counter = 0
    for episode in range(num_episodes):
        state = env.reset()[0]
        episode_reward = 0
        landed_successfully = False

        for step in range(1000):
            step_counter += 1

            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = model.predict(np.array([state]), verbose=0)[0]
                action = np.clip(action + np.random.normal(0, 0.1, size=action.shape), 0, 1)

            next_state, reward, done, _, info = env.step(action)
            episode_reward += reward

            if done and info.get("landed", False):
                landed_successfully = True

            replay_buffer.append((state, action, reward, next_state, done))

            state = next_state

            if done:
                break

        # Train if we have enough samples
        if len(replay_buffer) > batch_size:
            batch = random.sample(replay_buffer, batch_size)
            train_step(model, target_model, batch, optimizer, loss_fn, gamma)

        # Update target network
        if step_counter % target_update_freq == 0:
            target_model.set_weights(model.get_weights())

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # Log metrics
        with open(log_filename, 'a') as f:
            f.write(
                f'Episode: {episode + 1}/{num_episodes}, Reward: {episode_reward}, Epsilon: {epsilon}, Landed: {landed_successfully}\n')
        print(
            f'Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}, Epsilon: {epsilon}, Landed: {landed_successfully}')

        # Evaluation with UI
        if (episode + 1) % evaluation_frequency == 0:
            eval_env = EarthLanderEnv(render_mode="human")
            avg_reward, success_rate = evaluate_model(model, eval_env, num_episodes=5)
            print(f"Evaluation after Episode {episode + 1}:")
            print(f"  Average Reward: {avg_reward:.2f}")
            print(f"  Success Rate: {success_rate:.2f}%")
            eval_env.close()

        # Save model periodically
        if (episode + 1) % 100 == 0:
            model.save(f"models/earth_lander_episode_{episode + 1}.keras")

    # Save final model
    model.save("models/earth_lander_final.keras")
    env.close()


if __name__ == "__main__":
    train_model()