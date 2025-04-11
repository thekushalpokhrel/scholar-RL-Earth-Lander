import numpy as np
import tensorflow as tf
from earth_lander.envs.earth_lander_env import EarthLanderEnv


def evaluate_model(model, env, num_episodes=10, render=True):
    total_reward = 0
    successful_landings = 0

    for episode in range(num_episodes):
        state = env.reset()[0]
        episode_reward = 0
        done = False

        while not done:
            if render:
                env.render()

            # Get the action from the model
            action = model.predict(np.array([state]), verbose=0)[0]

            # Add some noise for evaluation
            action = np.clip(action + np.random.normal(0, 0.1, size=action.shape), 0, 1)

            # Take the action in the environment
            next_state, reward, done, _, info = env.step(action)
            episode_reward += reward
            state = next_state

            # Check for successful landing
            if done and info.get("landed", False):
                successful_landings += 1

        total_reward += episode_reward

    avg_reward = total_reward / num_episodes
    success_rate = successful_landings / num_episodes * 100

    return avg_reward, success_rate


def evaluate_saved_model(model_path, num_episodes=10, render=True):
    env = EarthLanderEnv(render_mode="human" if render else None)
    model = tf.keras.models.load_model(model_path)

    avg_reward, success_rate = evaluate_model(model, env, num_episodes, render)

    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.2f}%")

    env.close()


if __name__ == "__main__":
    evaluate_saved_model("models/earth_lander_final.h5")