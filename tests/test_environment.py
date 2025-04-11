import unittest
import numpy as np
from earth_lander.envs.earth_lander_env import EarthLanderEnv


class TestEarthLanderEnv(unittest.TestCase):
    def setUp(self):
        self.env = EarthLanderEnv(render_mode=human)

    def test_reset(self):
        state, _ = self.env.reset()
        self.assertEqual(len(state), 9)  # Check state vector size
        self.assertTrue(self.env.lander_fuel > 0)

    def test_step(self):
        self.env.reset()
        action = np.array([0.5, 0.5, 0.5])  # Example action
        next_state, reward, done, _, _ = self.env.step(action)
        self.assertEqual(len(next_state), 9)
        self.assertTrue(isinstance(reward, float))
        self.assertTrue(isinstance(done, bool))

    def test_observation_space(self):
        state, _ = self.env.reset()
        self.assertTrue(self.env.observation_space.contains(state))

    def test_action_space(self):
        action = np.array([0.5, 0.5, 0.5])
        self.assertTrue(self.env.action_space.contains(action))

    def test_landing_success(self):
        self.env.reset()
        # Force lander to landing pad
        self.env.lander_x = self.env.landing_pad_x
        self.env.lander_y = self.env.ground_height + 0.05
        self.env.lander_vx = 0
        self.env.lander_vy = 0
        self.env.lander_angle = 0

        _, _, done, _, info = self.env.step(np.array([0, 0, 0]))
        self.assertTrue(done)
        self.assertTrue(info["landed"])

    def tearDown(self):
        self.env.close()


if __name__ == "__main__":
    unittest.main()