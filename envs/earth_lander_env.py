import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from typing import Optional, Tuple, Union
import pygame
import random
from pygame import gfxdraw


class EarthLanderEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):
        # Environment parameters
        self.gravity = 9.81  # Earth gravity (m/s^2)
        self.wind_speed = 10.0  # Constant wind speed (m/s)
        self.wind_direction = np.random.uniform(0, 2 * np.pi)  # Random wind direction

        # Drone parameters
        self.lander_mass = 1.0  # kg
        self.lander_width = 1.0
        self.lander_height = 1.0
        self.initial_fuel = 100.0
        self.max_thrust = 15.0
        self.min_thrust = 2.0
        self.thrust_noise = 0.1

        # Environment dimensions
        self.world_width = 100.0  # meters
        self.world_height = 100.0  # meters
        self.ground_height = 10.0  # height of ground from bottom

        # Landing parameters
        self.landing_pad_width = 5.0
        self.landing_pad_height = 1.0
        self._reset_landing_target()

        # Physics parameters
        self.frames_per_second = 50
        self.dt = 1.0 / self.frames_per_second

        # Observation space: [x, y, x_vel, y_vel, angle, angular_vel, left_ground_contact, right_ground_contact, fuel]
        self.observation_space = spaces.Box(
            low=np.array([
                -self.world_width / 2,  # x
                0,  # y
                -np.inf,  # x velocity
                -np.inf,  # y velocity
                -np.pi,  # angle
                -np.pi,  # angular velocity
                0,  # left ground contact
                0,  # right ground contact
                0  # fuel
            ]),
            high=np.array([
                self.world_width / 2,  # x
                self.world_height,  # y
                np.inf,  # x velocity
                np.inf,  # y velocity
                np.pi,  # angle
                np.pi,  # angular velocity
                1,  # left ground contact
                1,  # right ground contact
                self.initial_fuel  # fuel
            ]),
            dtype=np.float32
        )

        # Action space: [left_thrust, right_thrust, bottom_thrust]
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # Rendering setup
        self.render_mode = render_mode
        self.screen_width = 800
        self.screen_height = 800
        self.scale = self.screen_width / self.world_width
        self.screen = None
        self.clock = None
        self.isopen = True

        # Enhanced Earth-like visuals
        self.sky_color = (135, 206, 235)  # Sky blue
        self.sun_color = (255, 255, 100)  # Warm yellow
        self.sun_glow_color = (255, 200, 50)  # Sun glow
        self.land_color = (139, 69, 19)  # Brown earth
        self.grass_color = (34, 139, 34)  # Green grass
        self.ocean_color = (30, 144, 255)  # Deep blue ocean
        self.beach_color = (238, 214, 175)  # Sandy beach
        self.cloud_color = (255, 255, 255, 150)  # Semi-transparent white

        # Drone colors
        self.drone_body_color = (60, 60, 60)  # Dark gray
        self.drone_arm_color = (80, 80, 80)  # Slightly lighter gray
        self.drone_propeller_color = (200, 200, 200)  # Light gray
        self.thruster_flame_color = (255, 100, 0)  # Orange flame

        # UI elements
        self.ui_background_color = (0, 0, 0, 150)  # Semi-transparent black
        self.ui_text_color = (255, 255, 255)  # White

        # Environmental elements
        self.clouds = []
        self._generate_clouds(10)
        self.stars = []
        self._generate_stars(50)

        # Initialize state
        self.reset()

    def _generate_clouds(self, count):
        """Generate random cloud positions and sizes"""
        self.clouds = []
        for _ in range(count):
            cloud = {
                'x': random.randint(0, self.screen_width),
                'y': random.randint(50, int(self.screen_height * 0.6)),
                'size': random.randint(30, 100),
                'speed': random.uniform(0.1, 0.5)
            }
            self.clouds.append(cloud)

    def _generate_stars(self, count):
        """Generate random star positions for the night sky"""
        self.stars = []
        for _ in range(count):
            star = {
                'x': random.randint(0, self.screen_width),
                'y': random.randint(0, int(self.screen_height * 0.7)),
                'size': random.randint(1, 3),
                'brightness': random.randint(150, 255)
            }
            self.stars.append(star)

    def _reset_landing_target(self):
        """Randomly place the landing target within safe bounds"""
        pad_x_min = -self.world_width / 2 + self.landing_pad_width
        pad_x_max = self.world_width / 2 - self.landing_pad_width
        self.landing_pad_x = np.random.uniform(pad_x_min, pad_x_max)

    def _apply_wind(self):
        """Apply wind force to the lander"""
        wind_force_x = math.cos(self.wind_direction) * self.wind_speed * self.dt
        wind_force_y = math.sin(self.wind_direction) * self.wind_speed * self.dt
        self.lander_vx += wind_force_x
        self.lander_vy += wind_force_y

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Randomize wind direction on each reset
        self.wind_direction = np.random.uniform(0, 2 * np.pi)

        # Reset landing target position
        self._reset_landing_target()

        # Initialize lander state
        self.lander_x = np.random.uniform(-self.world_width / 4, self.world_width / 4)
        self.lander_y = self.world_height * 0.75
        self.lander_vx = np.random.uniform(-2.0, 2.0)
        self.lander_vy = np.random.uniform(-2.0, 0)
        self.lander_angle = np.random.uniform(-0.2, 0.2)
        self.lander_angular_vel = np.random.uniform(-0.1, 0.1)
        self.lander_fuel = self.initial_fuel
        self.left_ground_contact = False
        self.right_ground_contact = False

        # Reset episode variables
        self.episode_steps = 0
        self.episode_reward = 0.0

        # Return initial observation
        observation = self._get_observation()
        info = {}

        if self.render_mode == "human":
            self.render()

        return observation, info

    def _get_observation(self):
        return np.array([
            self.lander_x,
            self.lander_y,
            self.lander_vx,
            self.lander_vy,
            self.lander_angle,
            self.lander_angular_vel,
            float(self.left_ground_contact),
            float(self.right_ground_contact),
            self.lander_fuel
        ], dtype=np.float32)

    def _calculate_ground_contact(self):
        """Check if the lander's legs are touching the ground"""
        leg_length = 0.3
        leg_width = 0.1

        # Calculate leg positions
        left_leg_x = self.lander_x - (self.lander_width / 2 + leg_width / 2) * math.cos(self.lander_angle)
        left_leg_y = self.lander_y - (self.lander_width / 2 + leg_width / 2) * math.sin(self.lander_angle)

        right_leg_x = self.lander_x + (self.lander_width / 2 + leg_width / 2) * math.cos(self.lander_angle)
        right_leg_y = self.lander_y + (self.lander_width / 2 + leg_width / 2) * math.sin(self.lander_angle)

        # Check if legs are touching the ground
        self.left_ground_contact = left_leg_y <= self.ground_height + leg_length
        self.right_ground_contact = right_leg_y <= self.ground_height + leg_length

    def step(self, action):
        # Store action for rendering
        self.last_action = action

        # Normalize and clip actions
        left_thrust = np.clip(action[0], 0.0, 1.0) * self.max_thrust
        right_thrust = np.clip(action[1], 0.0, 1.0) * self.max_thrust
        bottom_thrust = np.clip(action[2], 0.0, 1.0) * self.max_thrust

        # Apply thrust noise
        left_thrust += np.random.normal(0, self.thrust_noise)
        right_thrust += np.random.normal(0, self.thrust_noise)
        bottom_thrust += np.random.normal(0, self.thrust_noise)

        # Clip thrust to ensure it's positive
        left_thrust = max(0, left_thrust)
        right_thrust = max(0, right_thrust)
        bottom_thrust = max(0, bottom_thrust)

        # Calculate total thrust and torque
        total_thrust = bottom_thrust
        torque = (right_thrust - left_thrust) * (self.lander_width / 2)

        # Calculate thrust components
        thrust_x = math.sin(self.lander_angle) * total_thrust
        thrust_y = math.cos(self.lander_angle) * total_thrust

        # Update physics
        self.lander_vx += (thrust_x / self.lander_mass) * self.dt
        self.lander_vy += (thrust_y / self.lander_mass - self.gravity) * self.dt
        self.lander_angular_vel += (torque / (
                self.lander_mass * (self.lander_width ** 2 + self.lander_height ** 2) / 12)) * self.dt

        # Apply wind
        self._apply_wind()

        # Update position and angle
        self.lander_x += self.lander_vx * self.dt
        self.lander_y += self.lander_vy * self.dt
        self.lander_angle += self.lander_angular_vel * self.dt

        # Normalize angle to [-pi, pi]
        self.lander_angle = (self.lander_angle + np.pi) % (2 * np.pi) - np.pi

        # Update fuel
        self.lander_fuel -= (left_thrust + right_thrust + bottom_thrust) * self.dt / 10.0
        self.lander_fuel = max(0, self.lander_fuel)

        # Check ground contact
        self._calculate_ground_contact()

        # Check if landed
        done = False
        landed = False
        crashed = False

        # Check if landed on pad
        if (self.lander_y <= self.ground_height + 0.1 and
                abs(self.lander_x - self.landing_pad_x) < self.landing_pad_width / 2):
            landed = True
            done = True

        # Check if crashed
        if (self.lander_y <= self.ground_height + 0.1 and
                abs(self.lander_x - self.landing_pad_x) >= self.landing_pad_width / 2):
            crashed = True
            done = True

        # Check if out of bounds
        if (abs(self.lander_x) > self.world_width / 2 or
                self.lander_y > self.world_height or
                self.lander_y < 0):
            crashed = True
            done = True

        # Calculate reward
        reward = 0.0

        # Distance to landing pad reward
        distance_to_pad = abs(self.lander_x - self.landing_pad_x)
        distance_reward = 1.0 - (distance_to_pad / (self.world_width / 2))

        # Velocity penalty
        velocity_penalty = 0.0
        if abs(self.lander_vx) > 1.0:
            velocity_penalty += abs(self.lander_vx) * 0.1
        if abs(self.lander_vy) > 1.0:
            velocity_penalty += abs(self.lander_vy) * 0.1

        # Angle penalty
        angle_penalty = abs(self.lander_angle) * 0.1

        # Fuel reward
        fuel_reward = self.lander_fuel / self.initial_fuel * 0.1

        # Landing reward
        if landed:
            reward += 100.0
            # Additional rewards for good landing
            reward += (1.0 - abs(self.lander_vx)) * 10.0
            reward += (1.0 - abs(self.lander_vy)) * 10.0
            reward += (1.0 - abs(self.lander_angle)) * 10.0

        if crashed:
            reward -= 100.0

        # Continuous rewards
        reward += distance_reward * 0.1
        reward -= velocity_penalty
        reward -= angle_penalty
        reward += fuel_reward

        # Update episode variables
        self.episode_steps += 1
        self.episode_reward += reward

        # Get observation
        observation = self._get_observation()

        # Info dictionary
        info = {
            "landed": landed,
            "crashed": crashed,
            "distance_to_pad": distance_to_pad,
            "velocity_x": self.lander_vx,
            "velocity_y": self.lander_vy,
            "angle": self.lander_angle,
            "fuel_remaining": self.lander_fuel
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, done, False, info

    def render(self):
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Earth Lander - Realistic Simulation")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.screen_width, self.screen_height), pygame.SRCALPHA)
        canvas.fill(self.sky_color)

        # Draw gradient sky (darker at top)
        for y in range(self.screen_height // 2):
            alpha = int(255 * (1 - y / (self.screen_height // 2)))
            color = (135, 206, 235, alpha)
            pygame.draw.line(canvas, color, (0, y), (self.screen_width, y))

        # Draw stars (for night effect)
        for star in self.stars:
            brightness = star['brightness']
            pygame.draw.circle(canvas, (brightness, brightness, brightness),
                               (star['x'], star['y']), star['size'])

        # Draw sun with glow effect
        sun_pos = (self.screen_width - 80, 80)
        pygame.draw.circle(canvas, self.sun_glow_color, sun_pos, 50)
        pygame.draw.circle(canvas, self.sun_color, sun_pos, 40)

        # Draw clouds
        for cloud in self.clouds:
            # Update cloud position
            cloud['x'] += cloud['speed']
            if cloud['x'] > self.screen_width + cloud['size']:
                cloud['x'] = -cloud['size']

            # Draw cloud with fluffy effect
            size = cloud['size']
            pygame.draw.circle(canvas, self.cloud_color, (int(cloud['x']), int(cloud['y'])), size)
            pygame.draw.circle(canvas, self.cloud_color, (int(cloud['x'] - size // 2), int(cloud['y'] + size // 3)),
                               size // 2)
            pygame.draw.circle(canvas, self.cloud_color, (int(cloud['x'] + size // 2), int(cloud['y'] + size // 3)),
                               size // 2)

        # Draw mountains in the distance
        mountain_base = self.screen_height - self.scale * self.ground_height - 50
        pygame.draw.polygon(canvas, (70, 70, 70), [
            (0, mountain_base + 100),
            (self.screen_width // 4, mountain_base),
            (self.screen_width // 2, mountain_base + 50),
            (3 * self.screen_width // 4, mountain_base + 20),
            (self.screen_width, mountain_base + 80),
            (self.screen_width, mountain_base + 100)
        ])

        # Draw ocean (bottom 20% of screen)
        ocean_height = int(self.screen_height * 0.2)
        pygame.draw.rect(canvas, self.ocean_color,
                         (0, self.screen_height - ocean_height,
                          self.screen_width, ocean_height))

        # Draw waves on ocean
        for x in range(0, self.screen_width, 20):
            wave_height = random.randint(2, 5)
            pygame.draw.arc(canvas, (255, 255, 255, 100),
                            (x, self.screen_height - ocean_height - wave_height, 20, wave_height * 2),
                            0, math.pi, 1)

        # Draw beach (transition between land and ocean)
        beach_height = int(ocean_height * 0.4)
        pygame.draw.rect(canvas, self.beach_color,
                         (0, self.screen_height - ocean_height - beach_height,
                          self.screen_width, beach_height))

        # Draw grassy land area
        land_height = int(self.screen_height * 0.1)
        pygame.draw.rect(canvas, self.land_color,
                         (0, self.screen_height - ocean_height - beach_height - land_height,
                          self.screen_width, land_height))

        # Draw grass details
        for x in range(0, self.screen_width, 5):
            blade_height = random.randint(3, 8)
            pygame.draw.line(canvas, self.grass_color,
                             (x, self.screen_height - ocean_height - beach_height - land_height),
                             (x, self.screen_height - ocean_height - beach_height - land_height - blade_height), 1)

        # Draw landing pad with more detail
        pad_left = self.scale * (self.landing_pad_x + self.world_width / 2 - self.landing_pad_width / 2)
        pad_top = self.screen_height - self.scale * self.ground_height - self.scale * self.landing_pad_height
        pad_width = self.scale * self.landing_pad_width
        pad_height = self.scale * self.landing_pad_height

        # Landing pad base
        pygame.draw.rect(canvas, (100, 100, 100),  # Gray pad
                         (pad_left, pad_top, pad_width, pad_height))

        # Landing pad markings
        pygame.draw.rect(canvas, (0, 200, 0),  # Bright green center
                         (pad_left + pad_width * 0.3, pad_top,
                          pad_width * 0.4, pad_height))

        # Landing pad lights
        for i in range(5):
            light_x = pad_left + (i + 0.5) * (pad_width / 5)
            pygame.draw.circle(canvas, (255, 255, 0),  # Yellow lights
                               (int(light_x), int(pad_top + pad_height / 2)), 3)

        # Draw lander (drone) with more detail
        lander_screen_x = self.scale * (self.lander_x + self.world_width / 2)
        lander_screen_y = self.screen_height - self.scale * self.lander_y

        # Create a surface for the rotated drone
        drone_surface = pygame.Surface((int(self.lander_width * self.scale * 1.5),
                                        int(self.lander_height * self.scale * 1.5)),
                                       pygame.SRCALPHA)

        # Draw drone body (more detailed)
        body_width = int(self.lander_width * self.scale)
        body_height = int(self.lander_height * self.scale)

        # Main body
        pygame.draw.rect(drone_surface, self.drone_body_color,
                         (body_width // 4, body_height // 4,
                          body_width // 2, body_height // 2), border_radius=5)

        # Rounded body ends
        pygame.draw.ellipse(drone_surface, self.drone_body_color,
                            (0, body_height // 4, body_width // 2, body_height // 2))
        pygame.draw.ellipse(drone_surface, self.drone_body_color,
                            (body_width // 2, body_height // 4, body_width // 2, body_height // 2))

        # Draw drone arms (more realistic)
        arm_length = self.lander_width * self.scale * 0.8
        arm_width = 6
        pygame.draw.line(drone_surface, self.drone_arm_color,
                         (body_width // 2, body_height // 2),
                         (body_width // 2 - arm_length // 2, body_height // 2),
                         arm_width)
        pygame.draw.line(drone_surface, self.drone_arm_color,
                         (body_width // 2, body_height // 2),
                         (body_width // 2 + arm_length // 2, body_height // 2),
                         arm_width)

        # Draw propellers with more detail
        prop_radius = 12
        left_prop_pos = (body_width // 2 - arm_length // 2, body_height // 2)
        right_prop_pos = (body_width // 2 + arm_length // 2, body_height // 2)

        # Propeller circles
        pygame.draw.circle(drone_surface, self.drone_propeller_color,
                           left_prop_pos, prop_radius)
        pygame.draw.circle(drone_surface, self.drone_propeller_color,
                           right_prop_pos, prop_radius)

        # Propeller blades (rotating effect)
        blade_length = 15
        angle = pygame.time.get_ticks() / 50  # Rotate based on time

        for i in range(2):  # Two blades per propeller
            # Left propeller
            blade_angle = angle + i * math.pi
            blade_end_x = left_prop_pos[0] + math.cos(blade_angle) * blade_length
            blade_end_y = left_prop_pos[1] + math.sin(blade_angle) * blade_length
            pygame.draw.line(drone_surface, (150, 150, 150),
                             left_prop_pos, (blade_end_x, blade_end_y), 3)

            # Right propeller
            blade_angle = angle + i * math.pi
            blade_end_x = right_prop_pos[0] + math.cos(blade_angle) * blade_length
            blade_end_y = right_prop_pos[1] + math.sin(blade_angle) * blade_length
            pygame.draw.line(drone_surface, (150, 150, 150),
                             right_prop_pos, (blade_end_x, blade_end_y), 3)

        # Draw thruster flames if thrust is being applied
        if hasattr(self, 'last_action'):
            thrust = max(self.last_action)  # Use the maximum thrust value

            if thrust > 0.1:
                flame_height = int(thrust * 20)
                flame_width = 8

                # Bottom thruster flame
                flame_start = (body_width // 2, body_height)
                flame_points = [
                    flame_start,
                    (flame_start[0] - flame_width, flame_start[1] + flame_height),
                    (flame_start[0] + flame_width, flame_start[1] + flame_height)
                ]

                # Inner flame (brighter)
                pygame.draw.polygon(drone_surface, (255, 150, 0), flame_points)

                # Outer flame (more transparent)
                outer_flame_points = [
                    (flame_start[0] - flame_width - 2, flame_start[1] + flame_height + 5),
                    (flame_start[0] + flame_width + 2, flame_start[1] + flame_height + 5)
                ]
                pygame.draw.polygon(drone_surface, (255, 100, 0, 150),
                                    [flame_start] + outer_flame_points)

        # Rotate the drone
        rotated_drone = pygame.transform.rotate(drone_surface, math.degrees(self.lander_angle))

        # Draw the rotated drone with shadow
        drone_rect = rotated_drone.get_rect(center=(lander_screen_x, lander_screen_y))

        # Draw shadow
        shadow_offset = 5
        shadow_surface = pygame.Surface((rotated_drone.get_width(), rotated_drone.get_height()), pygame.SRCALPHA)
        shadow_surface.fill((0, 0, 0, 50))
        shadow_rect = shadow_surface.get_rect(center=(lander_screen_x + shadow_offset,
                                                      lander_screen_y + shadow_offset))
        canvas.blit(shadow_surface, shadow_rect)

        # Draw drone
        canvas.blit(rotated_drone, drone_rect)

        # Draw ground contact indicators (more visible)
        if self.left_ground_contact:
            pygame.draw.circle(canvas, (255, 0, 0, 200),
                               (int(lander_screen_x - self.lander_width * self.scale / 2),
                                int(lander_screen_y + self.lander_height * self.scale / 2)), 8)
        if self.right_ground_contact:
            pygame.draw.circle(canvas, (255, 0, 0, 200),
                               (int(lander_screen_x + self.lander_width * self.scale / 2),
                                int(lander_screen_y + self.lander_height * self.scale / 2)), 8)

        # Draw wind indicator (more detailed)
        wind_indicator_x = 30
        wind_indicator_y = 30
        wind_end_x = wind_indicator_x + math.cos(self.wind_direction) * 40
        wind_end_y = wind_indicator_y + math.sin(self.wind_direction) * 40

        # Wind direction indicator
        pygame.draw.line(canvas, (255, 255, 255, 200),
                         (wind_indicator_x, wind_indicator_y),
                         (wind_end_x, wind_end_y), 3)

        # Arrow head
        arrow_angle = math.atan2(wind_end_y - wind_indicator_y, wind_end_x - wind_indicator_x)
        arrow_size = 10
        arrow_points = [
            (wind_end_x, wind_end_y),
            (wind_end_x - arrow_size * math.cos(arrow_angle + math.pi / 6),
             wind_end_y - arrow_size * math.sin(arrow_angle + math.pi / 6)),
            (wind_end_x - arrow_size * math.cos(arrow_angle - math.pi / 6),
             wind_end_y - arrow_size * math.sin(arrow_angle - math.pi / 6))
        ]
        pygame.draw.polygon(canvas, (255, 255, 255, 200), arrow_points)

        # Wind speed text
        font_small = pygame.font.SysFont('Arial', 16)
        wind_text = f"Wind: {self.wind_speed:.1f} m/s"
        text_surface = font_small.render(wind_text, True, (255, 255, 255))
        pygame.draw.rect(canvas, (0, 0, 0, 150),
                         (wind_indicator_x - 5, wind_indicator_y - 25,
                          text_surface.get_width() + 10, text_surface.get_height() + 10))
        canvas.blit(text_surface, (wind_indicator_x, wind_indicator_y - 20))

        # Enhanced information display (modern UI panel)
        font = pygame.font.SysFont('Arial', 18)
        info_text = [
            f"Horizontal Velocity: {self.lander_vx:.2f} m/s",
            f"Vertical Velocity: {self.lander_vy:.2f} m/s",
            f"Angle: {math.degrees(self.lander_angle):.1f}°",
            f"Fuel Remaining: {self.lander_fuel:.1f}%",
            f"Distance to Pad: {abs(self.lander_x - self.landing_pad_x):.1f} m",
            f"Altitude: {self.lander_y - self.ground_height:.1f} m"
        ]

        # Create UI panel background
        text_widths = [font.size(text)[0] for text in info_text]
        panel_width = max(text_widths) + 20
        panel_height = len(info_text) * 25 + 15
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((0, 0, 0, 180))  # Semi-transparent black

        # Draw panel with border
        pygame.draw.rect(panel_surface, (100, 100, 100, 150),
                         (0, 0, panel_width, panel_height), 2)

        # Add title
        title_font = pygame.font.SysFont('Arial', 20, bold=True)
        title_surface = title_font.render("DRONE STATUS", True, (255, 255, 255))
        panel_surface.blit(title_surface, (panel_width // 2 - title_surface.get_width() // 2, 5))

        # Add info text
        for i, text in enumerate(info_text):
            text_surface = font.render(text, True, (255, 255, 255))
            panel_surface.blit(text_surface, (10, 35 + i * 22))

        # Blit panel to canvas
        canvas.blit(panel_surface, (self.screen_width - panel_width - 20, 20))

        # Add fuel gauge
        self._draw_fuel_gauge(canvas)

        if self.render_mode == "human":
            self.screen.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def _draw_fuel_gauge(self, canvas):
        """Draw a visual fuel gauge"""
        gauge_width = 200
        gauge_height = 20
        gauge_x = 20
        gauge_y = self.screen_height - 40

        # Background
        pygame.draw.rect(canvas, (50, 50, 50, 200),
                         (gauge_x, gauge_y, gauge_width, gauge_height))

        # Fuel level
        fuel_width = int(gauge_width * (self.lander_fuel / self.initial_fuel))
        fuel_color = (
            int(255 * (1 - self.lander_fuel / self.initial_fuel)),
            int(255 * (self.lander_fuel / self.initial_fuel)),
            0
        )
        pygame.draw.rect(canvas, (*fuel_color, 200),
                         (gauge_x, gauge_y, fuel_width, gauge_height))

        # Border and text
        pygame.draw.rect(canvas, (200, 200, 200, 200),
                         (gauge_x, gauge_y, gauge_width, gauge_height), 2)

        font = pygame.font.SysFont('Arial', 16)
        text = font.render("FUEL", True, (255, 255, 255))
        canvas.blit(text, (gauge_x, gauge_y - 20))

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False