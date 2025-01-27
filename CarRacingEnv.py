from Weather_simulator import WeatherSimulator
import numpy as np


class CarRacingEnv:
    def __init__(self, env, weather_simulator):
        self.env = env
        self.weather = weather_simulator
        self.max_fuel = 1000.0
        self.fuel = self.max_fuel
        self.prev_speed = 0
        self.speed_history = deque(maxlen=10)
        self.on_track_timer = 0
        self.off_track_count = 0
        self.state = None  # Initialize state as None

        self.discrete_actions = {
            0: [0.0, 0.0, 0.0],
            1: [-1.0, 0.0, 0.0],
            2: [1.0, 0.0, 0.0],
            3: [0.0, 1.0, 0.0],
            4: [0.0, 0.0, 0.8]
        }

    def reset(self):
        self.state, info = self.env.reset()  # Initialize state on reset
        self.weather = WeatherSimulator()  # Use new weather system
        self.fuel = self.max_fuel
        self.prev_speed = 0
        self.speed_history.clear()
        self.on_track_timer = 0
        self.off_track_count = 0
        return self.state, info

    def check_off_track(self, state):
        road_color = state[64:72, 48:56, 0]
        return 0 if np.mean(road_color) > 100 else 1

    def step(self, discrete_action):
        continuous_action = np.array(self.discrete_actions[discrete_action], dtype=np.float16)
        next_state, reward, done, truncated, info = self.env.step(continuous_action)

        self.state = next_state 

        weather_info = self.weather.get_current_conditions()
        info.update({
            'weather': weather_info['weather'],
            'friction': weather_info['friction'],
            'fuel_level': self.fuel,
            'off_track': self.check_off_track(next_state)
        })

        enhanced_reward, reward_info = self.calculate_reward(reward, discrete_action,
                                                            weather_info['friction'])

        return next_state, enhanced_reward, done, truncated, info

    def calculate_reward(self, original_reward, action, friction_factor):
        """Calculate enhanced reward with weather and fuel penalties."""
        if self.state is None:
            raise ValueError("State is not initialized. Call reset() first.")

        fuel_consumption = 0.5 if action == 3 else 0.1 
        self.fuel -= fuel_consumption
        fuel_penalty = -2.0 if self.fuel <= 0 else 0

        speed = np.mean(self.state[84:94, 12, 0]) / 255.0 * 100  
        self.speed_history.append(speed)
        speed_variance = np.var(list(self.speed_history)) if len(self.speed_history) > 1 else 0
        speed_consistency_reward = -0.1 * speed_variance

        if self.weather.current_weather != 'clear':
            weather_penalty = -0.2 * max(0, speed - 50) if speed > 50 else 0
        else:
            weather_penalty = 0

        acceleration = speed - self.prev_speed
        acceleration_penalty = -0.1 * abs(acceleration)
        self.prev_speed = speed

        total_reward = (
            original_reward * friction_factor +
            fuel_penalty +
            speed_consistency_reward +
            weather_penalty +
            acceleration_penalty
        )

        reward_info = {
            'reward_base': original_reward,
            'reward_friction': original_reward * friction_factor - original_reward,
            'reward_fuel': fuel_penalty,
            'reward_speed': speed_consistency_reward,
            'reward_weather': weather_penalty,
            'reward_acceleration': acceleration_penalty
        }

        return total_reward, reward_info