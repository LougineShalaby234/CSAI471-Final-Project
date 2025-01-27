
import random
# ----------------------------
# 1. Enhanced Weather System
# ----------------------------
class WeatherSimulator:
    def __init__(self):
        self.weather_conditions = ['clear', 'rain', 'snow', 'fog']  
        self.current_weather = random.choice(self.weather_conditions)
        self.weather_duration = random.randint(50, 200)
        self.weather_change_prob = 0.02  
        self.friction_ranges = {
            'clear': (0.9, 1.0),
            'rain': (0.6, 0.75),
            'snow': (0.4, 0.55),
            'fog': (0.8, 0.9) 
        }
        self.current_friction = self._get_initial_friction()

    def _get_initial_friction(self):
        min_f, max_f = self.friction_ranges[self.current_weather]
        return random.uniform(min_f, max_f)

    def update(self):
        if random.random() < self.weather_change_prob:
            self.current_weather = random.choice(self.weather_conditions)
            self.weather_duration = random.randint(100, 300)
            self.current_friction = self._get_initial_friction()
        else:
            self.weather_duration = max(0, self.weather_duration - 1)
            friction_delta = random.uniform(-0.005, 0.005)
            min_f, max_f = self.friction_ranges[self.current_weather]
            self.current_friction = np.clip(
                self.current_friction + friction_delta, min_f, max_f
            )

    def get_current_conditions(self):
        return {
            'weather': self.current_weather,
            'friction': self.current_friction,
            'duration': self.weather_duration
        }
