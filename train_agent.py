import os
import csv
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO

# ==============================================================================
# 1. بيئة المحاكاة الأساسية (التدريب الأولي - Baseline)
# ==============================================================================
class GeneratorInverterEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        # التغيير المطلوب في تيار الشحن [-5A, +5A]
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)

        # الملاحظات: [التردد, معدل تغير التردد, تيار الشحن, حرارة البطارية, حمل المنزل]
        self.observation_space = spaces.Box(
            low=np.array([40.0, -10.0, 0.0, 20.0, 0.0], dtype=np.float32),
            high=np.array([60.0, 10.0, 60.0, 70.0, 5.0], dtype=np.float32),
            dtype=np.float32
        )

        self.dt = 0.2
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.freq = 50.0
        self.df_dt = 0.0
        self.charge_current = 5.0
        self.battery_temp = 25.0
        self.house_load = 0.5
        self.step_count = 0

        obs = np.array([self.freq, self.df_dt, self.charge_current, self.battery_temp, self.house_load], dtype=np.float32)
        return obs, {}

    def step(self, action):
        self.step_count += 1
        delta_i = float(action[0])

        prev_current = self.charge_current
        self.charge_current = float(np.clip(self.charge_current + delta_i, 0.0, 50.0))

        if self.step_count % 30 == 0:
            self.house_load = float(np.random.choice([0.5, 1.5, 3.0]))

        # ديناميكا المولد
        charging_power_kw = (self.charge_current * 50.0) / 1000.0
        total_load_kw = self.house_load + charging_power_kw
        generator_capacity_kw = 4.0

        power_imbalance = total_load_kw - generator_capacity_kw * 0.75
        target_freq = 50.0 - (power_imbalance * 1.8)

        governor_response = 0.35
        new_freq = self.freq + governor_response * (target_freq - self.freq)
        self.df_dt = (new_freq - self.freq) / self.dt
        self.freq = float(new_freq)

        # ديناميكا البطارية
        i_jump_stress = abs(self.charge_current - prev_current) / self.dt
        joule_heat = (self.charge_current ** 2) * 0.002
        temp_rise = (joule_heat + (i_jump_stress * 0.01)) * self.dt
        cooling = (self.battery_temp - 25.0) * 0.02 * self.dt
        self.battery_temp += (temp_rise - cooling)

        # دالة المكافأة
        freq_penalty = -5.0 * ((self.freq - 50.0) ** 2)
        battery_stress_penalty = -0.5 * (i_jump_stress ** 2)
        temp_penalty = -2.0 * max(0.0, self.battery_temp - 45.0)

        reward = freq_penalty + battery_stress_penalty + temp_penalty

        terminated = False
        if self.freq < 45.0 or self.freq > 55.0 or self.battery_temp > 60.0:
            reward -= 200.0
            terminated = True

        truncated = self.step_count >= 200
        obs = np.array([self.freq, self.df_dt, self.charge_current, self.battery_temp, self.house_load], dtype=np.float32)

        return obs, reward, terminated, truncated, {}


# ==============================================================================
# 2. وحدة تسجيل التجارب الحقيقية (Real-World Experience Logger)
# ==============================================================================
DATA_FILE = "generator_real_data.csv"

def log_real_experience(obs, action, reward, next_obs, done):
    """تسجيل تجارب التشغيل اليومية لتدريب النموذج عليها لاحقاً"""
    file_exists = os.path.isfile(DATA_FILE)
    with open(DATA_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["f", "df_dt", "i_bat", "temp", "p_load", "action", "reward", "next_f", "done"])
        writer.writerow([obs[0], obs[1], obs[2], obs[3], obs[4], float(action[0]), reward, next_obs[0], done])


# ==============================================================================
# 3. بيئة التعلم المستمر من البيانات الواقعية (Data-Driven Env for Continual Learning)
# ==============================================================================
class ContinualLearningEnv(gym.Env):
    """بيئة تعيد تشغيل الأحداث الواقعية المسجلة وتكيف استجابة الـ Agent"""
    def __init__(self, data_path=DATA_FILE):
        super().__init__()
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([40.0, -10.0, 0.0, 20.0, 0.0], dtype=np.float32),
            high=np.array([60.0, 10.0, 60.0, 70.0, 5.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # قراءة البيانات المخزنة
        self.data = []
        if os.path.exists(data_path):
            with open(data_path, mode="r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.data.append(row)
        self.current_idx = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = 0
        if not self.data:
            return np.array([50.0, 0.0, 5.0, 25.0, 0.5], dtype=np.float32), {}
        
        row = self.data[self.current_idx]
        obs = np.array([float(row["f"]), float(row["df_dt"]), float(row["i_bat"]), float(row["temp"]), float(row["p_load"])], dtype=np.float32)
        return obs, {}

    def step(self, action):
        row = self.data[self.current_idx]
        self.current_idx += 1
        
        obs = np.array([float(row["f"]), float(row["df_dt"]), float(row["i_bat"]), float(row["temp"]), float(row["p_load"])], dtype=np.float32)
        reward = float(row["reward"])
        terminated = row["done"] == "True" or self.current_idx >= len(self.data) - 1
        truncated = False
        
        return obs, reward, terminated, truncated, {}


# ==============================================================================
# 4. نقطة الانطلاق والتدريب
# ==============================================================================
if __name__ == "__main__":
    MODEL_PATH = "inverter_agent_ppo"

    # التحقق: إذا كان هناك نموذج سابق ومسجل بيانات حقيقية كافية -> تحديث النموذج (Fine-Tuning)
    if os.path.exists(f"{MODEL_PATH}.zip") and os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 5000:
        print("[+] تم العثور على نموذج وسجلات واقعية سابقة: جاري التكيف التلقائي (Continual Learning)...")
        model = PPO.load(MODEL_PATH)
        real_env = ContinualLearningEnv(DATA_FILE)
        model.set_env(real_env)
        model.learn(total_timesteps=10000)
        model.save(MODEL_PATH)
        print("[✓] تم تحديث النموذج بنجاح بالبيانات الواقعية.")
    else:
        # التدريب الأولي للمحاكاة
        print("[+] بدء التدريب الأولي الأساسي للـ Agent...")
        env = GeneratorInverterEnv()
        model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003)
        model.learn(total_timesteps=50000)
        model.save(MODEL_PATH)
        print("[✓] تم حفظ النموذج الأولي بنجاح باسم: inverter_agent_ppo.zip")