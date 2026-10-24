from stable_baselines3 import PPO
from train_agent import GeneratorInverterEnv, log_real_experience

# 1. تحميل النموذج المدرب
model = PPO.load("inverter_agent_ppo")

# 2. إنشاء بيئة الاختبار
env = GeneratorInverterEnv()
obs, _ = env.reset()

print("=" * 80)
print(f"{'الخطوة':<6} | {'التردد (Hz)':<12} | {'تيار الشحن (A)':<15} | {'حمل المنزل (kW)':<16} | {'حرارة البطارية (°C)'}")
print("=" * 80)

total_reward = 0.0

for step in range(40):
    # الـ Agent يتخذ القرار الأمثل بناءً على الحالة الحالية
    action, _states = model.predict(obs, deterministic=True)
    
    # تنفيذ القرار في البيئة
    next_obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    
    # تسجيل الخطوة في سجل البيانات لتغذية التعلم المستمر لاحقاً
    log_real_experience(obs, action, reward, next_obs, terminated or truncated)
    
    freq, df_dt, current, temp, load = next_obs
    print(f"{step+1:<6} | {freq:<12.2f} | {current:<15.2f} | {load:<16.2f} | {temp:.2f}")
    
    obs = next_obs
    
    if terminated:
        print("\n[!] توقف الاختبار: حدث تجاوز لأحد معايير الأمان الحرجة.")
        break

print("=" * 80)
print(f"[✓] اكتمل الاختبار بنجاح. إجمالي المكافأة: {total_reward:.2f}")
print("[✓] تم حفظ بيانات التشغيل في ملف: generator_real_data.csv")