import time
import signal

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_env import DroneEnv  # your env file

MODEL_PATH = "ppo_pioneer_multiinput2.zip"
EPISODES = 3
SHOW_CAMERA = False          # disable OpenCV to avoid Qt/Wayland issues
SLOWDOWN_SEC = 0.0

stop = False
def on_sigint(sig, frame):
    global stop
    stop = True
signal.signal(signal.SIGINT, on_sigint)

def main():
    env = DroneEnv()
    venv = DummyVecEnv([lambda: env])

    print("Loading model:", MODEL_PATH)
    model = PPO.load(MODEL_PATH, env=venv)
    print("Model loaded. Running... (Ctrl+C to stop)")

    try:
        obs = venv.reset()  # this should stop/start sim and place robot at PointA

        for ep in range(1, EPISODES + 1):
            done = False
            steps = 0

            while not done and not stop:
                action, _ = model.predict(obs, deterministic=True)
                obs, rewards, dones, infos = venv.step(action)
                done = bool(dones[0])
                steps += 1

                if SLOWDOWN_SEC > 0:
                    time.sleep(SLOWDOWN_SEC)

            info = infos[0]
            dist = info.get("dist_to_goal", None)

            if stop:
                print("Stopped by user.")
                break

            if dist is not None and dist <= env.goal_tol:
                print(f"Episode {ep}: ✅ SUCCESS (steps={steps}, dist={dist:.3f})")
            else:
                print(f"Episode {ep}: ❌ FAIL (steps={steps}, dist={dist})")

            obs = venv.reset()

    finally:
        venv.close()
        print("Done.")

if __name__ == "__main__":
    main()
