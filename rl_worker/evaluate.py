import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from rl_env import DroneEnv

def evaluate_model(model_path: str, episodes: int, slowdown_sec: float = 0.0, max_steps: int = 2000):
    env = DroneEnv()
    venv = DummyVecEnv([lambda: env])

    model = PPO.load(model_path, env=venv)

    results = {
        "model_path": model_path,
        "episodes": episodes,
        "success_count": 0,
        "episodes_data": []  # per-episode details
    }

    obs = venv.reset()

    for ep in range(1, episodes + 1):
        done = False
        steps = 0
        ep_reward = 0.0
        t0 = time.time()

        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = venv.step(action)

            done = bool(dones[0])
            steps += 1
            ep_reward += float(rewards[0])

            if slowdown_sec > 0:
                time.sleep(slowdown_sec)

        info = infos[0]
        dist = info.get("dist_to_goal", None)
        duration = time.time() - t0

        success = (dist is not None and dist <= env.goal_tol)
        if success:
            results["success_count"] += 1

        results["episodes_data"].append({
            "episode": ep,
            "success": bool(success),
            "steps": steps,
            "reward": ep_reward,
            "dist_to_goal": dist,
            "duration_sec": duration
        })

        obs = venv.reset()

    results["success_rate"] = results["success_count"] / episodes
    venv.close()
    return results
