import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


LLM_DATA_PATH = "/gemini/code/data/FrozenLake_Qwen2.5-7B-Instruct_Neps_200_20260428223255.pkl"
CACHE_PATH = "/gemini/output/cache/cache_FrozenLake_Neps_10.pkl"
SAVE_DIR = "/gemini/output/figs"
SAVE_PATH = os.path.join(SAVE_DIR, "FrozenLake_pretrain_steps_1000_vs_3000.png")

N_EXP = 5
N_LLM_EPISODES = 10

os.makedirs(SAVE_DIR, exist_ok=True)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_episode_return(ep):
    if hasattr(ep, "compute_return"):
        return ep.compute_return()
    if hasattr(ep, "rewards"):
        return np.sum(ep.rewards)
    if isinstance(ep, dict) and "rewards" in ep:
        return np.sum(ep["rewards"])
    raise TypeError(f"Unsupported episode type: {type(ep)}")


def mean_sem(runs):
    runs = np.asarray(runs, dtype=float)
    return np.nanmean(runs, axis=0), np.nanstd(runs, axis=0) / np.sqrt(runs.shape[0])


def build_loro_runs(cache, llm_prefix, step):
    runs = []
    for i in range(N_EXP):
        online_part = np.asarray(cache[f"pretrain_7b_{step}_{i}"], dtype=float)
        runs.append(np.concatenate([llm_prefix, online_part]))
    return runs


llm_dataset = load_pickle(LLM_DATA_PATH)
cache = load_pickle(CACHE_PATH)

llm_rewards = np.array(
    [get_episode_return(ep) for ep in llm_dataset.episodes[:N_LLM_EPISODES]],
    dtype=float,
)

# 与原 LORO 可视化保持一致：前 N 个 episode 用 LLM 平均 reward 填充
llm_prefix = np.ones(N_LLM_EPISODES) * np.mean(llm_rewards)

online_runs = [np.asarray(cache[f"online_{i}"], dtype=float) for i in range(N_EXP)]
loro_1000_runs = build_loro_runs(cache, llm_prefix, step=1000)
loro_3000_runs = build_loro_runs(cache, llm_prefix, step=3000)

online_mean, online_sem = mean_sem(online_runs)
loro_1000_mean, loro_1000_sem = mean_sem(loro_1000_runs)
loro_3000_mean, loro_3000_sem = mean_sem(loro_3000_runs)

n_episodes = min(len(online_mean), len(loro_1000_mean), len(loro_3000_mean))
x = np.arange(n_episodes)

online_mean = online_mean[:n_episodes]
online_sem = online_sem[:n_episodes]
loro_1000_mean = loro_1000_mean[:n_episodes]
loro_1000_sem = loro_1000_sem[:n_episodes]
loro_3000_mean = loro_3000_mean[:n_episodes]
loro_3000_sem = loro_3000_sem[:n_episodes]

plt.rcParams["font.size"] = 18
fig, ax = plt.subplots(figsize=(16, 8))

ax.axvspan(0, N_LLM_EPISODES, color="lightgray", alpha=0.35, zorder=0)
ax.axvline(N_LLM_EPISODES, color="gray", linestyle="--", linewidth=1.8)

ax.plot(
    x,
    loro_1000_mean,
    color="#4575b4",
    linewidth=2.8,
    label=f"LORO 1000 steps (cum={int(np.nansum(loro_1000_mean))})",
)
ax.fill_between(
    x,
    loro_1000_mean - loro_1000_sem,
    loro_1000_mean + loro_1000_sem,
    color="#4575b4",
    alpha=0.15,
)

ax.plot(
    x,
    loro_3000_mean,
    color="#d73027",
    linewidth=2.8,
    label=f"LORO 3000 steps (cum={int(np.nansum(loro_3000_mean))})",
)
ax.fill_between(
    x,
    loro_3000_mean - loro_3000_sem,
    loro_3000_mean + loro_3000_sem,
    color="#d73027",
    alpha=0.15,
)

ax.plot(
    x,
    online_mean,
    color="#444444",
    linestyle="--",
    linewidth=2.4,
    label=f"On-policy (cum={int(np.nansum(online_mean))})",
)
ax.fill_between(
    x,
    online_mean - online_sem,
    online_mean + online_sem,
    color="#444444",
    alpha=0.12,
)

ax.text(
    1,
    0.96,
    "LLM data collection\n+ offline pre-training",
    transform=ax.get_xaxis_transform(),
    color="gray",
    fontsize=18,
    va="top",
)

ax.text(
    N_LLM_EPISODES + 25,
    0.96,
    "Online fine-tuning",
    transform=ax.get_xaxis_transform(),
    color="gray",
    fontsize=18,
    va="top",
)

ax.set_title("FrozenLake Pre-training Steps Comparison", fontsize=24)
ax.set_xlabel("# of episodes")
ax.set_ylabel("Episode reward")
ax.set_xlim(0, n_episodes - 1)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", fontsize=16, frameon=True)

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {SAVE_PATH}")
print(f"LLM rewards: {llm_rewards}")
print(f"LLM mean reward: {np.mean(llm_rewards):.4f}")
