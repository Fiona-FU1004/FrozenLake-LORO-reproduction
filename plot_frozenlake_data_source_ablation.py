import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


LLM_DATA_PATH = "/gemini/code/data/FrozenLake_Qwen2.5-7B-Instruct_Neps_200_20260428223255.pkl"
LORO_CACHE_PATH = "/gemini/output/cache/cache_FrozenLake_Neps_10.pkl"
ON_POLICY_PRETRAIN_CACHE_PATH = "/gemini/output/cache/cache_FrozenLake_on_policy_pretrain_exp.pkl"
RAND_PRETRAIN_CACHE_PATH = "/gemini/output/cache/cache_FrozenLake_on_policy_pretrain_exp_rand.pkl"

SAVE_DIR = "/gemini/output/figs"
SAVE_PATH = os.path.join(SAVE_DIR, "FrozenLake_data_source_ablation.png")

N_EXP = 5
N_LLM_EPISODES = 10
PRETRAIN_STEPS = 3000

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


def cut_all_to_same_length(*arrays):
    n = min(len(arr) for arr in arrays)
    return [arr[:n] for arr in arrays], n


llm_dataset = load_pickle(LLM_DATA_PATH)
loro_cache = load_pickle(LORO_CACHE_PATH)
on_policy_pretrain_cache = load_pickle(ON_POLICY_PRETRAIN_CACHE_PATH)
rand_pretrain_cache = load_pickle(RAND_PRETRAIN_CACHE_PATH)

llm_rewards = np.array(
    [get_episode_return(ep) for ep in llm_dataset.episodes[:N_LLM_EPISODES]],
    dtype=float,
)

# LORO 左侧前 10 个 episode 使用 LLM 数据收益
llm_prefix = np.ones(N_LLM_EPISODES) * np.mean(llm_rewards)

loro_runs = []
on_policy_runs = []
on_policy_data_pretrain_runs = []
random_data_pretrain_runs = []

for i in range(N_EXP):
    loro_online = np.asarray(
        loro_cache[f"pretrain_7b_{PRETRAIN_STEPS}_{i}"],
        dtype=float,
    )
    loro_runs.append(np.concatenate([llm_prefix, loro_online]))

    on_policy_runs.append(
        np.asarray(loro_cache[f"online_{i}"], dtype=float)
    )

    on_policy_data_pretrain_runs.append(
        np.asarray(
            on_policy_pretrain_cache[f"pretrain_10_eps_{PRETRAIN_STEPS}_steps_{i}"],
            dtype=float,
        )
    )

    random_data_pretrain_runs.append(
        np.asarray(
            rand_pretrain_cache[f"pretrain_10_eps_{PRETRAIN_STEPS}_steps_{i}_rand"],
            dtype=float,
        )
    )

loro_mean, loro_sem = mean_sem(loro_runs)
online_mean, online_sem = mean_sem(on_policy_runs)
on_policy_data_mean, on_policy_data_sem = mean_sem(on_policy_data_pretrain_runs)
random_data_mean, random_data_sem = mean_sem(random_data_pretrain_runs)

(
    loro_mean,
    loro_sem,
    online_mean,
    online_sem,
    on_policy_data_mean,
    on_policy_data_sem,
    random_data_mean,
    random_data_sem,
), n_episodes = cut_all_to_same_length(
    loro_mean,
    loro_sem,
    online_mean,
    online_sem,
    on_policy_data_mean,
    on_policy_data_sem,
    random_data_mean,
    random_data_sem,
)

x = np.arange(n_episodes)

plt.rcParams["font.size"] = 18
fig, ax = plt.subplots(figsize=(16, 8))

ax.axvspan(0, N_LLM_EPISODES, color="lightgray", alpha=0.35, zorder=0)
ax.axvline(N_LLM_EPISODES, color="gray", linestyle="--", linewidth=1.8)

ax.plot(
    x,
    loro_mean,
    color="#d73027",
    linewidth=2.8,
    label=f"LORO / LLM data (cum={int(np.nansum(loro_mean))})",
)
ax.fill_between(
    x,
    loro_mean - loro_sem,
    loro_mean + loro_sem,
    color="#d73027",
    alpha=0.15,
)

ax.plot(
    x,
    on_policy_data_mean,
    color="#7b3294",
    linewidth=2.8,
    label=f"Pretrain w/ on-policy data (cum={int(np.nansum(on_policy_data_mean))})",
)
ax.fill_between(
    x,
    on_policy_data_mean - on_policy_data_sem,
    on_policy_data_mean + on_policy_data_sem,
    color="#7b3294",
    alpha=0.15,
)

ax.plot(
    x,
    random_data_mean,
    color="#fdae61",
    linewidth=2.8,
    label=f"Pretrain w/ random data (cum={int(np.nansum(random_data_mean))})",
)
ax.fill_between(
    x,
    random_data_mean - random_data_sem,
    random_data_mean + random_data_sem,
    color="#fdae61",
    alpha=0.15,
)

ax.plot(
    x,
    online_mean,
    color="#444444",
    linestyle="--",
    linewidth=2.4,
    label=f"On-policy baseline (cum={int(np.nansum(online_mean))})",
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
    "Offline data\n+ pre-training",
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

ax.set_title("FrozenLake Data-source Ablation", fontsize=24)
ax.set_xlabel("# of episodes")
ax.set_ylabel("Episode reward")
ax.set_xlim(0, n_episodes - 1)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", fontsize=15, frameon=True)

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {SAVE_PATH}")
print(f"LLM rewards: {llm_rewards}")
print(f"LLM mean reward: {np.mean(llm_rewards):.4f}")
