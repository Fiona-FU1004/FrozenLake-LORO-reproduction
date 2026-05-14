import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


# =========================
# Paths
# =========================
LLM_DATA_PATH = "/gemini/code/data/FrozenLake_Qwen2.5-7B-Instruct_Neps_200_20260428223255.pkl"
CACHE_PATH = "/gemini/output/cache/cache_FrozenLake_Neps_10.pkl"
SAVE_DIR = "/gemini/output/figs"
SAVE_PATH = os.path.join(SAVE_DIR, "FrozenLake_two_stage_three_lines.png")

os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# Config
# =========================
N_EXP = 5
N_EPISODES = 90
N_LLM_EPISODES = 10

# 你也可以改成 1000
LORO_PRETRAIN_STEPS = 3000


# =========================
# Helpers
# =========================
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


def mean_and_std(runs):
    runs = np.asarray(runs, dtype=float)
    return np.nanmean(runs, axis=0), np.nanstd(runs, axis=0)



def pad_or_trim(arr, length):
    arr = np.asarray(arr, dtype=float)
    if len(arr) >= length:
        return arr[:length]

    out = np.full(length, np.nan, dtype=float)
    out[: len(arr)] = arr
    return out



# =========================
# Load data
# =========================
llm_dataset = load_pickle(LLM_DATA_PATH)
cache = load_pickle(CACHE_PATH)

llm_rewards = np.array(
    [get_episode_return(ep) for ep in llm_dataset.episodes[:N_LLM_EPISODES]],
    dtype=float,
)

# 原作者可视化代码里这里用的是 mean(Qwen_7B_rewards) 作为前 10 个 episode 的值。
# 如果你想显示每一局 LLM reward 的波动，可以把下面这一行改成：
# llm_prefix = llm_rewards
llm_prefix = np.ones(N_LLM_EPISODES) * np.mean(llm_rewards)


# =========================
# Build three curves
# =========================
online_runs = []
loro_runs = []
mix_runs = []

online_len_after_llm = N_EPISODES - N_LLM_EPISODES

for i in range(N_EXP):
    # 1. On-policy:
    # 不使用 LLM 数据，不加载预训练模型，直接在线训练。
    online = pad_or_trim(cache[f"online_{i}"], N_EPISODES)
    online_runs.append(online)

    # 2. LORO:
    # 前 10 局显示 LLM 数据收益；之后接加载预训练模型后的 online fine-tuning reward。
    loro_online = pad_or_trim(
        cache[f"pretrain_7b_{LORO_PRETRAIN_STEPS}_{i}"],
        online_len_after_llm,
    )
    loro = np.concatenate([llm_prefix, loro_online])
    loro_runs.append(loro)

    # 3. Mix data w/o pretrain:
    # 不加载预训练模型，但把 LLM 数据加入 replay buffer，再在线训练。
    mix_online = pad_or_trim(
        cache[f"finetune_7b_{i}"],
        online_len_after_llm,
    )
    mix = np.concatenate([llm_prefix, mix_online])
    mix_runs.append(mix)


online_mean, online_std = mean_and_std(online_runs)
loro_mean, loro_std = mean_and_std(loro_runs)
mix_mean, mix_std = mean_and_std(mix_runs)

x = np.arange(N_EPISODES)


# =========================
# Plot
# =========================
plt.rcParams["font.size"] = 18

fig, ax = plt.subplots(figsize=(16, 8))

# Left stage background
ax.axvspan(
    0,
    N_LLM_EPISODES,
    color="lightgray",
    alpha=0.35,
    zorder=0,
)

# Stage split line
ax.axvline(
    N_LLM_EPISODES,
    color="gray",
    linestyle="--",
    linewidth=1.8,
)

# Lines
ax.plot(
    x,
    loro_mean,
    color="#d73027",
    linewidth=2.8,
    label=f"LORO (cum={int(np.nansum(loro_mean))})",
)
ax.plot(
    x,
    online_mean,
    color="#4575b4",
    linewidth=2.8,
    label=f"On-policy (cum={int(np.nansum(online_mean))})",
)
ax.plot(
    x,
    mix_mean,
    color="#1a9850",
    linewidth=2.8,
    label=f"Mix data w/o pretrain (cum={int(np.nansum(mix_mean))})",
)

# Optional uncertainty bands
ax.fill_between(
    x,
    loro_mean - loro_std / np.sqrt(N_EXP),
    loro_mean + loro_std / np.sqrt(N_EXP),
    color="#d73027",
    alpha=0.15,
)
ax.fill_between(
    x,
    online_mean - online_std / np.sqrt(N_EXP),
    online_mean + online_std / np.sqrt(N_EXP),
    color="#4575b4",
    alpha=0.15,
)
ax.fill_between(
    x,
    mix_mean - mix_std / np.sqrt(N_EXP),
    mix_mean + mix_std / np.sqrt(N_EXP),
    color="#1a9850",
    alpha=0.15,
)

# Text labels
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
    70,
    0.96,
    "Online fine-tuning",
    transform=ax.get_xaxis_transform(),
    color="gray",
    fontsize=18,
    va="top",
)

ax.set_title("FrozenLake Two-stage Training", fontsize=24)
ax.set_xlabel("# of episodes")
ax.set_ylabel("Episode reward")

ax.set_xlim(0, N_EPISODES - 1)
ax.set_ylim(-0.05, 1.05)

ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", fontsize=16, frameon=True)

plt.tight_layout()
plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {SAVE_PATH}")
print(f"LLM first {N_LLM_EPISODES} rewards: {llm_rewards}")
print(f"LLM mean reward: {np.mean(llm_rewards):.4f}")
