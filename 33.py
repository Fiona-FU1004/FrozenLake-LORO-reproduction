import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATA_PATH = "data/FrozenLake_Qwen2.5-7B-Instruct_Neps_200_20260428223255.pkl"
DEFAULT_CACHE_DIR = "../output/cache"
DEFAULT_FIG_DIR = "../output/figs"


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def episode_return(episode):
    if hasattr(episode, "compute_return"):
        return float(episode.compute_return())
    if hasattr(episode, "rewards"):
        return float(np.asarray(episode.rewards).sum())
    if isinstance(episode, dict) and "rewards" in episode:
        return float(np.asarray(episode["rewards"]).sum())
    raise TypeError(f"Cannot read rewards from episode type: {type(episode)}")


def load_llm_returns(data_path, n_required):
    dataset = load_pickle(data_path)
    if not hasattr(dataset, "episodes"):
        raise TypeError(f"{data_path} does not look like a d3rlpy dataset with .episodes")
    if len(dataset.episodes) < n_required:
        raise ValueError(
            f"{data_path} only has {len(dataset.episodes)} episodes, need {n_required}"
        )
    return np.asarray([episode_return(ep) for ep in dataset.episodes[:n_required]])


def infer_n_exp(cache, prefixes):
    n_exp = 0
    for key in cache:
        for prefix in prefixes:
            if key.startswith(prefix):
                try:
                    n_exp = max(n_exp, int(key.rsplit("_", 1)[-1]) + 1)
                except ValueError:
                    pass
    if n_exp == 0:
        raise KeyError(f"Cannot infer n_exp from keys like: {list(cache)[:10]}")
    return n_exp


def pad_or_trim(values, length):
    values = np.asarray(values, dtype=float)
    if len(values) >= length:
        return values[:length]
    out = np.full(length, np.nan, dtype=float)
    out[: len(values)] = values
    return out


def build_two_stage_curve(cache, llm_returns, n_episodes, n_exp, pretrain_eps, step):
    runs = np.full((n_exp, n_episodes), np.nan, dtype=float)
    runs[:, :pretrain_eps] = llm_returns[:pretrain_eps]

    key_prefix = f"pretrain_7b_{step}_"
    online_len = n_episodes - pretrain_eps
    for i in range(n_exp):
        key = f"{key_prefix}{i}"
        if key not in cache:
            raise KeyError(f"Missing key {key} in cache")
        runs[i, pretrain_eps:] = pad_or_trim(cache[key], online_len)

    mean = np.nanmean(runs, axis=0)
    sem = np.nanstd(runs, axis=0) / np.sqrt(n_exp)
    return mean, sem


def build_direct_curve(cache, n_episodes, n_exp, key_template):
    runs = np.full((n_exp, n_episodes), np.nan, dtype=float)
    for i in range(n_exp):
        key = key_template.format(i=i)
        if key not in cache:
            raise KeyError(f"Missing key {key} in cache")
        runs[i] = pad_or_trim(cache[key], n_episodes)
    mean = np.nanmean(runs, axis=0)
    sem = np.nanstd(runs, axis=0) / np.sqrt(n_exp)
    return mean, sem


def plot_line(ax, x, mean, sem, label, color, linestyle="-", linewidth=2.2):
    ax.plot(x, mean, label=f"{label}. Cum. reward={np.nansum(mean):.0f}", color=color, linestyle=linestyle, linewidth=linewidth)
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.18, linewidth=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".", help="LORO-legacy repository root")
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--fig-dir", default=DEFAULT_FIG_DIR)
    parser.add_argument("--n-episodes", type=int, default=150)
    parser.add_argument("--pretrain-eps", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--steps", type=int, nargs="+", default=[3000])
    parser.add_argument("--show-on-policy", action="store_true")
    parser.add_argument("--output-name", default="FrozenLake_two_stage_7b.pdf")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    data_path = repo_root / args.data_path
    cache_dir = repo_root / args.cache_dir
    fig_dir = repo_root / args.fig_dir
    fig_dir.mkdir(parents=True, exist_ok=True)

    max_pretrain_eps = max(args.pretrain_eps)
    llm_returns = load_llm_returns(data_path, max_pretrain_eps)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(args.n_episodes)

    colors = {
        (10, 1000): "#8e44ad",
        (20, 1000): "#2ca02c",
        (30, 1000): "#1f77b4",
        (10, 3000): "#d62728",
        (20, 3000): "#17becf",
        (30, 3000): "#ff7f0e",
    }

    for pretrain_eps in args.pretrain_eps:
        cache = load_pickle(cache_dir / f"cache_FrozenLake_Neps_{pretrain_eps}.pkl")
        n_exp = infer_n_exp(cache, ["online_", "pretrain_7b_1000_", "pretrain_7b_3000_"])

        if args.show_on_policy and pretrain_eps == args.pretrain_eps[0]:
            mean, sem = build_direct_curve(cache, args.n_episodes, n_exp, "online_{i}")
            plot_line(ax, x, mean, sem, "On-policy DDQN", "#666666", linestyle="--", linewidth=1.8)

        for step in args.steps:
            mean, sem = build_two_stage_curve(
                cache=cache,
                llm_returns=llm_returns,
                n_episodes=args.n_episodes,
                n_exp=n_exp,
                pretrain_eps=pretrain_eps,
                step=step,
            )
            label = f"LORO 7B, {pretrain_eps} pretrain eps, {step} steps"
            plot_line(ax, x, mean, sem, label, colors.get((pretrain_eps, step), None))

    # Draw the largest offline stage as the background split, matching the two-phase reading.
    split = max_pretrain_eps - 0.5
    ax.axvspan(-0.5, split, color="#f2f2f2", zorder=-2)
    ax.axvline(split, color="#999999", linestyle=":", linewidth=1.2)
    ax.text(max_pretrain_eps / 2, 1.02, "Offline Pre-training", transform=ax.get_xaxis_transform(), ha="center", va="bottom", color="#777777", fontsize=11)
    ax.text((max_pretrain_eps + args.n_episodes) / 2, 1.02, "Online Fine-tuning", transform=ax.get_xaxis_transform(), ha="center", va="bottom", color="#777777", fontsize=11)

    ax.set_title("FrozenLake")
    ax.set_xlabel("# of episodes")
    ax.set_ylabel("Episode reward")
    ax.set_xlim(0, args.n_episodes - 1)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    fig.tight_layout()

    output_path = fig_dir / args.output_name
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()