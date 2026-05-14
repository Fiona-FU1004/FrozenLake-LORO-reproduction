# FrozenLake LORO Reproduction

This repository contains a FrozenLake-only reproduction of the LORO workflow from
[duongnhatthang/LORO-legacy](https://github.com/duongnhatthang/LORO-legacy).

The reproduction focuses on:

- environment: `FrozenLake-v1`
- model: `Qwen2.5-7B-Instruct`
- LLM data count: 10, 20, and 30 offline episodes
- DDQN offline pre-training steps: 1000 and 3000
- online fine-tuning and visualization

## Workflow

The original LORO workflow has three experiment stages and one visualization stage:

1. Collect LLM trajectories:

   ```bash
   python llm_main.py
   ```

2. Pre-train DDQN from LLM trajectories:

   ```bash
   python pretrain_from_llm.py
   ```

3. Run online fine-tuning and baselines:

   ```bash
   python online_main.py
   python on_policy_pretrain_exp.py
   ```

4. Plot FrozenLake figures:

   ```bash
   python plot_frozenlake_pretrain_steps.py
   python plot_frozenlake_data_source_ablation.py
   ```

## Expected Local Data Layout

Large generated artifacts are intentionally not committed to Git. Place or generate
them with the following paths when reproducing the figures:

```text
gemini/code/data/FrozenLake_Qwen2.5-7B-Instruct_Neps_200_20260428223255.pkl

gemini/output/cache/cache_FrozenLake_Neps_10.pkl
gemini/output/cache/cache_FrozenLake_Neps_20.pkl
gemini/output/cache/cache_FrozenLake_Neps_30.pkl
gemini/output/cache/cache_FrozenLake_on_policy_pretrain_exp.pkl
gemini/output/cache/cache_FrozenLake_on_policy_pretrain_exp_rand.pkl

gemini/output/models/FrozenLake_ddqn_pretrain_7b_1000_steps_10.pkl
gemini/output/models/FrozenLake_ddqn_pretrain_7b_3000_steps_10.pkl
```

The cache file `cache_FrozenLake_Neps_10.pkl` is enough to plot the main
three-line comparison:

- `pretrain_7b_3000_i`: LORO
- `online_i`: on-policy baseline
- `finetune_7b_i`: mix data without pre-training

In the current reproduced cache, the 10-episode setup contains 90 total plotted
episodes: 10 offline LLM/pre-training episodes followed by 80 online fine-tuning
episodes.

## Figures

The included plotting scripts create:

- `FrozenLake_pretrain_steps_1000_vs_3000.png`: LORO 1000-step vs 3000-step
  pre-training comparison.
- `FrozenLake_data_source_ablation.png`: LLM data pre-training vs on-policy
  data pre-training vs random data pre-training.

Generated figures are saved under `gemini/output/figs/` by default.

## Requirements

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The original project also used `d3rlpy`, `gymnasium`, and optional Atari/MuJoCo
dependencies for other environments. This reproduction keeps the FrozenLake
configuration as the primary target.

## Notes

This repository is a project reproduction and adaptation of LORO-legacy for a
FrozenLake-only experiment. Large generated data, caches, and trained model files
are excluded from version control; regenerate them with the scripts above or place
them in the expected local paths.

## License

This project retains the license from the original codebase. See [LICENSE](LICENSE).
