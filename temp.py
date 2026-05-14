import pickle

with open("/gemini/output/cache/cache_FrozenLake_Neps_10.pkl", "rb") as f:
    cache = pickle.load(f)

for k in cache:
    print(k, len(cache[k]))
