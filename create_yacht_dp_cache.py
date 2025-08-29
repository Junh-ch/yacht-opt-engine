import os, gzip, pickle
import yacht_dp_opt_fastpath as y

SLIM = os.path.join(os.getcwd(),"yacht_dp_cache_slim.pkl.gz")  # 경로만 바꿔

# Q/EV0만 슬림 저장 (버전은 현재 코드의 CACHE_VERSION로)
os.makedirs(os.path.dirname(SLIM), exist_ok=True)
with gzip.open(SLIM, "wb") as f:
    pickle.dump({"version": y.CACHE_VERSION, "Q": y.Q_MEMO, "EV0": y.EV0_VEC},
                f, protocol=pickle.HIGHEST_PROTOCOL)
print("saved:", SLIM)
