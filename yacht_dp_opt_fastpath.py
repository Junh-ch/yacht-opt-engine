from __future__ import annotations
from collections import Counter
from math import factorial
from typing import Iterable, Sequence, Tuple, Dict, List
import math, os, pickle, atexit

# ============================
# Categories / indices
# ============================
CATS = [
    "Aces","Twos","Threes","Fours","Fives","Sixes",
    "Choice","FourKind","FullHouse","SmallStraight","LargeStraight","Yacht"
]
IDX  = {c:i for i,c in enumerate(CATS)}
UPPER_SET = set(range(6))  # 0..5 (upper section)

# ============================
# Scoring rules
# ============================
def score_from_hist6(h: Tuple[int,int,int,int,int,int], cat_idx: int) -> int:
    # h[i] = count of face (i+1), sum h == 0..5
    s = sum((i+1)*h[i] for i in range(6))
    if cat_idx in UPPER_SET:
        face = cat_idx + 1
        return face * h[face-1]
    if cat_idx == IDX["Choice"]:
        return s
    if cat_idx == IDX["FourKind"]:
        for f in range(5,-1,-1):
            if h[f] >= 4: return (f+1) * h[f]      # Yacht도 face*5로 계산
        return 0
    if cat_idx == IDX["FullHouse"]:
        vals = sorted(h, reverse=True)
        return s if (vals[0]==5 or (vals[0]==3 and vals[1]==2)) else 0
    if cat_idx == IDX["LargeStraight"]:
        if all(h[i]==1 for i in range(5)) and h[5]==0: return 30   # 1..5
        if all(h[i]==1 for i in range(1,6)) and h[0]==0: return 30 # 2..6
        return 0
    if cat_idx == IDX["SmallStraight"]:
        # Large straight(5연속)은 스몰로 인정 X
        if score_from_hist6(h, IDX["LargeStraight"]) > 0: return 0
        ok = (all(h[i]>=1 for i in range(0,4)) or
              all(h[i]>=1 for i in range(1,5)) or
              all(h[i]>=1 for i in range(2,6)))
        return 15 if ok else 0
    if cat_idx == IDX["Yacht"]:
        return 50 if max(h)==5 else 0
    raise ValueError("unknown category")

def upper_bonus(up_sum: int) -> int:
    return 35 if up_sum >= 63 else 0

# ============================
# Histogram generators
# ============================
def gen_hist_sum_eq(n: int):
    for a in range(n+1):
      r1 = n-a
      for b in range(r1+1):
        r2 = r1-b
        for c in range(r2+1):
          r3 = r2-c
          for d in range(r3+1):
            r4 = r3-d
            for e in range(r4+1):
              f = r4 - e
              yield (a,b,c,d,e,f)

def gen_hist_sum_leq(n: int):
    for t in range(n+1):
        yield from gen_hist_sum_eq(t)

# States with sum==5  (252 states)
H5 = tuple(gen_hist_sum_eq(5))
H5_INDEX = {h:i for i,h in enumerate(H5)}

# States with sum<=5  (462 states) -- for holds
HLEQ5 = tuple(gen_hist_sum_leq(5))
HLEQ5_INDEX = {h:i for i,h in enumerate(HLEQ5)}
HLEQ5_SUM   = tuple(sum(h) for h in HLEQ5)

# ============================
# Multinomial PMF tables (k=0..5)
# ============================
def multinom_coeff(counts: Sequence[int]) -> int:
    n = sum(counts); x = factorial(n)
    for v in counts: x //= factorial(v)
    return x

def compositions6(k: int):
    return tuple(gen_hist_sum_eq(k))

DELTA = {}
for k in range(0, 6):
    deltas = compositions6(k)
    probs  = tuple(multinom_coeff(d) * (1/6)**k for d in deltas)
    DELTA[k] = (deltas, probs)

# First-roll distribution over H5 (k=5)
H5_PROBS = {d: p for d,p in zip(*DELTA[5])}

# ============================
# Transition tables & holds
# ============================
KEEP_TRANS: List[Tuple[Tuple[int,...], Tuple[float,...]]] = []
for kept in HLEQ5:
    k = 5 - sum(kept)
    deltas, probs = DELTA[k]
    nxt_idx = []
    for d in deltas:
        h2 = tuple(kept[i] + d[i] for i in range(6))  # becomes sum==5
        nxt_idx.append(H5_INDEX[h2])
    KEEP_TRANS.append((tuple(nxt_idx), probs))

def _enumerate_keeps_from(h: Tuple[int,...]) -> List[int]:
    out = set()
    def rec(i, keep):
        if i==6:
            kept = tuple(keep)
            out.add(HLEQ5_INDEX[kept])
            return
        for t in range(h[i]+1):
            keep.append(t); rec(i+1, keep); keep.pop()
    rec(0, [])
    return list(out)

HOLDS: List[List[int]] = [ _enumerate_keeps_from(h) for h in H5 ]

# ============================
# Masks & capping
# ============================
ALL_UP_MASK  = (1<<6) - 1
ALL_LOW_MASK = (1<<6) - 1
CAP = 63
def cap_up(x: int) -> int: return x if x < CAP else CAP

def cats_in_masks(up_mask: int, low_mask: int) -> List[int]:
    cats = []
    for i in range(6):
        if (up_mask >> i) & 1: cats.append(i)
    for j in range(6):
        if (low_mask >> j) & 1: cats.append(6+j)
    return cats

def masks_without(up_mask: int, low_mask: int, cat_idx: int) -> Tuple[int,int]:
    if cat_idx < 6:
        return (up_mask & ~(1<<cat_idx), low_mask)
    else:
        j = cat_idx - 6
        return (up_mask, low_mask & ~(1<<j))

# ============================
# Fastpaths (exact) for single-category states
# ============================
def _popcount(x: int) -> int:
    return x.bit_count()

def _only_yacht_left(up_mask: int, low_mask: int) -> bool:
    # Lower categories: [Choice, FourKind, FullHouse, SmallStraight, LargeStraight, Yacht] -> indices 0..5
    # Yacht-only => up_mask == 0 and low_mask == (1<<5)
    return up_mask == 0 and low_mask == (1 << 5)

def _only_single_upper_left(up_mask: int, low_mask: int) -> int | None:
    if low_mask != 0: return None
    return int(math.log2(up_mask)) if _popcount(up_mask) == 1 else None  # 0..5

def _only_fourkind_left(up_mask: int, low_mask: int) -> bool:
    # FourKind is lower index 1 (CATS[7])
    return up_mask == 0 and low_mask == (1 << 1)

def _only_large_straight_left(up_mask: int, low_mask: int) -> bool:
    # LargeStraight is lower index 4
    return up_mask == 0 and low_mask == (1 << 4)

def _only_small_straight_left(up_mask: int, low_mask: int) -> bool:
    # SmallStraight is lower index 3
    return up_mask == 0 and low_mask == (1 << 3)

# Occupancy distributions: k dice -> distribution over 6-bit masks
OCC_P: Dict[int, List[float]] = {}
for k in range(0, 6):
    dp = [0.0]*64
    dp[0] = 1.0
    for _ in range(k):
        nxt = [0.0]*64
        for S in range(64):
            pS = dp[S]
            if pS == 0.0: continue
            for face in range(6):
                nxt[S | (1<<face)] += pS * (1/6)
        dp = nxt
    OCC_P[k] = dp

def _prob_straight_only(start_mask: int, r: int, want: str) -> float:
    target_large = [(1<<5)-1, ((1<<5)-1)<<1]  # 1..5 or 2..6
    target_small_any4 = [0b0001111, 0b0011110, 0b0111100]  # 1..4, 2..5, 3..6
    cur = [0.0]*64
    cur[start_mask] = 1.0
    for _ in range(r):
        nxt = [0.0]*64
        for S, pS in enumerate(cur):
            if pS == 0.0: continue
            k = 5 - bin(S).count("1")
            add = OCC_P[k]
            for R, pR in enumerate(add):
                if pR == 0.0: continue
                nxt[S | R] += pS * pR
        cur = nxt
    if want == "large":
        return sum(cur[T] for T in target_large)
    else:
        prob = 0.0
        for S, pS in enumerate(cur):
            if pS == 0.0: continue
            if S in target_large:  # large straight은 small로 인정 X
                continue
            if any((S & m) == m for m in target_small_any4):
                prob += pS
        return prob

def fastpath_plan(up_mask: int, low_mask: int, up_cap: int, h5_idx: int, r: int):
    """
    단일 카테고리 fastpath일 때, 기대값 + 권장 hold(keep_hist/예시/재굴림개수)를 함께 리턴.
    해당 없으면 None.
    """
    if r <= 0:
        return None
    h = H5[h5_idx]

    # 1) Yacht only
    if _only_yacht_left(up_mask, low_mask):
        m = max(h)
        faces = [i for i,c in enumerate(h) if c == m]
        face_idx = max(faces)  # tie-break: 큰 눈
        Z = 5 - m
        q = 1 - (5/6)**r
        ev = 50.0 * (q ** Z)
        terminal = upper_bonus(up_cap)  # ★ 추가: 마지막 칸이면 보너스
        ev += terminal
        keep = [0]*6
        keep[face_idx] = m
        keep_hist = tuple(keep)
        keep_example = [face_idx+1]*m
        return {
            "ev": ev,
            "keep_hist": keep_hist,
            "keep_example": keep_example,
            "reroll_count": Z,
        }

    # 2) Single upper only
    face_idx = _only_single_upper_left(up_mask, low_mask)
    if face_idx is not None:
        f = face_idx + 1
        m = h[face_idx]; Z = 5 - m
        q = 1 - (5/6)**r
        ev_score = f * (m + Z * q)
        if up_cap >= 63:
            pb = 1.0
        else:
            need_x = max(0, math.ceil((63 - up_cap) / f) - m)
            pb = 0.0
            for x in range(need_x, Z + 1):
                pb += math.comb(Z, x) * (q ** x) * ((1 - q) ** (Z - x))
        ev = ev_score + 35.0 * pb
        keep = [0]*6
        keep[face_idx] = m
        keep_hist = tuple(keep)
        keep_example = [face_idx+1]*m
        return {
            "ev": ev,
            "keep_hist": keep_hist,
            "keep_example": keep_example,
            "reroll_count": Z,
        }

    # 3) Four of a kind only
    if _only_fourkind_left(up_mask, low_mask):
        m = max(h)
        candidates = [i for i,cnt in enumerate(h) if cnt == m]
        q = 1 - (5/6)**r; Z = 5 - m
        best_ev = -math.inf; best_face = max(candidates)
        for face_idx in candidates:
            f = face_idx + 1
            ev = 0.0
            for x in range(max(0, 4 - m), Z + 1):
                ev += (m + x) * math.comb(Z, x) * (q ** x) * ((1 - q) ** (Z - x))
            ev *= f
            if ev > best_ev:
                best_ev = ev; best_face = face_idx
        terminal = upper_bonus(up_cap)  # ★ 추가
        best_ev += terminal

        keep = [0]*6
        keep[best_face] = m
        keep_hist = tuple(keep)
        keep_example = [best_face+1]*m
        return {
            "ev": best_ev,
            "keep_hist": keep_hist,
            "keep_example": keep_example,
            "reroll_count": Z,
        }

    # 4) Large straight only
    if _only_large_straight_left(up_mask, low_mask):
        S0 = 0
        for face in range(6):
            if h[face] > 0: S0 |= (1<<face)
        p = _prob_straight_only(S0, r, "large")
        keep = tuple(1 if h[i]>0 else 0 for i in range(6))
        k = 5 - sum(keep)
        keep_example = [i+1 for i in range(6) if keep[i]]
        ev = 30.0 * p
        ev += upper_bonus(up_cap)  # ★ 추가
        return {
            "ev": ev,
            "keep_hist": keep,
            "keep_example": keep_example,
            "reroll_count": k,
        }

    # 5) Small straight only
    if _only_small_straight_left(up_mask, low_mask):
        S0 = 0
        for face in range(6):
            if h[face] > 0: S0 |= (1<<face)
        p = _prob_straight_only(S0, r, "small")
        keep = tuple(1 if h[i]>0 else 0 for i in range(6))
        k = 5 - sum(keep)
        keep_example = [i+1 for i in range(6) if keep[i]]
        ev = 15.0 * p
        ev += upper_bonus(up_cap)  # ★ 추가
        return {
            "ev": ev,
            "keep_hist": keep,
            "keep_example": keep_example,
            "reroll_count": k,
        }

    return None

def fastpath_value(up_mask: int, low_mask: int, up_cap: int, h5_idx: int, r: int) -> float | None:
    """(compat) fastpath 기대값만, 없으면 None."""
    plan = fastpath_plan(up_mask, low_mask, up_cap, h5_idx, r)
    return None if plan is None else plan["ev"]

# ============================
# Disk cache (pickle) + persistent memo for V/Q and EV0 vector
# ============================
CACHE_PATH = "yacht_dp_cache.pkl"
CACHE_VERSION = 3  # fastpath_plan 추가로 버전 증가
V_MEMO: dict[tuple[int,int,int,int,int], float] = {}
Q_MEMO: dict[tuple[int,int,int], float] = {}
EV0_VEC: dict[tuple[int,int,int], List[float]] = {}

def load_cache(path: str = CACHE_PATH) -> tuple[int,int,int]:
    if not os.path.exists(path): return (0,0,0)
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if data.get("version") != CACHE_VERSION:
            return (0,0,0)
        V_MEMO.update(data.get("V", {}))
        Q_MEMO.update(data.get("Q", {}))
        EV0_VEC.update(data.get("EV0", {}))
        return (len(V_MEMO), len(Q_MEMO), len(EV0_VEC))
    except Exception:
        return (0,0,0)

def save_cache(path: str = CACHE_PATH) -> None:
    data = {"version": CACHE_VERSION, "V": V_MEMO, "Q": Q_MEMO, "EV0": EV0_VEC}
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def enable_disk_cache(path: str = CACHE_PATH) -> tuple[int,int,int]:
    loaded = load_cache(path)
    atexit.register(save_cache, path)
    return loaded

# ============================
# DP core: V & Q (with fastpaths; r==1 uses EV0 vector)
# ============================
def Q(up_mask: int, low_mask: int, up_cap: int) -> float:
    key = (up_mask, low_mask, up_cap)
    if key in Q_MEMO:
        return Q_MEMO[key]
    if up_mask == 0 and low_mask == 0:
        val = float(upper_bonus(up_cap))  # <- 여기 필수
        Q_MEMO[key] = val
        return val
    expv = 0.0
    for h in H5:
        p = H5_PROBS[h]
        expv += p * V(up_mask, low_mask, up_cap, H5_INDEX[h], 2)
    Q_MEMO[key] = expv
    return expv

def get_EV0_vector(up_mask: int, low_mask: int, up_cap: int) -> List[float]:
    key = (up_mask, low_mask, up_cap)
    v = EV0_VEC.get(key)
    if v is not None:
        return v
    arr = [0.0]*len(H5)
    for h_idx, h in enumerate(H5):
        cats = cats_in_masks(up_mask, low_mask)
        if not cats:
            arr[h_idx] = float(upper_bonus(up_cap))
            continue
        best = -math.inf
        for c in cats:
            sc  = score_from_hist6(h, c)
            up2 = cap_up(up_cap + (sc if c in UPPER_SET else 0))
            up2_mask, low2_mask = masks_without(up_mask, low_mask, c)
            tail = upper_bonus(up2) if (up2_mask==0 and low2_mask==0) else Q(up2_mask, low2_mask, up2)
            best = max(best, sc + tail)
        arr[h_idx] = best
    EV0_VEC[key] = arr
    return arr

def V(up_mask: int, low_mask: int, up_cap: int, h5_idx: int, r: int) -> float:
    key = (up_mask, low_mask, up_cap, h5_idx, r)
    if key in V_MEMO:
        return V_MEMO[key]

    # Fastpath (exact) if applicable
    plan = fastpath_plan(up_mask, low_mask, up_cap, h5_idx, r)
    if plan is not None:
        V_MEMO[key] = plan["ev"]
        return plan["ev"]

    if r == 0:
        cats = cats_in_masks(up_mask, low_mask)
        if not cats:
            return float(upper_bonus(up_cap))
        h = H5[h5_idx]
        best = -math.inf
        for c in cats:
            sc = score_from_hist6(h, c)
            up2 = cap_up(up_cap + (sc if c in UPPER_SET else 0))
            up2_mask, low2_mask = masks_without(up_mask, low_mask, c)
            tail = upper_bonus(up2) if (up2_mask == 0 and low2_mask == 0) else Q(up2_mask, low2_mask, up2)
            best = max(best, sc + tail)
        return best

    # r > 0: hold enumeration + transition table dot product
    best = -math.inf
    if r == 1:
        EV0 = get_EV0_vector(up_mask, low_mask, up_cap)
        for kept_idx in HOLDS[h5_idx]:
            next_idx_list, prob_list = KEEP_TRANS[kept_idx]
            expv = 0.0
            for j, p in enumerate(prob_list):
                expv += p * EV0[next_idx_list[j]]
            if expv > best:
                best = expv
    else:
        for kept_idx in HOLDS[h5_idx]:
            next_idx_list, prob_list = KEEP_TRANS[kept_idx]
            expv = 0.0
            for j, p in enumerate(prob_list):
                expv += p * V(up_mask, low_mask, up_cap, next_idx_list[j], r-1)
            if expv > best:
                best = expv
    V_MEMO[key] = best
    return best

# ============================
# Public interfaces
# ============================
def choose_best_category_final(
    dice: Sequence[int], remaining: Iterable[str]|Iterable[int], upper_sum: int
):
    if isinstance(next(iter(remaining)), str):
        rem_idx = [IDX[c] for c in remaining]
    else:
        rem_idx = list(remaining)

    up_mask = 0; low_mask = 0
    for c in rem_idx:
        if c < 6: up_mask |= (1<<c)
        else:     low_mask |= (1<<(c-6))

    cnt = Counter(dice)
    h = tuple(cnt.get(i,0) for i in range(1,7))
    h5_idx = H5_INDEX[h]
    up_cap = cap_up(upper_sum)

    rows = []
    for c in rem_idx:
        sc  = score_from_hist6(h, c)
        up2 = cap_up(up_cap + (sc if c in UPPER_SET else 0))
        up2_mask, low2_mask = masks_without(up_mask, low_mask, c)
        tail = upper_bonus(up2) if (up2_mask==0 and low2_mask==0) else Q(up2_mask, low2_mask, up2)
        rows.append((sc+tail, c, sc, tail))
    rows.sort(reverse=True)
    tot, best_c, sc, tail = rows[0]
    return {
        "best_category": CATS[best_c],
        "total_value": tot,
        "immediate": sc,
        "future": tail,
        "ranked": [{"category": CATS[c], "total": t, "immediate": s, "future": f} for (t,c,s,f) in rows]
    }

def choose_best_hold(
    dice: Sequence[int], remaining: Iterable[str]|Iterable[int], upper_sum: int, r: int
):
    if r <= 0: raise ValueError("r must be >= 1")
    if isinstance(next(iter(remaining)), str):
        rem_idx = [IDX[c] for c in remaining]
    else:
        rem_idx = list(remaining)

    up_mask = 0; low_mask = 0
    for c in rem_idx:
        if c < 6: up_mask |= (1<<c)
        else:     low_mask |= (1<<(c-6))

    cnt = Counter(dice)
    h = tuple(cnt.get(i,0) for i in range(1,7))
    h5_idx = H5_INDEX[h]
    up_cap = cap_up(upper_sum)

    # 단일 카테고리 fastpath면 그대로 계획 리턴
    if ((_popcount(up_mask)==1 and low_mask==0) or (up_mask==0 and _popcount(low_mask)==1)):
        plan = fastpath_plan(up_mask, low_mask, up_cap, h5_idx, r)
        if plan is not None:
            return {
                "keep_hist": plan["keep_hist"],
                "keep_example": plan["keep_example"],
                "reroll_count": plan["reroll_count"],
                "expected_value": plan["ev"]
            }

    best = -math.inf; best_keep_idx = None; best_exp = None
    for kept_idx in HOLDS[h5_idx]:
        next_idx_list, prob_list = KEEP_TRANS[kept_idx]
        if r == 1:
            EV0 = get_EV0_vector(up_mask, low_mask, up_cap)
            expv = 0.0
            for j, p in enumerate(prob_list):
                expv += p * EV0[next_idx_list[j]]
        else:
            expv = 0.0
            for j, p in enumerate(prob_list):
                expv += p * V(up_mask, low_mask, up_cap, next_idx_list[j], r-1)
        if expv > best:
            best = expv; best_keep_idx = kept_idx; best_exp = expv

    kept = HLEQ5[best_keep_idx]
    kept_example = []
    for face,count in enumerate(kept, start=1):
        kept_example += [face]*count
    return {
        "keep_hist": kept,
        "keep_example": kept_example,
        "reroll_count": 5 - sum(kept),
        "expected_value": best_exp
    }

def prefill_after_choice(dice, remaining, upper_sum):
    """
    choose_best_category_final가 호출할 Q(up_mask, low_mask, up_cap)를
    '지금 주사위/남은 칸/upper_sum'에 대해 미리 모두 계산해 캐시에 채운다.
    정확도 동일. 이후 choose_best_category_final은 즉시 응답.
    """
    from collections import Counter

    # remaining -> 인덱스
    if isinstance(next(iter(remaining)), str):
        rem_idx = [IDX[c] for c in remaining]
    else:
        rem_idx = list(remaining)

    # 마스크 구성
    up_mask = 0; low_mask = 0
    for c in rem_idx:
        if c < 6: up_mask |= (1<<c)
        else:     low_mask |= (1<<(c-6))

    # 현재 히스토그램/상단합 캡
    cnt = Counter(dice)
    h = tuple(cnt.get(i,0) for i in range(1,7))
    up_cap0 = cap_up(upper_sum)

    # 각 카테고리를 택했을 때의 (up2_mask, low2_mask, up2) 만들고 Q를 선계산
    for c in rem_idx:
        sc  = score_from_hist6(h, c)
        up2 = cap_up(up_cap0 + (sc if c in UPPER_SET else 0))
        up2_mask, low2_mask = masks_without(up_mask, low_mask, c)
        _ = Q(up2_mask, low2_mask, up2)  # <- 여기서 깊은 DP가 돌며 캐시를 채움

def warm_start():
    """Optional: compute start-of-game EV (fills cache). Can be heavy; prefer warm_start_chunked."""
    return Q(ALL_UP_MASK, ALL_LOW_MASK, 0)

def warm_start_chunked(max_hists=40, time_budget_sec=3, save_every=10):
    """Gradually fill cache with time/amount budget."""
    import time
    t0 = time.perf_counter()
    upM, lowM, up0 = ALL_UP_MASK, ALL_LOW_MASK, 0
    done = 0; acc = 0.0
    for idx, h in enumerate(H5):
        p = H5_PROBS[h]
        acc += p * V(upM, lowM, up0, H5_INDEX[h], 2)
        done += 1
        if save_every and done % save_every == 0:
            save_cache(CACHE_PATH)
        if (time.perf_counter() - t0) > time_budget_sec or done >= max_hists:
            break
    save_cache(CACHE_PATH)
    return {"processed": done, "partial_EV": acc, "secs": time.perf_counter()-t0}
