# Yacht 최적 플레이 엔진 – 간단 사용 설명서

## 이게 뭐냐

Yacht(야추) 게임에서 **지금 최선의 선택**(어떤 주사위를 들고 갈지, 어떤 칸에 적을지)과 **기대 점수(EV)**를 정확 계산(DP)으로 알려준다.

Jupyter/파이썬에서 바로 쓴다.

------

## 파일 구성

- `yacht_dp_opt_fastpath.py` : 엔진(정확 DP + 빠른 경로 최적화 + 캐시)
- `create_yacht_dp_cache.py` : `yacht_dp_cache_slim.pkl.gz` 파일을 생성
- (선택) `yacht_dp_cache_slim.pkl.gz` : **빠른 로딩용 슬림 캐시**(Q/EV0만)

> 슬림 캐시가 있으면 첫 실행부터 빠르다. 없으면 처음 몇 번 계산은 시간이 (아주 많이) 걸릴 수 있다.

------

## 설치 요건

- Python 3.10+ (권장 3.11+)
- (선택) `numpy` 설치 시 r=1 계산이 더 빠름: `pip install numpy`
