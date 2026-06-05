**TABLE I: Performance comparison under the Classic scenario. Avg HR@N is the macro-average of HR@1, HR@3, HR@5 (%). Bold denotes the best result per column among completed runs.**

| Agent | Video Games<br>(Avg HR@N) | Video Games<br>(NDCG@5) | Musical Instruments<br>(Avg HR@N) | Musical Instruments<br>(NDCG@5) | Industrial & Scientific<br>(Avg HR@N) | Industrial & Scientific<br>(NDCG@5) | Goodreads<br>(Avg HR@N) | Goodreads<br>(NDCG@5) | Yelp<br>(Avg HR@N) | Yelp<br>(NDCG@5) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CoT | 22.61 | 0.2220 | 18.16 | 0.1796 | 19.54 | 0.1956 | 39.44 | 0.3874 | 29.77 | 0.2889 |
| CoTMemory | 21.91 | 0.2159 | 19.98 | 0.1959 | 24.48 | 0.2406 | 36.53 | 0.3608 | 30.03 | 0.2918 |
| Memory | 21.91 | 0.2156 | 20.08 | 0.1957 | 23.21 | 0.2260 | 35.42 | 0.3492 | 29.19 | 0.2838 |
| DummyAgent | 33.71 | 0.3274 | 32.92 | 0.3219 | 33.73 | 0.3302 | 27.23 | 0.2682 | 35.76 | 0.3486 |
| RecHacker | 33.31 | 0.3245 | 31.92 | 0.3090 | 36.44 | 0.3550 | **47.37** | **0.4589** | 35.78 | 0.3465 |
| Baseline666 | 33.99 | 0.3314 | 32.74 | 0.3370 | 35.64 | 0.3526 | 35.13 | 0.3422 | 30.73 | 0.2964 |
| LightGCN | 55.90 | 0.5467 | 42.20 | 0.5153 | 53.13 | 0.5193 | 63.02 | 0.6234 | 46.96 | 0.4598 |
| SASRec | 68.51 | 0.6763 | 50.50 | 0.4941 | 42.20 | 0.4126 | 52.12 | 0.5142 | 70.29 | 0.6914 |
| MoERec | **70.18** | **0.6876** | **59.60** | **0.5811** | **55.11** | **0.5389** | 60.83 | 0.5995 | **70.46** | **0.6979** |


<br>

**TABLE II: Performance comparison under the Cold Start scenario. Avg HR@N is the macro-average of HR@1, HR@3, HR@5 (%). Bold denotes the best result per column among completed runs.**

| Agent | Video Games<br>(Avg HR@N) | Video Games<br>(NDCG@5) | Musical Instruments<br>(Avg HR@N) | Musical Instruments<br>(NDCG@5) | Industrial & Scientific<br>(Avg HR@N) | Industrial & Scientific<br>(NDCG@5) | Goodreads<br>(Avg HR@N) | Goodreads<br>(NDCG@5) | Yelp<br>(Avg HR@N) | Yelp<br>(NDCG@5) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CoT | 21.62 | 0.2102 | 17.35 | 0.1700 | 17.68 | 0.1736 | 43.35 | 0.4273 | 29.16 | 0.2897 |
| CoTMemory | 22.03 | 0.2167 | 19.19 | 0.1865 | 19.70 | 0.1937 | 39.51 | 0.3867 | 29.87 | 0.2899 |
| Memory | 21.72 | 0.2135 | 19.79 | 0.1930 | 19.25 | 0.1896 | 39.35 | 0.3862 | 28.19 | 0.2752 |
| DummyAgent | 32.00 | 0.3126 | 32.89 | 0.3232 | 29.57 | 0.2899 | 29.75 | 0.2892 | 33.94 | 0.3329 |
| RecHacker | 30.75 | 0.3019 | 31.22 | 0.3059 | 31.44 | 0.3069 | 45.12 | 0.4388 | 33.43 | 0.3254 |
| Baseline666 | 34.39 | 0.3352 | 34.59 | 0.3363 | 31.72 | 0.3102 | 38.46 | 0.3755 | 30.78 | 0.2986 |
| LightGCN | 56.54 | 0.5541 | 52.43 | 0.5129 | 46.84 | 0.4586 | **64.44** | **0.6374** | 43.41 | 0.4228 |
| SASRec | 65.42 | 0.6422 | 52.83 | 0.5187 | 35.56 | 0.3480 | 55.75 | 0.5512 | **70.74** | **0.6957** |
| MoERec | **69.75** | **0.6832** | **61.19** | **0.5975** | **47.41** | **0.4622** | 64.12 | 0.6332 | 69.52 | 0.6797 |
