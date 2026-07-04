# Strategy Discovery Analysis

- beam mode: `semantic_pair_split_family_semantic_dedup`
- completed cases: 775 / 775
- runtime: 37536.272479 s
- diff definition: `analytical_cc - beam_cc`, so positive means beam is faster.

## Overall

- beam better/equal/worse than analytical: 22/528/225
- diff counts: {0: 528, -33792: 209, 33792: 22, -22528: 16}
- avg analytical/ideal: 1.044518
- avg beam64/ideal: 1.056740
- avg best-of(analytical, beam64)/ideal: 1.043384
- median analytical/ideal: 1.037037
- median beam64/ideal: 1.045455
- analytical cases improved by beam64: 22

## Main Interpretation

- Beam64 found a real missing strategy in 22 cases, always in `cache=none`, usually `hot-hot + medium-medium + tiny`.
- The improvement is always one block of 33792 cc in this suite.
- Beam64 is not yet a safe universal reference: it is worse than analytical in 225 cases, mostly tiny-tail cases where it keeps a tiny-first path.
- For this result set, analytical remains the safer baseline; beam64 is currently more useful as a strategy-discovery tool.

## Best Beam Wins

- case 90: diff=33792 cc, tokens=[16, 15, 12, 11, 1, 1], cache=none, A/I=1.1429, B/I=1.0714, path=SPLIT(E0:8,8) -> SPLIT(E1:7,8) -> SINGLE-C2(E4) -> SINGLE-C3(E5) -> PAIR(2+3)
- case 205: diff=33792 cc, tokens=[24, 23, 13, 12, 1], cache=none, A/I=1.0959, B/I=1.0411, path=PAIR(2+3) -> SINGLE-C3(E4) -> PF-C3(E0,A(M8,bw32)) -> PAIR(0+1)
- case 210: diff=33792 cc, tokens=[24, 23, 13, 12, 2], cache=none, A/I=1.0811, B/I=1.0270, path=PAIR(2+3) -> SINGLE-C3(E4) -> PF-C3(E0,A(M8,bw32)) -> PAIR(0+1)
- case 220: diff=33792 cc, tokens=[24, 22, 13, 11, 1], cache=none, A/I=1.1268, B/I=1.0704, path=PAIR(2+3) -> SINGLE-C3(E4) -> PF-C3(E0,A(M8,bw32)) -> PAIR(0+1)
- case 230: diff=33792 cc, tokens=[24, 23, 15, 14, 1], cache=none, A/I=1.0909, B/I=1.0390, path=PAIR(2+3) -> SINGLE-C3(E4) -> PF-C3(E0,A(M8,bw32)) -> PAIR(0+1)
- case 235: diff=33792 cc, tokens=[24, 23, 15, 14, 2], cache=none, A/I=1.0769, B/I=1.0256, path=PAIR(2+3) -> SINGLE-C3(E4) -> PF-C3(E0,A(M8,bw32)) -> PAIR(0+1)
- case 245: diff=33792 cc, tokens=[24, 22, 15, 13, 1], cache=none, A/I=1.1200, B/I=1.0667, path=PAIR(2+3) -> SINGLE-C3(E4) -> PF-C3(E0,A(M8,bw32)) -> PAIR(0+1)
- case 270: diff=33792 cc, tokens=[24, 22, 16, 14, 1], cache=none, A/I=1.0909, B/I=1.0390, path=PAIR(2+3) -> SINGLE-C3(E4) -> PF-C3(E0,A(M8,bw32)) -> PAIR(0+1)
- case 295: diff=33792 cc, tokens=[24, 22, 17, 15, 1], cache=none, A/I=1.1139, B/I=1.0633, path=SINGLE-C2(E4) -> PF-C2(E0,A(M8,bw32)) -> WAIT-PAIR(2+3) -> SINGLE-C3(E0) -> SINGLE-C2(E1)
- case 405: diff=33792 cc, tokens=[32, 31, 17, 16, 1], cache=none, A/I=1.0722, B/I=1.0309, path=PAIR(2+3) -> SINGLE-C3(E4) -> PF-C3(E0,A(M8,bw32)) -> PAIR(0+1)
- case 410: diff=33792 cc, tokens=[32, 31, 17, 16, 2], cache=none, A/I=1.0612, B/I=1.0204, path=PAIR(2+3) -> SINGLE-C3(E4) -> PF-C3(E0,A(M8,bw32)) -> PAIR(0+1)
- case 420: diff=33792 cc, tokens=[32, 30, 17, 15, 1], cache=none, A/I=1.0947, B/I=1.0526, path=PAIR(2+3) -> SINGLE-C3(E4) -> PF-C3(E0,A(M8,bw32)) -> PAIR(0+1)

## Worst Beam Losses

- case 5: diff=-33792 cc, tokens=[16, 15, 7, 6, 1], cache=none, A/I=1.0667, B/I=1.1556, path=SINGLE-C2(E4) -> PF-C2(E0,A(M8,bw32)) -> WAIT-PAIR(2+3) -> WAIT-PAIR(0+1)
- case 6: diff=-33792 cc, tokens=[16, 15, 7, 6, 1], cache=hot_pair, A/I=1.0667, B/I=1.1556, path=SINGLE-C2(E4) -> PF-C2(E0,A(M8,bw32)) -> WAIT-PAIR(2+3) -> WAIT-PAIR(0+1)
- case 7: diff=-33792 cc, tokens=[16, 15, 7, 6, 1], cache=medium_pair, A/I=1.0667, B/I=1.1556, path=SINGLE-C2(E4) -> PF-C2(E0,A(M8,bw32)) -> WAIT-PAIR(2+3) -> WAIT-PAIR(0+1)
- case 8: diff=-33792 cc, tokens=[16, 15, 7, 6, 1], cache=hot0, A/I=1.0667, B/I=1.1556, path=SINGLE-C2(E4) -> PF-C2(E0,A(M8,bw32)) -> WAIT-PAIR(2+3) -> WAIT-PAIR(0+1)
- case 9: diff=-33792 cc, tokens=[16, 15, 7, 6, 1], cache=medium0, A/I=1.0667, B/I=1.1556, path=SINGLE-C2(E4) -> PF-C2(E0,A(M8,bw32)) -> WAIT-PAIR(2+3) -> WAIT-PAIR(0+1)
- case 10: diff=-33792 cc, tokens=[16, 15, 7, 6, 2], cache=none, A/I=1.0435, B/I=1.1304, path=SINGLE-C2(E4) -> PF-C2(E0,A(M8,bw32)) -> WAIT-PAIR(2+3) -> WAIT-PAIR(0+1)
- case 11: diff=-33792 cc, tokens=[16, 15, 7, 6, 2], cache=hot_pair, A/I=1.0435, B/I=1.1304, path=SINGLE-C2(E4) -> PF-C2(E0,A(M8,bw32)) -> WAIT-PAIR(2+3) -> WAIT-PAIR(0+1)
- case 12: diff=-33792 cc, tokens=[16, 15, 7, 6, 2], cache=medium_pair, A/I=1.0435, B/I=1.1304, path=SINGLE-C2(E4) -> PF-C2(E0,A(M8,bw32)) -> WAIT-PAIR(2+3) -> WAIT-PAIR(0+1)
- case 13: diff=-33792 cc, tokens=[16, 15, 7, 6, 2], cache=hot0, A/I=1.0435, B/I=1.1304, path=SINGLE-C2(E4) -> PF-C2(E0,A(M8,bw32)) -> WAIT-PAIR(2+3) -> WAIT-PAIR(0+1)
- case 14: diff=-33792 cc, tokens=[16, 15, 7, 6, 2], cache=medium0, A/I=1.0435, B/I=1.1304, path=SINGLE-C2(E4) -> PF-C2(E0,A(M8,bw32)) -> WAIT-PAIR(2+3) -> WAIT-PAIR(0+1)
- case 15: diff=-33792 cc, tokens=[16, 15, 7, 6, 1, 1], cache=none, A/I=1.1304, B/I=1.2174, path=PAIR(4+5) -> SPLIT(E0:8,8) -> SPLIT(E1:7,8) -> PAIR(2+3)
- case 16: diff=-33792 cc, tokens=[16, 15, 7, 6, 1, 1], cache=hot_pair, A/I=1.1304, B/I=1.2174, path=PAIR(4+5) -> SPLIT(E0:8,8) -> SPLIT(E1:7,8) -> PAIR(2+3)

### By Cache Mode

| group | n | beam win/tie/loss | avg analytical/ideal | avg beam/ideal | avg diff cc | min diff | max diff |
|---|---:|---:|---:|---:|---:|---:|---:|
| hot0 | 155 | 0/104/51 | 1.042200 | 1.057259 | -10828.0 | -33792 | 0 |
| hot_pair | 155 | 0/103/52 | 1.042200 | 1.057720 | -11046.0 | -33792 | 0 |
| medium0 | 155 | 0/103/52 | 1.042200 | 1.057720 | -11046.0 | -33792 | 0 |
| medium_pair | 155 | 0/114/41 | 1.042200 | 1.053741 | -8938.5 | -33792 | 0 |
| none | 155 | 22/104/29 | 1.053790 | 1.057259 | -1235.4 | -33792 | 33792 |

### By Variant

| group | n | beam win/tie/loss | avg analytical/ideal | avg beam/ideal | avg diff cc | min diff | max diff |
|---|---:|---:|---:|---:|---:|---:|---:|
| asym_tiny | 155 | 11/124/20 | 1.049950 | 1.052877 | -799.4 | -33792 | 33792 |
| no_tiny | 155 | 0/155/0 | 1.035521 | 1.035521 | 0.0 | 0 | 0 |
| tiny1 | 155 | 5/118/32 | 1.044343 | 1.054128 | -5886.3 | -33792 | 33792 |
| tiny2 | 155 | 5/118/32 | 1.032948 | 1.042575 | -5886.3 | -33792 | 33792 |
| two_tiny | 155 | 1/13/141 | 1.059828 | 1.098596 | -30521.8 | -33792 | 33792 |

### By Active Expert Count

| group | n | beam win/tie/loss | avg analytical/ideal | avg beam/ideal | avg diff cc | min diff | max diff |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 155 | 0/155/0 | 1.035521 | 1.035521 | 0.0 | 0 | 0 |
| 5 | 465 | 21/360/84 | 1.042414 | 1.049860 | -4190.7 | -33792 | 33792 |
| 6 | 155 | 1/13/141 | 1.059828 | 1.098596 | -30521.8 | -33792 | 33792 |
