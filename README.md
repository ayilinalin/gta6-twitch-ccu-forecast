[![make.fun](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ayilinalin/gta6-twitch-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ayilinalin/gta6-twitch-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch CCU forecaster

Entry for the [make.fun competition](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
**gta6-twitch-ccu-forecast-01**: forecast, hour by hour, how many people watch
Grand Theft Auto VI live on Twitch over the 24 UTC hours of its release day
(2026-11-19), from everything recorded up to the end of hour 1 — five quantiles
(5%, 25%, 50%, 75%, 95%) per hour, hours 2–24.

`main.ts` (Node 22, standard library only, no network) reads every `*.in` case in
`$MAKEFUN_INPUT_DIR` (else `/input`) and writes one `*.out` forecast per case to
`$MAKEFUN_OUTPUT_DIR` (else `/output`). Deterministic, ~0.5 s per case.

## Method

The program fits its model at runtime from the case file's own comparables
(38–42 past big launches, hourly SullyGnome counts):

1. Each comparable's release-day curve is expressed as log-ratios to its hour-1
   print — hour 1 (00:00–01:00 UTC) is a local trough between the New
   Zealand/Australia evening wave and the UK/US waves, so the curve's *shape*
   (dip, then rise to an evening peak) is what must be learned, not the level.
2. Only "fresh-category" launches are kept (median hourly viewers in the 7 days
   before release < 1000); live-service / early-access games whose category
   already had viewers follow a different shape and poison the fit.
3. Four diverse predictors, ensembled with equal weight in log space:
   - per-hour linear OLS of log-ratio on log(hour-1 viewers), normal residual spread;
   - per-hour quadratic OLS on log(hour-1 viewers), normal residual spread;
   - 8 nearest neighbours by hour-1 size, empirical ratio quantiles;
   - 8 nearest neighbours by pre-midnight profile distance (the NZ/Australia
     unlock wave observed in the 13 hours before midnight fingerprints the
     launch's shape), empirical ratio quantiles.

## Local backtest

Scorer (mean pinball loss on the log scale) verified against the worked example
in the kit's RULES (loss 0.112545). Practice answers taken from SullyGnome's
public hourly data; hour-1 prints match the case files exactly.

| case | loss | typical miss | 90% coverage |
|------|------|--------------|--------------|
| p00 Marvel's Wolverine | 0.215 | 38.5% | 1.00 |
| p01 Red Dead Redemption 2 | 0.194 | 35.3% | 1.00 |
| p02 God of War Ragnarök | 0.169 | 27.6% | 1.00 |
| **mean (p00–p02)** | **0.193** | | |
| house baseline (starter) | 0.415 | | |

(p03 Marvel's Spider-Man 2 is not scored locally: the source returns no hourly
series for its release month.)

## Run locally

```sh
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out node main.ts
```
