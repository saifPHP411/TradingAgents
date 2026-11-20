# Frozen Binance configuration check

The selected September 2023 configuration is tested without fitting or changing any parameters. It scans the same 24 contracts at 50x with the 2.5 absolute relative momentum score threshold. The three test blocks cover 20 additional weekly windows in August–November 2024 and September–October 2026. The September 2023 target week is excluded. Both cost cases replay the same weeks using historical Binance trade and mark data and funding, with the same approximate contract filters and modeled fills as the original study. Some dates appeared in experiments with other signals, so this is historical confirmation for a fixed configuration.

No week ended above its starting 50 USDT in either cost case. At base costs 110 of 346 trades were profitable, the median ending balance was 0.18 USDT, and the best week ended at 33.38 USDT. Under stress 74 of 292 trades were profitable, the median was 0.14 USDT, and the best week ended at 24.31 USDT. These results provide no support for migrating execution to Binance to pursue the 1,000 USDT and 75% target with this configuration.

Run `python binance_confirmation.py` for the full weekly breakdown.
