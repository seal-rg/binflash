"""v096: maxnreg=128 in autotune — NEUTRAL/slightly worse.

Added maxnreg=128 option to autotune configs (10 → 20 configs).
Theory: limiting registers increases occupancy for memory-bound D=64.
Reality: within noise for D=128, slightly worse at D=64 (2.31 vs 2.12ms)
due to doubled autotune config count introducing more noise.

REVERTED — extra configs add noise without clear benefit.
"""
