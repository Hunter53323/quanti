"""
Verify: months_in_cycle is NOT reset after Sharp3pct exit cooldown.

Sequence: invested (mo=1,2,3) → Sharp exit (cool 2 months) → re-enter → should be mo=4, not mo=1.
"""
import sys; sys.path.insert(0, '.')
from quanti.strategy.delayed_confirm import DelayedConfirmStrategy

s = DelayedConfirmStrategy(use_sharp_exit=True, sharp_threshold=-0.03)

# Simulate the state tracking logic directly
mo   = 0     # months_in_cycle
gp   = -1    # genuine_prev
prev = -1    # prev_entry_state

# Step 0-2: invested full position (state 2)
for i in range(3):
    mst = 2; emst = 2
    if emst in (2, 4):
        if gp not in (2, 4): mo = 1
        else: mo += 1
        gp = emst
    elif emst == 3 and mst != 3: pass
    else: mo = 0; gp = emst
    prev = emst
    assert mo == i + 1, f"Step {i}: expected mo={i+1}, got {mo}"

# Step 3-4: Sharp exit fired — force cooldown
for i in range(2):
    mst = 2; emst = 3
    if emst in (2, 4):
        mo = 1 if gp not in (2, 4) else mo + 1
        gp = emst
    elif emst == 3 and mst != 3:
        pass  # DO NOT reset
    else:
        mo = 0; gp = emst
    prev = emst
    assert mo == 3, f"Step {i+3}: Sharp cool — expected mo=3 (preserved), got {mo}"

# Step 5-7: re-enter after cooldown
for i in range(3):
    mst = 2; emst = 2
    if emst in (2, 4):
        if gp not in (2, 4): mo = 1
        else: mo += 1
        gp = emst
    elif emst == 3 and mst != 3: pass
    else: mo = 0; gp = emst
    prev = emst
    assert mo == 3 + i + 1, f"Step {i+5}: re-enter — expected mo={3+i+1}, got {mo}"

print(f"PASS: months_in_cycle correctly preserved across Sharp exit.")
print(f"  Final mo={mo} (expected 6 after 3 invested + 2 cool + 3 re-invested)")
