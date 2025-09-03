import random
import re

def _gauss_skip_count(mu=1.0, sigma=0.6, lo=1, hi=2):
    while True:
        k = int(round(random.gauss(mu, sigma)))
        if lo <= k <= hi:
            return k

def _sample_offset_int(sigma=1.0, kmax=3):
    while True:
        k = int(round(random.gauss(0.0, sigma)))
        if k != 0:
            return max(-kmax, min(kmax, k))

def _noisify_desc_per_flag(desc, per_flag_prob, depth_range=(0,4), side_range=(-2,2), sigma=1.0):
    changes = []
    pattern = re.compile(r"\(\s*([-+]?\d+)\s*,\s*([-+]?\d+)\s*\)")
    def repl(m):
        d0 = int(m.group(1)); s0 = int(m.group(2))
        d, s = d0, s0
        import random as _rnd
        if _rnd.random() < per_flag_prob:
            d = d0 + _sample_offset_int(sigma=sigma, kmax=3)
            d = max(depth_range[0], min(depth_range[1], d))
        if _rnd.random() < per_flag_prob:
            s = s0 + _sample_offset_int(sigma=sigma, kmax=3)
            s = max(side_range[0], min(side_range[1], s))
        if (d, s) != (d0, s0):
            changes.append(((d0, s0), (d, s)))
        return f"({d}, {s})"
    new_desc = pattern.sub(repl, desc)
    return new_desc, changes
