# rbpf_dlg_solver_dynamic_colors.py
# RBPF + Discrete Landmark Grid, colors parsed from the log (no hardcoding)

import re
import random
from collections import defaultdict

# ----------------------------- constants ---------------------------------

DIRS = ["East", "South", "West", "North"]  # 0,1,2,3
FWD = {0:(1,0), 1:(0,1), 2:(-1,0), 3:(0,-1)}
RIGHT = {0:(0,1), 1:(-1,0), 2:(0,-1), 3:(1,0)}
LEFT_TURN = {0:3, 1:0, 2:1, 3:2}
RIGHT_TURN = {0:1, 1:2, 2:3, 3:0}
TURNAROUND = {0:2, 1:3, 2:0, 3:1}

GRID_W = 10
GRID_H = 10
ALL_CELLS = [(x,y) for x in range(GRID_W) for y in range(GRID_H)]

# ----------------------------- parsing ---------------------------------

def extract_colors(text: str):
    # find any "<color> flag at coordinate (...)" and collect colors
    cols = set(m.group(1).lower() for m in re.finditer(
        r"([A-Za-z]+)\s+flag\s+at\s+coordinate\s*\(\s*[-\d]+\s*,\s*[-\d]+\s*\)", text))
    return sorted(cols)

def parse_start_pos(text):
    m = re.search(r"starting absolute position is\s*\(([-\d]+)\s*,\s*([-\d]+)\)", text, re.I)
    if not m:
        raise ValueError("Cannot find starting absolute position.")
    return (int(m.group(1)), int(m.group(2)))

def parse_initial_dir_and_obs(text):
    mdir = re.search(r"Initial Observation:\s*You['’]re facing\s+(East|South|West|North)", text, re.I)
    if not mdir:
        raise ValueError("Cannot find initial facing direction.")
    dir_str = mdir.group(1).capitalize()
    d0 = DIRS.index(dir_str)

    init_end = re.search(r"\bStep\s*1\s*:", text)
    boundary_idx = init_end.start() if init_end else len(text)
    init_block = text[:boundary_idx]

    obs = []
    # generic color capture; ignores walls since it requires the word 'flag'
    for color, dx, dy in re.findall(r"([A-Za-z]+)\s+flag at coordinate\s*\(([-\d]+)\s*,\s*([-\d]+)\)", init_block, re.I):
        obs.append( (color.lower(), (int(dx), int(dy))) )
    return d0, obs

def parse_steps(text):
    steps = []
    parts = re.split(r"\bStep\s*(\d+)\s*:\s*", text)
    for i in range(1, len(parts), 2):
        step_id = int(parts[i])
        rest = parts[i+1]
        mact = re.match(r"\s*([A-Za-z]+)\s*\.", rest)
        action = mact.group(1).lower() if mact else "noop"
        obs = []
        for color, dx, dy in re.findall(r"([A-Za-z]+)\s+flag at coordinate\s*\(([-\d]+)\s*,\s*([-\d]+)\)", rest, re.I):
            obs.append( (color.lower(), (int(dx), int(dy))) )
        steps.append({"id": step_id, "action": action, "flags": obs})
    return steps

# ----------------------------- geometry --------------------------------

def in_bounds(x, y):
    return 0 <= x < GRID_W and 0 <= y < GRID_H

def step_forward(x,y,dir_idx,steps=1):
    fx, fy = FWD[dir_idx]
    nx, ny = x, y
    for _ in range(steps):
        tx, ty = nx + fx, ny + fy
        if not in_bounds(tx, ty):
            break
        nx, ny = tx, ty
    return nx, ny, dir_idx

def apply_action_noisy(x,y,dir_idx,action, rng,
                       p_jump=0.02, p_noop=0.10, p_other_turn=0.05):
    # small-prob jump handles missing segments / teleport
    if rng.random() < p_jump:
        nx, ny = rng.randrange(GRID_W), rng.randrange(GRID_H)
        nd = dir_idx if rng.random() < 0.8 else rng.randrange(4)
        return nx, ny, nd
    if action == "forward":
        r = rng.random()
        steps = 0 if r < 0.10 else (1 if r < 0.90 else 2)
        return step_forward(x,y,dir_idx,steps)
    elif action == "left":
        r = rng.random()
        if r < (1-p_noop-p_other_turn): nd = LEFT_TURN[dir_idx]
        elif r < (1-p_other_turn):      nd = dir_idx
        else:                           nd = RIGHT_TURN[dir_idx] if rng.random()<0.7 else TURNAROUND[dir_idx]
        return x,y,nd
    elif action == "right":
        r = rng.random()
        if r < (1-p_noop-p_other_turn): nd = RIGHT_TURN[dir_idx]
        elif r < (1-p_other_turn):      nd = dir_idx
        else:                           nd = LEFT_TURN[dir_idx] if rng.random()<0.7 else TURNAROUND[dir_idx]
        return x,y,nd
    elif action == "turnaround":
        r = rng.random()
        if r < (1-p_noop-p_other_turn): nd = TURNAROUND[dir_idx]
        elif r < (1-p_other_turn):      nd = dir_idx
        else:                           nd = LEFT_TURN[dir_idx] if rng.random()<0.5 else RIGHT_TURN[dir_idx]
        return x,y,nd
    else:
        return (x,y,dir_idx) if rng.random() < 0.9 else apply_action_noisy(x,y,dir_idx,"forward", rng)

def world_to_local(cx, cy, x, y, dir_idx):
    dx, dy = cx - x, cy - y
    fx, fy = FWD[dir_idx]
    rx, ry = RIGHT[dir_idx]
    depth = dx*fx + dy*fy
    side  = dx*rx + dy*ry
    return depth, side

def in_fov(depth, side):
    return (0 <= depth <= 4) and (-2 <= side <= 2)

# ------------------------- observation model ---------------------------

def phi_distance(dist):
    # Chebyshev distance on the 5x5 view grid
    if dist == 0: return 1.0
    if dist == 1: return 0.5
    if dist == 2: return 0.2
    return 0.02

def color_likelihood_table(pose, obs_list, PD=0.9, clutter=1e-3):
    x,y,d = pose
    L = {}
    for (cx,cy) in ALL_CELLS:
        depth, side = world_to_local(cx,cy,x,y,d)
        if obs_list:
            best = 0.0
            for (od,os) in obs_list:
                if in_fov(depth, side):
                    dist = max(abs(depth-od), abs(side-os))
                    best = max(best, phi_distance(dist))
                else:
                    best = max(best, 0.02)
            L[(cx,cy)] = PD*best + clutter
        else:
            if in_fov(depth, side):
                L[(cx,cy)] = (1.0 - PD) + clutter  # missed detection
            else:
                L[(cx,cy)] = 1.0                   # neutral outside FoV
    return L

# ------------------------- particle and beliefs ------------------------

class Particle:
    def __init__(self, x, y, d, colors, rng):
        self.x = x; self.y = y; self.d = d
        self.w = 1.0
        u = 1.0 / (GRID_W * GRID_H)
        self.pi = {c: {cell: u for cell in ALL_CELLS} for c in colors}
        self.rng = rng
    def pose(self):
        return (self.x, self.y, self.d)

def normalize_weights(particles):
    s = sum(max(p.w,0.0) for p in particles)
    if s <= 0:
        for p in particles: p.w = 1.0/len(particles)
    else:
        for p in particles: p.w /= s

def effective_N(particles):
    s2 = sum(p.w*p.w for p in particles)
    return 0 if s2 == 0 else 1.0/s2

def systematic_resample(particles, rng):
    N = len(particles)
    positions = [(rng.random() + i)/N for i in range(N)]
    cumulative = []
    csum = 0.0
    for p in particles:
        csum += p.w
        cumulative.append(csum)
    i = 0
    new_particles = []
    for u in positions:
        while u > cumulative[i]:
            i += 1
        src = particles[i]
        q = Particle(src.x, src.y, src.d, list(src.pi.keys()), src.rng)
        q.pi = {c: dict(src.pi[c]) for c in src.pi}
        q.w = 1.0/N
        new_particles.append(q)
    return new_particles

# ---------------------------- solver core ------------------------------

def rbpf(problem_text, rng_seed=0, num_particles=200, resample_thresh=0.5):
    rng = random.Random(rng_seed)
    colors = extract_colors(problem_text)
    if not colors:
        raise ValueError("No flags found in the log.")

    start_x, start_y = parse_start_pos(problem_text)
    d0, init_obs = parse_initial_dir_and_obs(problem_text)
    steps = parse_steps(problem_text)

    particles = [Particle(start_x, start_y, d0, colors, rng) for _ in range(num_particles)]
    normalize_weights(particles)

    def dir_name(idx): return DIRS[idx]

    lines = []
    lines.append(f"Start at pose {(start_x, start_y)} facing {dir_name(d0)}; colors={colors}; initial obs={init_obs}")

    # initial measurement update
    obs_by_color = defaultdict(list)
    for (c,(od,os)) in init_obs:
        obs_by_color[c].append((od,os))

    for p in particles:
        w_like = 1.0
        for color in colors:
            L = color_likelihood_table(p.pose(), obs_by_color.get(color, []))
            pi_prev = p.pi[color]
            marginal = sum(pi_prev[cell]*L[cell] for cell in ALL_CELLS)
            w_like *= marginal
        p.w *= w_like
    normalize_weights(particles)

    for p in particles:
        for color in colors:
            L = color_likelihood_table(p.pose(), obs_by_color.get(color, []))
            pi_prev = p.pi[color]
            for cell in ALL_CELLS:
                pi_prev[cell] *= L[cell]
            s = sum(pi_prev.values())
            if s > 0:
                for cell in ALL_CELLS:
                    pi_prev[cell] /= s

    # iterate steps
    for st in steps:
        action = st["action"]
        flags_obs = st["flags"]
        obs_by_color = defaultdict(list)
        for (c,(od,os)) in flags_obs:
            obs_by_color[c].append((od,os))

        # propagate
        for p in particles:
            nx, ny, nd = apply_action_noisy(p.x, p.y, p.d, action, p.rng)
            p.x, p.y, p.d = nx, ny, nd

        # weight update
        for p in particles:
            w_like = 1.0
            for color in colors:
                L = color_likelihood_table(p.pose(), obs_by_color.get(color, []))
                pi_prev = p.pi[color]
                marginal = sum(pi_prev[cell]*L[cell] for cell in ALL_CELLS)
                w_like *= marginal
            p.w *= w_like
        normalize_weights(particles)

        # resample
        if effective_N(particles) < resample_thresh * len(particles):
            particles = systematic_resample(particles, rng)

        # belief update
        for p in particles:
            for color in colors:
                L = color_likelihood_table(p.pose(), obs_by_color.get(color, []))
                pi_prev = p.pi[color]
                for cell in ALL_CELLS:
                    pi_prev[cell] *= L[cell]
                s = sum(pi_prev.values())
                if s > 0:
                    for cell in ALL_CELLS:
                        pi_prev[cell] /= s

        # logging
        best = max(particles, key=lambda q: q.w)
        peaks = {color: max(best.pi[color].items(), key=lambda kv: kv[1])[0] for color in colors}
        lines.append(f"Step{st['id']} action={action}; est_pose={(best.x, best.y, dir_name(best.d))}; seen={flags_obs}; peaks={peaks}")

    # final
    best = max(particles, key=lambda q: q.w)
    final = {color: max(best.pi[color].items(), key=lambda kv: kv[1])[0] for color in colors}
    # deterministic order in summary
    items = [f"a {c} flag at coordinate {final[c]}" for c in colors]
    summary = "&&&\n" + " ".join(items) + "\n&&&"
    return "\n".join(lines + [summary])


if __name__ == "__main__":
    PROBLEM_TEXT = """<PASTE THE FULL PROMPT TEXT HERE>"""
    if "<PASTE" in PROBLEM_TEXT:
        print("Please paste the full problem text into PROBLEM_TEXT.")
    else:
        print(solve(PROBLEM_TEXT, rng_seed=7, num_particles=200))
