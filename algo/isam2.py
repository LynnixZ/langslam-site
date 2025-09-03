#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, re, math
import numpy as np
import gtsam
from gtsam import Pose2, Point2, Rot2
from gtsam.symbol_shorthand import X, L  # int key

# ---------- 解析工具 ----------
def lower_compact(s: str) -> str:
    return ''.join(c.lower() for c in s if not c.isspace())

def parse_start_pose(text: str):
    m = re.search(r"starting\s+absolute\s+position\s+is\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", text, re.I)
    if not m:
        raise ValueError("No starting position found.")
    x_in, y_in = int(m.group(1)), int(m.group(2))
    return float(x_in), float(-y_in)  # 内部 y 朝北

def parse_initial_facing(text: str):
    m = re.search(r"facing\s+(North|East|South|West)", text, re.I)
    if not m: return 0.0
    d = m.group(1).lower()
    return {"east":0.0,"north":math.pi/2,"west":math.pi,"south":-math.pi/2}[d]

def iter_steps(text: str):
    pat = re.compile(r"Step\s*\d+\s*:\s*(left|right|forward|turnaround)[^.\n]*\.\s*(.*)", re.I)
    for m in pat.finditer(text):
        yield m.group(1).lower(), m.group(2)

def iter_flags(s: str):
    pat = re.compile(r"(purple|grey|gray|red|yellow|green)\s+flag\s+at\s+coordinate\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", re.I)
    for m in pat.finditer(s):
        col = lower_compact(m.group(1))
        if col == "gray": col = "grey"
        d = int(m.group(2)); side = int(m.group(3))
        yield col, d, side

def ds_to_bearing_range(depth: int, side_right: int):
    y_left = -float(side_right)      # GTSAM 局部 y 是左为正
    x_fwd  = float(depth)
    return math.atan2(y_left, x_fwd), math.hypot(x_fwd, y_left)

# 兼容 Point2 或 numpy 数组
def point2_xy(p):
    try:
        return float(p.x()), float(p.y())
    except AttributeError:
        arr = np.asarray(p).reshape(2,)
        return float(arr[0]), float(arr[1])

# ---------- 主流程 ----------
def run(text: str):
    x0, y0 = parse_start_pose(text)
    th0    = parse_initial_facing(text)

    params = gtsam.ISAM2Params()
    params.relinearizeSkip = 1
    isam = gtsam.ISAM2(params)

    fg0 = gtsam.NonlinearFactorGraph()
    init0 = gtsam.Values()
    prior = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 5.0*math.pi/180.0]))
    fg0.add(gtsam.PriorFactorPose2(X(0), Pose2(x0, y0, th0), prior))
    init0.insert(X(0), Pose2(x0, y0, th0))
    isam.update(fg0, init0)

    huber = gtsam.noiseModel.mEstimator.Huber.Create(1.0)
    odo_base = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.6, 0.6, 0.20]))
    odo_noise = gtsam.noiseModel.Robust.Create(huber, odo_base)
    rot_base = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.15]))
    rot_noise = gtsam.noiseModel.Robust.Create(huber, rot_base)
    br_base = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.5, 1.0]))  # rad, grid
    br_noise = gtsam.noiseModel.Robust.Create(huber, br_base)

    current = isam.calculateEstimate()
    guess = current.atPose2(X(0))

    color_to_key = {}
    color_order = []
    def get_lm_key(color: str) -> int:
        if color in color_to_key: return color_to_key[color]
        k = L(len(color_order))
        color_to_key[color] = k
        color_order.append(color)
        return k

    def handle_obs_at_time(t_idx: int, obs_text: str,
                           fg: gtsam.NonlinearFactorGraph, init: gtsam.Values,
                           pose_guess: Pose2, current_vals: gtsam.Values):
        for col, d, s in iter_flags(obs_text):
            kL = get_lm_key(col)
            if not (current_vals.exists(kL) or init.exists(kL)):
                pw = pose_guess.transformFrom(Point2(float(d), float(-s)))
                init.insert(kL, pw)
            bearing, rng = ds_to_bearing_range(d, s)
            fg.add(gtsam.BearingRangeFactor2D(X(t_idx), kL, Rot2(bearing), rng, br_noise))

    m_init = re.search(r"Initial\s+Observation:.*", text, re.I)
    if m_init:
        fg = gtsam.NonlinearFactorGraph(); init = gtsam.Values()
        handle_obs_at_time(0, m_init.group(0), fg, init, guess, current)
        if fg.size() or init.size():
            isam.update(fg, init)
            current = isam.calculateEstimate()
            guess = current.atPose2(X(0))

    t = 0
    for action, obs in iter_steps(text):
        t += 1
        fg = gtsam.NonlinearFactorGraph(); init = gtsam.Values()

        if action == "forward":
            u = Pose2(1.0, 0.0, 0.0); noise = odo_noise
        elif action == "left":
            u = Pose2(0.0, 0.0, +math.pi/2); noise = rot_noise
        elif action == "right":
            u = Pose2(0.0, 0.0, -math.pi/2); noise = rot_noise
        elif action == "turnaround":
            u = Pose2(0.0, 0.0, math.pi); noise = rot_noise
        else:
            u = Pose2(0.0, 0.0, 0.0); noise = odo_noise

        fg.add(gtsam.BetweenFactorPose2(X(t-1), X(t), u, noise))
        guess = guess.compose(u)
        init.insert(X(t), guess)

        handle_obs_at_time(t, obs, fg, init, guess, current)

        isam.update(fg, init)
        current = isam.calculateEstimate()
        guess = current.atPose2(X(t))

    # 输出（把内部 y 取负，还原成“南为正”）
    parts = []
    for idx, col in enumerate(color_order):
        kL = L(idx)
        if not current.exists(kL): continue
        px, py = point2_xy(current.atPoint2(kL))
        xi = int(round(px))
        yi = int(round(-py))
        parts.append(f"a {col} flag at coordinate ({xi},{yi})")
    print("&&& " + " ".join(parts) + ". &&&")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file", nargs="?", help="path to input.txt; if omitted, read stdin")
    args = parser.parse_args()
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = sys.stdin.read()
    run(text)
