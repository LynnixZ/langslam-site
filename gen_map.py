# filename: scripts/gen_map_gifs.py

import os
import argparse
import random
import gymnasium as gym

# 你自己项目里的注册与工具
import create_env  # 只为确保自定义 env 已注册
from utils import build_script, run_scripted_playthrough

def make_one_gif(
    idx: int,
    out_dir: str,
    length: int,
    agent_view_size: int,
    rng_seed: int | None,
    widen_params: dict,
    post_params: dict,
):
    # 随机起点与朝向（在内层）
    rng = random.Random(None if rng_seed is None else rng_seed + idx)
    start_x = rng.randint(1, length - 2)
    start_y = rng.randint(1, length - 2)
    agent_start_pos = (start_x, start_y)
    agent_start_dir = rng.randint(0, 3)

    # 生成 2D 探索脚本（只用来“开路”）
    my_script = build_script(
        env_name="MiniGrid-CustomEmpty-5x5-v0",
        length=length,
        agent_view_size=agent_view_size,
        agent_start_pos=agent_start_pos,
        agent_start_dir=agent_start_dir,
        easy=False,
        oneD=False,
    )

    env = gym.make(
        "MiniGrid-CarvedPathRoom-v0",
        length=length,
        agent_view_size=agent_view_size,
        carve_script=my_script,
        start_pos=agent_start_pos,
        start_dir=agent_start_dir,
        render_mode="rgb_array",
        widen_mode="segments",
        seg_prob_left=0.3,
        seg_prob_right=0.3,
        seg_band=1,
        seg_end_trim=0,
        seg_skip_prob=0,
        post_fill_enable=True,
        post_fill_min_open_neighbors=3,
    )

    os.makedirs(out_dir, exist_ok=True)
    obs, info = env.reset(seed=None if rng_seed is None else (rng_seed + idx))

    gif_path = os.path.join(out_dir, f"map_{idx:03d}.gif")

    # 不引入观测故障，这里只为看地图形状
    _start_pos, _obs = run_scripted_playthrough(
        env,
        my_script,
        picfilename=gif_path,
        dir=True,
        oneDim=False,
        failure_prob=0.0,
        blackout=False,
        stale_obs_prob=0.0,
        debug=False,
    )
    env.close()
    return gif_path


def main():
    parser = argparse.ArgumentParser(description="Batch-generate CarvedPathRoom map GIFs for visual inspection.")
    parser.add_argument("--out_dir", type=str, default="map_gifs")
    parser.add_argument("--num", type=int, default=20, help="How many GIFs to make")
    parser.add_argument("--length", type=int, default=15)
    parser.add_argument("--agent_view_size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)

    # 常用的“更平滑”默认值。需要更尖锐就把 smooth_win 和 corner_shrink 调小，noise_amp 调大。
    parser.add_argument("--widen_base_width", type=float, default=0.3)
    parser.add_argument("--widen_noise_amp", type=float, default=0.45)
    parser.add_argument("--widen_band_width", type=int,   default=1)
    parser.add_argument("--widen_smooth_win", type=int,   default=15)
    parser.add_argument("--widen_corner_shrink", type=float, default=0.5)
    parser.add_argument("--widen_prob_two_sided", type=float, default=0.10)
    parser.add_argument("--widen_prob_left_when_one", type=float, default=0.7)
    parser.add_argument("--widen_target_ratio", type=float, default=0.4)

    parser.add_argument("--post_fill_min_open_neighbors", type=int, default=4)
    parser.add_argument("--post_fill_keep_ratio", type=float, default=None)

    args = parser.parse_args()

    widen_params = dict(
        widen_enable=True,
        widen_base_width=args.widen_base_width,
        widen_noise_amp=args.widen_noise_amp,
        widen_band_width=args.widen_band_width,
        widen_smooth_win=args.widen_smooth_win,
        widen_corner_shrink=args.widen_corner_shrink,
        widen_prob_two_sided=args.widen_prob_two_sided,
        widen_prob_left_when_one=args.widen_prob_left_when_one,
        widen_target_ratio=args.widen_target_ratio,
    )
    post_params = dict(
        post_fill_enable=True,
        post_fill_min_open_neighbors=args.post_fill_min_open_neighbors,
        post_fill_keep_ratio=args.post_fill_keep_ratio,
    )

    made = []
    for i in range(args.num):
        path = make_one_gif(
            idx=i,
            out_dir=args.out_dir,
            length=args.length,
            agent_view_size=args.agent_view_size,
            rng_seed=args.seed,
            widen_params=widen_params,
            post_params=post_params,
        )
        made.append(path)

    print(f"Done. Wrote {len(made)} GIFs to: {args.out_dir}")

if __name__ == "__main__":
    main()
