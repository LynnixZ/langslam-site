# filename: run_rbpf_eval.py

import gymnasium as gym
import logging
import time
import os
import json
import random
from typing import Any, Dict, List

from utils.vis import save_empty_prediction_image
from utils import (
    build_script, run_scripted_playthrough, setup_logging,
    generate_ground_truth, parse_llm_output, compare_maps,
    generate_prompt, map_to_records
)
import create_env  # noqa: F401  # keep env registration side-effects
import argparse
from algo import rbpf_carved


def run_single_test_episode(episode_id: int, args) -> Dict[str, Any]:
    """运行单次 RBPF 评测，返回一个可写入 JSON 的结果字典。"""
    logging.info(f"--- Starting RBPF Episode #{episode_id} ---")

    # 1) 随机起点与朝向（保证在内层）
    start_x = random.randint(1, args.length - 2)
    start_y = random.randint(1, args.length - 2)
    agent_start_pos = (start_x, start_y)
    agent_start_dir = random.randint(0, 3)

    # 2) 生成“开路脚本”（2D 探索器）
    script_env_name = "MiniGrid-CustomEmpty-5x5-v0"
    my_script = build_script(
        script_env_name,
        length=args.length,
        agent_view_size=args.agent_view_size,
        agent_start_pos=agent_start_pos,
        agent_start_dir=agent_start_dir,
        easy=False,
        oneD=False,
    )

    # 3) 用脚本刻出通道并创建评测环境
    env_name_eval = "MiniGrid-CarvedPathRoom-v0"
    my_env = gym.make(
        env_name_eval,
        length=args.length,
        agent_view_size=args.agent_view_size,
        carve_script=my_script,
        start_pos=agent_start_pos,
        start_dir=agent_start_dir,
        render_mode="rgb_array",
        # —— 连续墙面（segments）+ 末端不过度裁剪、不过度跳段，尽量平滑
        widen_mode="segments",
        seg_prob_left=0.3,
        seg_prob_right=0.3,
        seg_band=1,
        seg_end_trim=0,
        seg_skip_prob=0,
        # 迭代填充
        post_fill_enable=True,
        post_fill_min_open_neighbors=3,
        post_fill_keep_ratio=None,
        rng_seed=args.random_seed if args.random_seed is not None else 1234,
    )

    # 4) 运行脚本并生成 GIF 与观测日志
    obs, info = my_env.reset()
    os.makedirs(args.log_dir, exist_ok=True)
    gif_filename = os.path.join(args.log_dir, f"episode_{episode_id}_trace.gif")

    start_pos, obsstring = run_scripted_playthrough(
        my_env,
        my_script,
        picfilename=gif_filename,
        dir=args.dir,
        oneDim=False,
        failure_prob=args.failure_prob,
        blackout=args.blackout,
        stale_obs_prob=args.stale_obs_prob,
        list_unseen=args.list_unseen,
        list_empty=args.list_empty,
    )

    # 5) GT 与 Prompt
    ground_truth_map = generate_ground_truth(my_env, env_name=env_name_eval)
    my_env.close()

    prompt = generate_prompt(
        obsstring,
        start_pos,
        oneD=False,
        num_objects=args.num_objects,
        length=args.length,
        agent_view_size=args.agent_view_size,
        tools=None,                 # 仅 RBPF，不走 LLM 工具
        hint=args.hint,
        failure_prob=args.failure_prob,
        blackout=args.blackout,
        stale_obs_prob=args.stale_obs_prob,
        env_names=env_name_eval,
        list_unseen=args.list_unseen,
        list_empty=args.list_empty,
        unseen_example=args.unseen_example,
    )
    logging.info(f"Prompt for episode #{episode_id} prepared.")

    # 6) RBPF 推理（唯一评测对象）
    logging.info(f"Running RBPF for episode #{episode_id} ...")
    rbpf_text = rbpf_carved(prompt, 
                     num_particles=args.rbpf_num_particles, failure_prob=args.failure_prob, stale_obs_prob=args.stale_obs_prob)
    rbpf_map = parse_llm_output(rbpf_text, env_name=env_name_eval)
    result_rbpf = compare_maps(ground_truth_map, rbpf_map, env_name=env_name_eval)

    logging.info(f"RBPF raw text:\n{rbpf_text}")
    logging.info(f"RBPF Result: {'PERFECT MATCH' if result_rbpf.get('is_perfect_match') else 'IMPERFECT'}")
    logging.info(result_rbpf.get('details', ''))

    # 7) 可视化 RBPF 预测（灰=未知，黑=预测 empty）
    rbpf_img_path = os.path.join(args.log_dir, f"episode_{episode_id}_rbpf_pred.png")
    save_empty_prediction_image(
        llm_map=rbpf_map,           # 该工具函数同样适用于 RBPF 输出
        width=args.length,
        height=args.length,
        out_path=rbpf_img_path,
        cell=14,
        gray_unknown=140,
        gray_border=100,
        black_empty=0,
    )

    # 8) 组织 JSON 结果块（把 prompt 也写进去）
    episode_record: Dict[str, Any] = {
        "episode_id": episode_id,
        "start_pos": list(agent_start_pos),
        "start_dir": agent_start_dir,
        "prompt": prompt,
        "ground_truth": map_to_records(ground_truth_map),
        "rbpf": {
            "raw_text": rbpf_text,
            "rbpf_result": map_to_records(rbpf_map),
            "metrics": result_rbpf,
            "pred_image": rbpf_img_path,
            "trace_gif": gif_filename,
        },
    }

    logging.info(f"--- Finished RBPF Episode #{episode_id} ---\n")
    return episode_record


def main(args):
    setup_logging(args.log_dir)  # 仍然写 evaluation_log.txt

    all_episodes: List[Dict[str, Any]] = []
    metric_sums: Dict[str, float] = {}   # 动态累计数值型指标
    episodes_count = 0

    for i in range(args.num_episodes):
        rec = run_single_test_episode(i, args)
        all_episodes.append(rec)
        episodes_count += 1

        # 动态累计 RBPF 数值指标（跳过布尔/字符串）
        metrics = rec["rbpf"]["metrics"]
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                metric_sums[k] = metric_sums.get(k, 0.0) + float(v)

        if i < args.num_episodes - 1:
            time.sleep(5)

    # 生成 JSON 汇总（对所有数值型指标求平均）
    summary: Dict[str, Any] = {"total_episodes": episodes_count}
    for k, s in metric_sums.items():
        summary[f"avg_{k}"] = s / episodes_count if episodes_count else 0.0

    payload = {
        "config": {
            "log_dir": args.log_dir,
            "env_name": "MiniGrid-CarvedPathRoom-v0",
            "num_episodes": args.num_episodes,
            "length": args.length,
            "num_objects": args.num_objects,
            "agent_view_size": args.agent_view_size,
            "failure_prob": args.failure_prob,
            "blackout": args.blackout,
            "stale_obs_prob": args.stale_obs_prob,
            "random_seed": args.random_seed,
            "rbpf_num_particles": args.rbpf_num_particles,
            "list_unseen": args.list_unseen,
            "list_empty": args.list_empty,
            "unseen_example": args.unseen_example,
        },
        "episodes": all_episodes,
        "summary": summary,
    }

    os.makedirs(args.log_dir, exist_ok=True)
    json_path = os.path.join(args.log_dir, "evaluation_summary_rbpf.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logging.info("=" * 50)
    logging.info("--- RBPF EVALUATION SUMMARY (JSON written) ---")
    logging.info(summary)
    logging.info(f"Saved JSON to '{json_path}'")
    logging.info("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run RBPF-only evaluation.")
    parser.add_argument("--log_dir", type=str, default="evaluation_results",
                        help="Directory to save logs and JSON/GIFs/PNGs.")
    parser.add_argument("--env_name", type=str, default="MiniGrid-MyCorridor-v0",
                        help="(Unused) Compatibility flag.")
    parser.add_argument("--num_episodes", type=int, default=5,
                        help="Number of random episodes to test.")
    parser.add_argument("--length", type=int, default=10,
                        help="Length of the square room side.")
    parser.add_argument("--num_objects", type=int, default=3,
                        help="Compatibility; not used by CarvedPath.")
    parser.add_argument("--agent_view_size", type=int, default=5,
                        help="Agent's egocentric FOV size.")
    parser.add_argument("--hint", action='store_true',
                        help="Whether to include hints in the prompt.")
    parser.add_argument("--failure_prob", type=float, default=0.0,
                        help="Probability of action faults (observation-only).")
    parser.add_argument("--blackout", action='store_true',
                        help="Enable log blackout (missing steps) in observation text.")
    parser.add_argument("--stale_obs_prob", type=float, default=0.0,
                        help="Probability of stale observation jitter.")
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed for reproducibility.")
    parser.add_argument("--rbpf_num_particles", type=int, default=400,
                        help="Number of particles for RBPF.")
    parser.add_argument("--list_unseen", action='store_true',
                        help="Whether to list unseen objects in the observation.")
    parser.add_argument("--list_empty", action='store_true',
                        help="Whether to list empty locations in the observation.")
    parser.add_argument("--unseen_example", action='store_true',
                        help="Whether to include an example of unseen objects in the prompt.")
    parser.add_argument("--dir", action='store_true',
                        help="Include facing-direction text in observation strings.")

    args = parser.parse_args()

    if args.random_seed is not None:
        random.seed(args.random_seed)

    main(args)
