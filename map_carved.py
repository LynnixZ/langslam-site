# filename: run_evaluation.py

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
    generate_ground_truth, parse_llm_output, compare_maps, generate_prompt, map_to_records
)
import create_env  # noqa: F401  # keep env registration side-effects
import argparse
from llm import get_response
from algo import rbpf


def run_single_test_episode(episode_id: int, args) -> Dict[str, Any]:
    """运行单次自动化测试，返回一个可写入 JSON 的结果字典。"""
    logging.info(f"--- Starting Test Episode #{episode_id} ---")

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
        # 拓宽参数
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
        rng_seed=1234,
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

    # 5) GT 与提示
    ground_truth_map = generate_ground_truth(my_env, env_name=env_name_eval)
    my_env.close()

    prompt = generate_prompt(
        obsstring,
        start_pos,
        oneD=False,
        num_objects=args.num_objects,
        length=args.length,
        agent_view_size=args.agent_view_size,
        tools=args.use_tools,
        hint=args.hint,
        failure_prob=args.failure_prob,
        blackout=args.blackout,
        stale_obs_prob=args.stale_obs_prob,
        env_names=env_name_eval,
        list_unseen=args.list_unseen,
        list_empty=args.list_empty,
        unseen_example=args.unseen_example,
    )
    logging.info(f"Prompt for episode #{episode_id} prepared.: {prompt}")

    # 6) LLM 推理
    llm_response_text = get_response(prompt, args.model, use_tools=(args.use_tools == 'code'))
    llm_map = parse_llm_output(llm_response_text, env_name=env_name_eval)
    result_llm = compare_maps(ground_truth_map, llm_map, env_name=env_name_eval)
    logging.info(f"LLM Response for episode #{episode_id}:\n{llm_response_text}")
    logging.info(f"LLM Result: {'PERFECT MATCH' if result_llm['is_perfect_match'] else 'IMPERFECT'}")
    logging.info(result_llm['details'])

    # 7) 可选：RBPF
    rbpf_block = None
    if args.eval_rbpf:
        logging.info(f"Running RBPF for episode #{episode_id} ...")
        rbpf_text = rbpf(prompt, rng_seed=episode_id, num_particles=200)
        rbpf_map = parse_llm_output(rbpf_text, env_name=env_name_eval)
        result_rbpf = compare_maps(ground_truth_map, rbpf_map, env_name=env_name_eval)
        rbpf_block = {
            "raw_text": rbpf_text,
            "metrics": result_rbpf,
        }
        logging.info(f"RBPF Result: {'PERFECT MATCH' if result_rbpf['is_perfect_match'] else 'IMPERFECT'}")
        logging.info(result_rbpf['details'])
    llm_img_path = os.path.join(args.log_dir, f"episode_{episode_id}_llm_pred.png")
    save_empty_prediction_image(
        llm_map=llm_map,
        width=args.length,
        height=args.length,
        out_path=llm_img_path,
        cell=14,              # 可调：单格像素
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
        "ground_truth": map_to_records(ground_truth_map),   # <-- 变 JSON 友好
        "llm": {
            "raw_text": llm_response_text,
            "llm_result": map_to_records(llm_map),          # <-- 变 JSON 友好
            "metrics": result_llm,
        },
    }
    if rbpf_block is not None:
        episode_record["rbpf"] = rbpf_block

    logging.info(f"--- Finished Test Episode #{episode_id} ---\n")
    return episode_record


def main(args):
    setup_logging(args.log_dir)  # 仍然写 evaluation_log.txt

    all_episodes: List[Dict[str, Any]] = []
    sum_overall = 0.0
    sum_goal = 0.0
    sum_dist = 0.0

    sum_overall_rbpf = 0.0
    sum_goal_rbpf = 0.0
    sum_dist_rbpf = 0.0
    rbpf_count = 0

    for i in range(args.num_episodes):
        rec = run_single_test_episode(i, args)
        all_episodes.append(rec)

        # LLM 汇总
        m = rec["llm"]["metrics"]
        sum_overall += m["overall_accuracy"]
        sum_goal += m["goal_accuracy"]
        sum_dist += m["euclidean_distance_error"]

        # RBPF 汇总（可选）
        if args.eval_rbpf and "rbpf" in rec:
            mr = rec["rbpf"]["metrics"]
            sum_overall_rbpf += mr["overall_accuracy"]
            sum_goal_rbpf += mr["goal_accuracy"]
            sum_dist_rbpf += mr["euclidean_distance_error"]
            rbpf_count += 1

        if i < args.num_episodes - 1:
            time.sleep(5)

    # 生成 JSON 汇总
    summary: Dict[str, Any] = {
        "total_episodes": args.num_episodes,
        "avg_overall_accuracy": (sum_overall / args.num_episodes) if args.num_episodes else 0.0,
        "avg_goal_accuracy": (sum_goal / args.num_episodes) if args.num_episodes else 0.0,
        "avg_euclidean_distance_error": (sum_dist / args.num_episodes) if args.num_episodes else 0.0,
    }
    if args.eval_rbpf and rbpf_count > 0:
        summary.update({
            "rbpf_avg_overall_accuracy": sum_overall_rbpf / rbpf_count,
            "rbpf_avg_goal_accuracy": sum_goal_rbpf / rbpf_count,
            "rbpf_avg_euclidean_distance_error": sum_dist_rbpf / rbpf_count,
            "rbpf_episodes": rbpf_count,
        })

    payload = {
        "config": {
            "model": args.model,
            "log_dir": args.log_dir,
            "env_name": "MiniGrid-CarvedPathRoom-v0",
            "num_episodes": args.num_episodes,
            "length": args.length,
            "num_objects": args.num_objects,
            "agent_view_size": args.agent_view_size,
            "failure_prob": args.failure_prob,
            "blackout": args.blackout,
            "stale_obs_prob": args.stale_obs_prob,
            "use_tools": args.use_tools,
            "hint": args.hint,
            "eval_rbpf": args.eval_rbpf,
            "random_seed": args.random_seed,
        },
        "episodes": all_episodes,
        "summary": summary,
    }

    os.makedirs(args.log_dir, exist_ok=True)
    json_path = os.path.join(args.log_dir, "evaluation_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logging.info("=" * 50)
    logging.info("--- EVALUATION SUMMARY (JSON written) ---")
    logging.info(f"Saved JSON to '{json_path}'")
    logging.info("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run automated evaluation for LLM spatial reasoning.")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro",
                        help="Which language model to use.")
    parser.add_argument("--log_dir", type=str, default="evaluation_results",
                        help="Directory to save logs and JSON/GIFs.")
    parser.add_argument("--env_name", type=str, default="MiniGrid-MyCorridor-v0",
                        help="Name of the environment to use. (Not used for CarvedPath runs)")
    parser.add_argument("--num_episodes", type=int, default=5,
                        help="Number of random episodes to test.")
    parser.add_argument("--length", type=int, default=10,
                        help="Length of the square room side.")
    parser.add_argument("--num_objects", type=int, default=3,
                        help="Kept for compatibility; not used by CarvedPath.")
    parser.add_argument("--easy", action='store_true',
                        help="Use easy mode with fewer objects and simpler tasks.")
    parser.add_argument("--oneD", action='store_true',
                        help="Use 1D mode (compat flag, not used here).")
    parser.add_argument("--dir", action='store_true',
                        help="Include facing-direction text in observation strings.")
    parser.add_argument("--use_tools", type=str, default=None, choices=['code', 'ASCIIart'],
                        help="Use tools for the LLM.")
    parser.add_argument("--agent_view_size", type=int, default=5,
                        help="Agent's egocentric FOV size.")
    parser.add_argument("--hint", action='store_true',
                        help="Whether to include hints in the prompt.")
    parser.add_argument("--failure_prob", type=float, default=0.0,
                        help="Probability of action faults.")
    parser.add_argument("--blackout", action='store_true',
                        help="Enable log blackout (missing steps) in observation text.")
    parser.add_argument("--stale_obs_prob", type=float, default=0.0,
                        help="Probability of stale observation jitter.")
    parser.add_argument("--same_color_goals", action='store_true',
                        help="Whether to use the same color for all goals (not used here).")
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed for reproducibility.")
    parser.add_argument("--eval_rbpf", action='store_true',
                        help="If set, also evaluate RBPF and record its metrics.")
    parser.add_argument("--list_unseen", action='store_true',
                        help="Whether to list unseen objects in the observation.")
    parser.add_argument("--list_empty", action='store_true',
                        help="Whether to list empty locations in the observation.")
    parser.add_argument("--unseen_example", action='store_true',
                        help="Whether to include an example of unseen objects.")
    args = parser.parse_args()

    if args.random_seed is not None:
        random.seed(args.random_seed)

    main(args)
