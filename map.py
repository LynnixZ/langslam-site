# filename: run_evaluation.py

import gymnasium as gym
import logging
import time
import os
from utils import (
    build_script, run_scripted_playthrough, describe_observation, setup_logging,
    generate_ground_truth, parse_llm_output, compare_maps, generate_prompt
)
import random

import create_env
import argparse # 导入命令行参数库
from llm import get_response
from algo import rbpf

# --- 2. 自动化测试的核心流程 ---
def run_single_test_episode(episode_id: int,args):
    """运行单次自动化测试"""
    log_dir = "evaluation_results"
    
    # --- 使用 logging.info() 来代替 print() ---
    logging.info(f"--- Starting Test Episode #{episode_id} ---")
    env_kwargs = {
        'length': args.length,
        'num_objects': args.num_objects,
        'agent_view_size': args.agent_view_size,
        'render_mode': "rgb_array"
    }

    if args.env_name== "MiniGrid-TwoRooms-v0":
        door_y_pos = random.randint(1, args.length - 2)
        env_kwargs['door_pos'] = (args.length, door_y_pos)
    elif args.env_name == "MiniGrid-CustomEmpty-5x5-v0":
        env_kwargs['same_color_goals'] = args.same_color_goals

    my_env = gym.make(
        args.env_name, 
        **env_kwargs  
    )

    obs, info = my_env.reset()
    agent_start_pos = my_env.unwrapped.agent_pos
    agent_start_dir = obs['direction']

    my_script = build_script(args.env_name,args.length, agent_view_size=args.agent_view_size, 
                            agent_start_pos=agent_start_pos, agent_start_dir=agent_start_dir,
                             easy=args.easy, oneD=args.oneD)
                              # door_pos=env_kwargs['door_pos'])
    
    gif_filename = os.path.join(args.log_dir, f"episode_{episode_id}_trace.gif")
    print(args.stale_obs_prob)
    start_pos,obsstring = run_scripted_playthrough(my_env, my_script, picfilename=gif_filename, dir=args.dir, oneDim=args.oneD,
                                                   failure_prob=args.failure_prob,    
                                                   blackout=args.blackout,    
                                                   stale_obs_prob=args.stale_obs_prob )

    ground_truth_map = generate_ground_truth(my_env, args.oneD)
    my_env.close()
    logging.info(f"Sending prompt to Gemini for episode #{episode_id}...")
    prompt = generate_prompt(obsstring,start_pos, args.oneD, num_objects=args.num_objects, length=args.length, agent_view_size=args.agent_view_size, tools=args.use_tools , hint=args.hint,
                                                                                failure_prob=args.failure_prob,    
                                                   blackout=args.blackout,    
                                                   stale_obs_prob=args.stale_obs_prob )
    logging.info(f"Prompt for episode #{episode_id}:\n{prompt}")
  #rbpf
    rbpf_text = rbpf(prompt, rng_seed=episode_id, num_particles=200)
    rbpf_map = parse_llm_output(rbpf_text, args.oneD)

    print("RBPF Map:", rbpf_text)
    result_rbpf = compare_maps(ground_truth_map, rbpf_map)
    logging.info(f"RBPF Result: {'PERFECT MATCH' if result_rbpf['is_perfect_match'] else 'IMPERFECT'}")
    logging.info(result_rbpf['details'])
    print("RBPF Result:", result_rbpf)
    logging.info(f"--- Finished Test Episode #{episode_id} ---\n")
    llm_response_text = get_response(prompt,args.model, use_tools=args.use_tools=='code')
    logging.info(f"LLM Response for episode #{episode_id}:\n{llm_response_text}")
    llm_map = parse_llm_output(llm_response_text, args.oneD)
    result = compare_maps(ground_truth_map, llm_map)
    
    logging.info(f"--- Judgment for Episode #{episode_id} ---")
    logging.info(f"Result: {'PERFECT MATCH' if result['is_perfect_match'] else 'IMPERFECT'}")
    logging.info(result['details'])
    

    return result,result_rbpf

def main(args):
    setup_logging(args.log_dir)

    total_overall_acc = 0
    total_goal_acc = 0
    total_overall_acc_rbpf = 0
    total_goal_acc_rbpf = 0
    total_euclidean_distance_error = 0.0
    total_euclidean_distance_error_rbpf = 0.0

    for i in range(args.num_episodes):
        episode_result, episode_result_rbpf = run_single_test_episode(i, args)
        total_overall_acc += episode_result['overall_accuracy']
        total_goal_acc += episode_result['goal_accuracy']
        total_euclidean_distance_error += episode_result['euclidean_distance_error']

        total_overall_acc_rbpf += episode_result_rbpf['overall_accuracy']
        total_goal_acc_rbpf += episode_result_rbpf['goal_accuracy']
        total_euclidean_distance_error_rbpf += episode_result_rbpf['euclidean_distance_error']
        if i < args.num_episodes - 1:
             time.sleep(5) 
        
    logging.info("="*50)
    logging.info("--- EVALUATION SUMMARY ---")
    logging.info(f"Total episodes run: {args.num_episodes}")
    logging.info(f"Average Overall Accuracy: {total_overall_acc / args.num_episodes:.2%}")
    logging.info(f"Average Goal-Only Accuracy: {total_goal_acc / args.num_episodes:.2%}")
    logging.info(f"Average Euclidean Distance Error: {total_euclidean_distance_error / args.num_episodes:.2f}")
    logging.info(f"RBPF Average Overall Accuracy: {total_overall_acc_rbpf / args.num_episodes:.2%}")
    logging.info(f"RBPF Average Goal-Only Accuracy: {total_goal_acc_rbpf / args.num_episodes:.2%}")
    logging.info(f"RBPF Average Euclidean Distance Error: {total_euclidean_distance_error_rbpf / args.num_episodes:.2f}")
    logging.info("="*50)
    logging.info(f"Full log has been saved to '{os.path.join(args.log_dir, 'evaluation_log.txt')}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run automated evaluation for LLM spatial reasoning.")
    
    parser.add_argument("--model", type=str, default="gemini-2.5-pro",
                        help="Which language model to use.")
    parser.add_argument("--log_dir", type=str, default="evaluation_results", 
                        help="Directory to save logs and GIFs.")
    parser.add_argument("--env_name", type=str, default="MiniGrid-MyCorridor-v0", 
                        help="Name of the environment to use.")
    parser.add_argument("--num_episodes", type=int, default=5, 
                        help="Number of random episodes to test.")
    parser.add_argument("--length", type=int, default=10, 
                        help="Length of the corridor for the environment.")
    parser.add_argument("--num_objects", type=int, default=3, 
                        help="Number of goal objects to place in the corridor.")
    parser.add_argument("--easy", action='store_true',
                        help="Use easy mode with fewer objects and simpler tasks.")
    parser.add_argument("--oneD", action='store_true',
                        help="Use 1D mode with only one object type and simpler tasks.")
    parser.add_argument("--dir", action='store_true',
                        help="Use 1D mode with only one object type and simpler tasks.")
    parser.add_argument("--use_tools", type=str, default=None, choices=['code', 'ASCIIart'],
                        help="Use tools for the LLM.")
    parser.add_argument("--agent_view_size", type=int, default=5,
                        help="Size of the agent's view in the environment.")
    parser.add_argument("--hint", action='store_true',
                        help="Whether to include hints in the prompt.")
    parser.add_argument("--failure_prob", type=float, default=0.0,
                        help="Probability of failure in the scripted playthrough.")
    parser.add_argument("--blackout", action='store_true',
                        help="Enable log blackout feature.")
    parser.add_argument("--stale_obs_prob", type=float, default=0.0,
                        help="Probability of stale observations in the scripted playthrough.")
    parser.add_argument("--same_color_goals", action='store_true',
                        help="Whether to use the same color for all goals.")
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()
    main(args)