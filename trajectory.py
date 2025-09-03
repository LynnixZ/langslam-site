# filename: run_evaluation.py

import gymnasium as gym
import logging
import time
import os
from utils import *
from Minigrid.utils.compare import *
import create_env
import argparse # 导入命令行参数库
from llm import get_response


# --- 2. 自动化测试的核心流程 ---
def run_single_test_episode(episode_id: int,args):
    """运行单次自动化测试"""
    log_dir = "evaluation_results"
    
    # --- 使用 logging.info() 来代替 print() ---
    logging.info(f"--- Starting Test Episode #{episode_id} ---")
    DOOR_Y_POS = random.randint(1, args.length - 2)  # 随机选择门的Y位置
    door_pos = (args.length, DOOR_Y_POS)
    my_env = gym.make(
        args.env_name, 
        length=args.length,
        num_objects=args.num_objects,
        agent_view_size=args.agent_view_size,
        door_pos=door_pos ,
        render_mode="rgb_array" 
    )
    obs, info = my_env.reset()
    agent_start_pos = my_env.unwrapped.agent_pos
    agent_start_dir = obs['direction']
    # 5. 【关键】将获取到的起始状态传递给 build_script 函数


    my_script = build_script(args.env_name,args.length, agent_view_size=args.agent_view_size, 
                            agent_start_pos=agent_start_pos, agent_start_dir=agent_start_dir,
                             easy=args.easy, oneD=args.oneD, door_pos=door_pos)
    print(f"Script for episode {episode_id}: {my_script}")
    gif_filename = os.path.join(args.log_dir, f"episode_{episode_id}_trace.gif")

    run_scripted_playthrough(my_env, my_script, picfilename=gif_filename, oneDim=args.oneD)

    
    return None

def main(args):
    setup_logging(args.log_dir)

    
    for i in range(args.num_episodes):
        run_single_test_episode(i, args)

        time.sleep(5) 
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run automated evaluation for LLM spatial reasoning.")
    
    parser.add_argument("--log_dir", type=str, default="evaluation_results", 
                        help="Directory to save logs and GIFs.")
    parser.add_argument("--env_name", type=str, default="MiniGrid-MyCorridor-v0", 
                        help="Name of the environment to use.")
    parser.add_argument("--num_episodes", type=int, default=1, 
                        help="Number of random episodes to test.")
    parser.add_argument("--length", type=int, default=10, 
                        help="Length of the corridor for the environment.")
    parser.add_argument("--num_objects", type=int, default=3, 
                        help="Number of goal objects to place in the corridor.")
    parser.add_argument("--easy", action='store_false',
                        help="Use easy mode with fewer objects and simpler tasks.")
    parser.add_argument("--oneD", action='store_false',
                        help="Use 1D mode with only one object type and simpler tasks.")
    parser.add_argument("--agent_view_size", type=int, default=5,
                        help="Size of the agent's view in the environment.")
    args = parser.parse_args()
    main(args)