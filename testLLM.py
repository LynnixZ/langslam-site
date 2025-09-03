import gymnasium as gym
import google.generativeai as genai
import os
import time
import re
import argparse
import imageio
import logging
import sys
from collections import deque, defaultdict
from llm import get_response
from utils import describe_observation, IDX_TO_OBJECT, IDX_TO_COLOR, IDX_TO_STATE, COMMAND_TO_ACTION, setup_logging
import create_env
from test_utils import build_prompt,extract_commands



def main(args):
    """Main function to run the agent in the environment."""
    # --- Setup Directories and Logging ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.output_dir, f"{args.env_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    log_filepath = os.path.join(run_dir, "run.log")
    setup_logging(log_filepath)

    gif_filepath = os.path.join(run_dir, "playthrough.gif")

    logging.info("--- Agent Configuration ---")
    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")
    logging.info("---------------------------\n")

    # --- Initialize Environment and Agent State ---
    env = gym.make(args.env_name,render_mode="rgb_array")
    obs, info = env.reset(seed=args.seed)

    history = deque(maxlen=10)
    frames = []
    history.append(f"Initial Observation: {describe_observation(obs)}. Let's go!")
    total_steps = 0
    
    logging.info(f"--- Game Start (Env: {args.env_name}, Seed: {args.seed}) ---")

    # --- Main Agent Loop ---
    for turn in range(args.max_turns):
        logging.info("\n" + "="*50 + f"\n--- Agent Turn #{turn + 1} ---")
        
        # 1. Observe the world and render frame
        frames.append(env.render())
        mission = obs['mission']
        
        # 2. Build Prompt and Get Action Plan from LLM
        prompt = build_prompt(obs=obs, history="\n".join(history), planned=True)
        logging.info(f"--- Sending Prompt to {args.model_name} ---\n{prompt}")
        
        response_text = get_response(prompt, args.model_name)
        action_plan = extract_commands(response_text)

        if not action_plan:
            logging.warning("LLM returned an empty or invalid plan. Defaulting to 'forward'.")
            action_plan = ['forward']

        logging.info(f">>> LLM Action Plan: {action_plan}")

        # 3. Execute the action plan
        for i, action_name in enumerate(action_plan):
            if total_steps >= args.max_steps:
                logging.info(f"Max total steps ({args.max_steps}) reached. Terminating.")
                break
        
            action_id = COMMAND_TO_ACTION.get(action_name, None)

            obs, reward, terminated, truncated, info = env.step(action_id)
            total_steps += 1
            logging.info(f"    Step {total_steps}: Executed '{action_name}'.")

            if terminated:
                logging.info("--- MISSION ACCOMPLISHED! ---")
                frames.append(env.render())
                break
            if truncated:
                logging.info("--- Episode truncated. ---")
                break
        state_description = describe_observation(obs)

        history.append(f"Turn {turn+1}: Action: {action_plan}. Saw '{state_description}'.")
        
        if terminated or truncated or total_steps >= args.max_steps:
            break
            
        time.sleep(1) # Pause between turns to respect API rate limits

    # --- Cleanup and Save Artifacts ---
    env.close()
    if frames:
        logging.info(f"Saving playthrough to {gif_filepath}")
        imageio.mimsave(gif_filepath, frames, fps=3)

    logging.info(f"\n--- Game Over! ---")
    logging.info(f"Episode finished after {total_steps} steps.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a Gemini-powered agent in a MiniGrid environment.")
    parser.add_argument("--env_name", type=str, default="MiniGrid-Empty-5x5-v0", help="The name of the MiniGrid environment.")
    parser.add_argument("--seed", type=int, default=None, help="The random seed for the environment.")
    parser.add_argument("--max_turns", type=int, default=15, help="Max number of times the agent asks the LLM for a plan.")
    parser.add_argument("--max_steps", type=int, default=50, help="Max total number of steps the agent can take.")
    parser.add_argument("--model_name", type=str, default="gemini-1.5-flash", help="Name of the Gemini model to use.")
    parser.add_argument("--output_dir", type=str, default="agent_runs", help="Directory to save logs and GIFs.")
    parser.add_argument("--room_size", type=int, default=5, help="Size of the room (width and height).")

    args = parser.parse_args()
    main(args)