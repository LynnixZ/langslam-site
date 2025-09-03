import gymnasium as gym
import minigrid
import numpy as np
from collections import defaultdict, deque
import google.generativeai as genai
import os
import time
import re
import argparse
import imageio
import logging
import sys

# --- 从minigrid库中导入我们需要的核心组件 ---
from minigrid.core.actions import Actions
from minigrid.wrappers import FullyObsWrapper

# --- Logging Setup Function ---
def setup_logging(log_filepath: str):
    """Configures logging to output to both a file and the console."""
    # Ensure the directory for the log file exists
    os.makedirs(os.path.dirname(log_filepath), exist_ok=True)
    
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
    root_logger = logging.getLogger()
    # Clear existing handlers to prevent duplicate logging
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    console_formatter = logging.Formatter("%(message)s")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

# --- API Configuration ---
def configure_api():
    """Configures the Gemini API, trying environment variables first."""
    try:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        logging.info("✅ Gemini API configured successfully from environment variable.")
    except (KeyError, TypeError):
        logging.warning("❌ WARNING: GOOGLE_API_KEY environment variable not found.")
        try:
            api_key = input("Please enter your Google API Key: ")
            if api_key:
                genai.configure(api_key=api_key)
                logging.info("✅ Gemini API configured successfully with user-provided key.")
            else:
                raise ValueError("API Key cannot be empty.")
        except (ValueError, EOFError, KeyboardInterrupt) as e:
            logging.error(f"❌ FATAL: Could not configure Gemini API. {e}")
            exit()

# --- Mappings and Helper Functions (Unchanged) ---
IDX_TO_OBJECT = {0:'unseen', 1:'empty', 2:'wall', 3:'floor', 4:'door', 5:'key', 6:'ball', 7:'box', 8:'goal', 9:'lava', 10:'agent'}
IDX_TO_COLOR = {0:'red', 1:'green', 2:'blue', 3:'purple', 4:'yellow', 5:'grey'}
IDX_TO_STATE = {0:'open', 1:'closed', 2:'locked'}

def describe_world_from_full_grid(obs: dict, agent_pos: tuple, agent_dir: int) -> str:
    full_grid = obs['image']
    height, width, _ = full_grid.shape
    agent_direction_str = describe_agent_orientation(agent_pos, agent_dir)
    grouped_objects = defaultdict(list)
    object_grid, color_grid, state_grid = full_grid[:, :, 0], full_grid[:, :, 1], full_grid[:, :, 2]
    
    for y in range(height):
        for x in range(width):
            obj_id = object_grid[y, x]
            if obj_id not in [0, 1, 10]:
                color_id, state_id = color_grid[y, x], state_grid[y, x]
                obj_name = IDX_TO_OBJECT.get(obj_id, f'object(ID:{obj_id})')
                color_name = IDX_TO_COLOR.get(color_id, f'color(ID:{color_id})')
                object_key = (color_name, obj_name)
                if obj_name == 'door':
                    state_name = IDX_TO_STATE.get(state_id, f'state(ID:{state_id})')
                    object_key = (color_name, obj_name, state_name)
                grouped_objects[object_key].append(f"({y}, {x})")

    object_descriptions = []
    for object_key, coords in grouped_objects.items():
        color_name, obj_name = object_key[0], object_key[1]
        coords_str = ", ".join(coords)
        description = f"- A {color_name} {obj_name} is at absolute coordinate {coords_str}." if len(coords) == 1 else f"- There are {color_name} {obj_name}s at absolute coordinates: {coords_str}."
        if obj_name == 'door':
            description = description.replace('.', f', and it is {object_key[2]}.')
        object_descriptions.append(description)

    return agent_direction_str + "\nHere is a list of all noteworthy objects on the map:\n" + "\n".join(object_descriptions)

def describe_agent_orientation(agent_pos, agent_dir):
    agent_pos_tuple = (int(agent_pos[0]), int(agent_pos[1]))
    direction_map = {0: 'right (East)', 1: 'down (South)', 2: 'left (West)', 3: 'up (North)'}
    return f"Your current position is {agent_pos_tuple} and you are facing {direction_map.get(agent_dir, 'unknown')}."

def build_prompt_absolute(mission: str, world_description: str, history: str, planned: bool) -> str:
    # For brevity, the large string templates are omitted, but they are identical
    fixed_actions =  """- 'left': Turn left by 90 degrees on the spot.
- 'right': Turn right by 90 degrees on the spot.
- 'forward': Move one step forward in the direction you are facing.
- 'pickup': Pick up an object directly in front of you.
- 'drop': Drop the object you are carrying.
- 'toggle': Interact with an object in front of you (e.g., open a door)."""
    
    planned_text = """
If you are certain about what to do, you can select a sequence of actions, for example,
[forward, right, forward, forward, pickup, right]
But you don't have to rush. Remember turning or moving will give you a new view and new information!
""" if planned else ""
    
    rule= """
### Rules
Lava is deadly, and you cannot step on it. if you do, you will lose the game.
You can not toggle or cross the wall, it is a solid object.
You can only interact with objects that are directly in front of you.
    """
    
    prompt_template = f"""You are a logical robot agent with a top-down view of your entire grid world. Your task is to create a plan to solve a mission.
In this world, all objects are represented by 2 coordinates.
The first coordinate is the row (x-axis). The postive direction is east/right, and the negative direction is west/left.
The second coordinate is the column (y-axis). The positive direction is south/down, and the negative direction is north/up.

### Mission
"{mission}"
{rule}
### Recent Action History (up to 10 steps)
{history if history else 'This is the first step.'}

### Current World State
{world_description}

### Your Actions
Based on your mission and current situation, you MUST choose your next action from the following fixed list:
{fixed_actions}
Take a deep thought of what to do and respond in a new line with only the action name in [], for example: 
[forward]
{planned_text}
"""
    return prompt_template

def extract_commands(response: str) -> list[str]:
    match = re.search(r'\[(.*?)\]', response)
    if match:
        return [cmd.strip().lower().replace("'", "").replace('"',"") for cmd in match.group(1).split(',')]
    return []

def get_gemini_action_sequence(prompt: str, model:'gemini-2.5-flash') -> list[str]:
    model = genai.GenerativeModel(model)
    try:
        response = model.generate_content(prompt, request_options={"timeout": 100})
        logging.info("--- Gemini Raw Response ---")
        for line in response.text.splitlines():
            logging.info(line)
        logging.info("--------------------------")
        return extract_commands(response.text)
    except Exception as e:
        logging.error(f"Error calling Gemini API: {e}")
        return ["forward"]

def visualize_playthrough(env_name: str, seed: int, actions: list, filepath: str):
    logging.info(f"\n--- Creating visualization. Saving to {filepath} ---")
    env = gym.make(env_name, render_mode="rgb_array")
    obs, info = env.reset(seed=seed)
    frames = [env.render()]
    for action_id in actions:
        obs, reward, terminated, truncated, info = env.step(action_id)
        frames.append(env.render())
        if terminated or truncated: break
    env.close()
    imageio.mimsave(filepath, frames, fps=3)
    logging.info(f"--- Visualization saved to {filepath}! ---")

def main():
    parser = argparse.ArgumentParser(description="Run a Gemini-powered agent in a MiniGrid environment.")
    parser.add_argument("--env_name", type=str, default="MiniGrid-SimpleCrossingS11N5-v0", help="The name of the MiniGrid environment.")
    parser.add_argument("--seed", type=int, default=42, help="The random seed for the environment.")
    parser.add_argument("--max_steps", type=int, default=25, help="The maximum number of agent steps per episode.")
    parser.add_argument("--visual_name", type=str, default=None, help="Filename for the output GIF.")
    parser.add_argument("--planned", action="store_true", help="If set, enables the 'planned sequence' text in the prompt.")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash", help="Directory to save logs and GIFs.")
    parser.add_argument("--output_dir", type=str, default="runs", help="Directory to save logs and GIFs.")
    args = parser.parse_args()

    # --- Create Output Directory ---
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    model_name = args.model_name
    # --- Setup Logging with Full Path ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_filename = f"log_{args.env_name.replace('-', '_')}_seed{args.seed}_{timestamp}.log"
    log_filepath = os.path.join(output_dir, log_filename)
    setup_logging(log_filepath)

    # --- Configure API ---
    configure_api()

    # --- Construct Full Paths for Outputs ---
    env_name, seed, max_steps = args.env_name, args.seed, args.max_steps
    gif_filename = args.visual_name or f"{env_name.replace('-', '_')}_seed{seed}.gif"
    gif_filepath = os.path.join(output_dir, gif_filename)
    use_planned_prompt = args.planned

    logging.info("--- Configuration ---")
    logging.info(f"Log File: {log_filepath}")
    logging.info(f"Output GIF: {gif_filepath}")
    logging.info(f"Environment: {env_name}")
    logging.info(f"Seed: {seed}")
    logging.info(f"Max Steps: {max_steps}")
    logging.info(f"Planned Prompting: {'Enabled' if use_planned_prompt else 'Disabled'}")
    logging.info("---------------------\n")

    env = FullyObsWrapper(gym.make(env_name))
    ACTION_MAP = {action.name.lower(): action.value for action in Actions}
    obs, info = env.reset(seed=seed)
    
    history_deque = deque(maxlen=10)
    moves = 0
    done = False
    recorded_actions = []
    
    logging.info(f"--- Game Start (Env: {env_name}, Seed: {seed}) ---")
       
    while not done:
        if moves >= max_steps:
            logging.info(f"--- Exceeded max steps ({max_steps}). Stopping execution. ---")
            break
            
        agent_pos, agent_dir = env.unwrapped.agent_pos, obs['direction']
        world_text = describe_world_from_full_grid(obs, agent_pos, agent_dir)
        mission_text = obs['mission']
        
        logging.info("\n" + "="*50)
        logging.info(f"--- Agent's Turn (Overall Move #{moves+1}) ---")
        logging.info("="*50)
        
        history_for_prompt = "\n".join(list(history_deque))
        prompt = build_prompt_absolute(mission_text, world_text, history_for_prompt, planned=use_planned_prompt)
        logging.info(prompt)
        
        action_sequence = get_gemini_action_sequence(prompt, model=model_name)
        if not action_sequence:
            logging.warning("--- Gemini returned an empty plan. Defaulting to ['forward']. ---")
            action_sequence = ['forward']

        logging.info(f">>> Gemini's action plan: {action_sequence}")

        for action_name in action_sequence:
            if done: break
            if action_name in ACTION_MAP:
                action_id = ACTION_MAP[action_name]
                recorded_actions.append(action_id)
                obs, reward, terminated, truncated, info = env.step(action_id)
                done = terminated
                if terminated: logging.info("    MISSION ACCOMPLISHED!")
            else:
                logging.warning(f"--- Invalid action '{action_name}' in sequence. Skipping. ---")
            time.sleep(1) 

        agent_pos, agent_dir = env.unwrapped.agent_pos, obs['direction']
        agent_direction_str = describe_agent_orientation(agent_pos, agent_dir)
        history_entry = f"- Step {moves+1}: Chose action sequence {action_sequence}. {agent_direction_str}"
        history_deque.append(history_entry)
        logging.info(history_entry)
            
        moves += 1

    if recorded_actions:
        visualize_playthrough(env_name, seed, recorded_actions, filepath=gif_filepath)
        
    env.close()
    logging.info(f"\n--- Game Over! ---")
    logging.info(f"Episode finished in {moves} moves.")

if __name__ == "__main__":
    main()