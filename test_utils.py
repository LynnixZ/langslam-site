from utils import *
def build_prompt(obs: dict, history: str = '', planned: bool = False, oneD: bool = False, absolute: bool = False) -> str:
    if absolute:
        return _build_prompt_absolute(obs['mission'], describe_observation(obs), history, planned) ###!这里还没建设describe_observation的absolute版，暂时不可用
    else:
        return _build_prompt_relative(obs, history,planned)

def _build_prompt_absolute(mission: str, world_description: str, history: str, planned: bool) -> str:
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
{mission}
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

def _build_prompt_relative(obs, history: str,planned: bool = False, oneD: bool = False, agent_view_size: int=7) -> str:
    mission = obs['mission']
    planned_text = """
If you are certain about what to do, you can select a sequence of actions, for example,
[forward, right, forward, forward, pickup, right]
But you don't have to rush. Remember turning or moving will give you a new view and new information!
    """ 
    rel_position = (agent_view_size-1) // 2

    prompt = f"""You are an intelligent agent in a grid-based video game. Your goal is to navigate and interact with objects to solve a mission.
Your Mission is to {mission}.

### YOUR COORDINATE SYSTEM ###
Your view is described using a special (Depth, Side) coordinate system:
1.  **Depth (0-{agent_view_size-1})**: Represents the front-back direction. Depth 0 is your location. Larger numbers are farther away from you.
2.  **Side ({-rel_position}-{rel_position})**: Represents the left-right direction. Side 0 is your center line. Negative Numbers are to your left; positive numbers are to your right.
3.  **Your Position**: You are ALWAYS at coordinate (0, 0) in this system. The space directly in front of you is (1, 0).

### RULES OF THE WORLD ###
1.  The world is a single, continuous map which may contain multiple rooms connected by doors.
2.  A change in your view means you have moved or turned, NOT that you have entered a new, separate room.
3.  You can only interact with objects that are directly in front of you. You cannot toggle the wall.
4.  Everything you need may not be visible in your current view. You can move yourself and explore the world to find all objects.
### ACTION HISTORY LOG (Last 10 Steps) ###
{history if history else 'This is the first step. Lets go!'}

Based on your mission and current situation, you MUST choose your next action from the following fixed list:
- 'left': Turn left by 90 degrees on the spot. For example, if you are facing East, turning left you will now face North.
- 'right': Turn right by 90 degrees on the spot. For example, if you are facing East, turning right you will now face South.
- 'forward': Move one step forward in the direction you are facing if there is no wall in front of you.
- 'pickup': Pick up an object directly in front of you.
- 'drop': Drop the object you are carrying.
- 'toggle': Interact with an object in front of you (e.g., open a door).

Take a deep thought of what to do and respond in a new line with only the action name in [], for example: 
[forward]
{planned_text if planned else ""}
"""
    return prompt
import ast 

import re
def extract_commands(response: str) -> list[str]:
    """
    (新版) 能够安全地解析字符串形式的列表，例如 "['forward', 'left']" 
    """
    response = response.strip()
    
    # 方案1：优先尝试用 ast.literal_eval 解析列表字符串
    # 这种方法最稳健，可以直接处理 "['forward', 'left']" 这样的格式
    if response.startswith('[') and response.endswith(']'):
        try:
            commands = ast.literal_eval(response)
            if isinstance(commands, list):
                # 确保列表中的所有元素都是清理过的字符串
                return [str(cmd).strip().lower() for cmd in commands]
        except (ValueError, SyntaxError):
            # 如果解析失败（比如格式不规范），则忽略，并继续尝试下面的方法
            pass

    # 方案2：如果上面方法失败，退回到我们之前的正则表达式方法，但增加引号去除
    match = re.search(r'\[(.*?)\]', response)
    if match:
        commands_str = match.group(1)
        # 关键修改：在strip()之外，再增加一次对单引号和双引号的去除
        return [cmd.strip().lower().strip("'\"") for cmd in commands_str.split(',')]

    # 方案3：最后的备用方案，逐行寻找独立的命令词
    found_commands = []
    valid_actions = ['left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done']
    for line in response.splitlines():
        cleaned_line = line.strip().lower()
        if cleaned_line in valid_actions:
            found_commands.append(cleaned_line)
    
    if found_commands:
        return found_commands

    return [] # 如果完全没找到，返回空列表