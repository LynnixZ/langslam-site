
def generate_prompt(obsstring: str, startpos, oneD: bool = True, num_objects: int = 0, length: int = 10, agent_view_size: int = 5, tools: str = None, hint: bool = False,     failure_prob: float = 0.0,      # “幽灵移动”的概率
    blackout: bool=False, stale_obs_prob: float = 0.0 ,list_unseen: bool = False, list_empty: bool = False, unseen_example: bool = False,
      env_names="MiniGrid-CustomEmpty-5x5-v0") -> str:
    """
    生成用于AI推理的提示文本。
    包含环境规则、任务描述和观察日志。
    """
    if oneD:
        return f"""
You are an expert spatial reasoning AI. Your task is to build a consistent 1D map of a corridor based on a sequence of limited, first-person observations.

### RULES OF THE WORLD ###
1.  The world is a static, continuous 1D track. Your absolute position on this track is a single integer.
2.  Your observation consists ONLY of the {agent_view_size-1} grid cells directly in front of you.
3.  The number you see (e.g., "at coordinate 2") is a **relative position**, indexed 1 to {agent_view_size-1}.
    - Position 1 is the cell immediately in front of you.
    - Position  {agent_view_size-1} is the farthest cell you can see.
    - An object at relative position {agent_view_size} or greater is invisible to you.
4.  Your movement is governed by your absolute position and direction:
    - 'forward': Changes your absolute position by +1 if facing East (positive direction), or -1 if facing West (negative direction).
    - 'turnaround': Reverses your direction (East becomes West, West becomes East).

### YOUR TASK ###
Your mission is to process the following log step-by-step and create a single, unified map of the world.
Your starting absolute position is 1. East is the direction of increasing position numbers.

### OBSERVATION LOG ###
{obsstring}

### YOUR RECONSTRUCTION ###
First, reason step-by-step about your position and what you see.
Finally, provide the final map summary in a new line, wrapped in "&&&" delimiters. For example:
&&&
a red flag at coordinate 3, a green flag at coordinate 8, a blue wall at coordinate 5, a blue wall at coordinate 0
&&&
"""

    else:
        
        hint_string= f"""
### HINT ###
Facing East: (X, Y) = (x + Depth, y + Side) Turning right, you are facing South; turning left, you are facing North.
Facing South: (X, Y) = (x - Side, y + Depth) Turning right, you are facing West; turning left, you are facing East.
Facing West: (X, Y) = (x - Depth, y - Side) Turning right, you are facing North; turning left, you are facing South.
Facing North: (X, Y) = (x + Side, y - Depth) Turning right, you are facing East; turning left, you are facing West.
Note that going forward may change your absolute position in one dimensions if there is no wall in front of you.
For example, if you are at (2, 3) and facing East, going forward will change your position to (3, 3).
if you are at (2, 3) and facing South, going forward will change your position to (2, 4).
"""
        if tools is not None and tools == 'ASCIIart':
            tools_prompt = f"""
### TOOLS: ###
You can use ASCII art as a "thinking tool" to show your work for every significant step. 
Your output for each step must could be like this:

---
**Step 3 Analysis:**
* **Status:** Position=(1, 1), Direction=east
* **Observation:** Purple Flag at (2, 1), Green Flag at (1, 1).
* **Map:**
    ```
    Legend: > (Robot facing east), P (Purple), G (Green)
      y
    3 | . . . . .
    2 | . . . . .
    1 | . > . . .
    0 | . . G P .
    --+----------->
        0 1 2 3 4 x
---
"""
        elif tools == 'code':
            tools_prompt = f"""
### TOOLS: ###
You can generate Python code to reason about the world and build your map."""
        else:
            tools_prompt = ""

        if failure_prob > 0.0 or stale_obs_prob > 0.0 or blackout:
                    
            failure_desc = (
                "- **Action Faults:** Two kinds may occur:\n"
                "  - *Spurious action in the log*: an action is recorded (e.g., 'left', 'forward') but the robot does not move; its pose stays the same.\n"
                "  - *Multi-step merge*: after executing the current command, the robot silently move one or two steps without any log entry. "
                if failure_prob > 0.0 else ""
            )

            # 描述“观测抖动”：离散高斯偏移旗子坐标
            stale_desc = (
    f"- **Noisy Observations:** Any reported coordinate pair in the local view (flags or walls) may be perturbed by discrete Gaussian noise. "
    f"For each coordinate, offsets of ±1 are common, ±2 less common, and ±3 rare. Coordinates are clipped to the {agent_view_size}×{agent_view_size} bounds. "
                if stale_obs_prob > 0.0 else ""
            )

            # 描述“日志黑洞”：连续缺失 5–10 步
            blackout_desc = (
                "- **Log Blackout:** A continuous block of actions (5–10 steps) may be missing from the log. "
                "You may see, for example, Step 40 followed by Step 41, but the robot's pose will have changed sharply as if it teleported."
                if blackout else ""
            )

            # --- 将它们干净地组合起来 ---
            # 使用列表推导式和 join 来过滤空字符串并用换行符连接
            descriptions = [desc for desc in [failure_desc, stale_desc, blackout_desc] if desc]
            
            corruption_warning = f"""
### WARNING: DATA CORRUPTION ###
Your primary challenge is that the data acquisition process is unreliable. You must account for the following potential anomalies:
{chr(10).join(descriptions)}

You cannot blindly trust every entry in the log. You must use the full sequence of observations to infer the most likely map and trajectory.
        """
        else:
            corruption_warning = ""


        if env_names=="MiniGrid-CustomEmpty-5x5-v0":
            mission=f"""
Your mission is to process the following log step-by-step and return the coordinates of the flags.
You will see walls in your view, but they are not part of the map you need to build. Neglect them. 
Note that each flag is uniquely colored."""
            answer = "a red flag at coordinate (3,4) a green flag at coordinate (2,-2)."

        else:
            # carved-room（只墙与空）
            seen_clause = (
                "To make this task simpler, unknown cells **inside your local view** that are blocked by walls are listed in the log as "
                "`You cannot see coordinates (d,s), ...`."
                if list_unseen else
                "Cells blocked by walls inside your view are not listed in the log."
            )
            empty_clause = (
                "Empty cells inside your local view may be listed as `Coordinates (d,s), ... are empty`."
                if list_empty else
                "Empty cells are not explicitly listed in the log."
            )
            identify_unseen_example = (
                "Take this as an example: You can see grey walls at coordinates (0, -1), (1, -1), (2, -1), (2, 0), (2, 1), (3, 1), (4, 1)."
                "In this case, since the walls in (2, -1), (2, 0), (2, 1), (3, 1), (4, 1) wrap around the grid (3, 0), (3, -1), (4, 0), (4, -1) in your view, those coordinates are considered unseen."
                f"""
Specifically, apply the Ray rule per Side s:
- For s = -1, nearest wall depth d_wall(-1) = 0  → all cells with (d ≥ 1, s = -1) are UNKNOWN (behind the nearest wall).
- For s =  0, nearest wall depth d_wall(0)  = 2  → (1,0) is VISIBLE EMPTY; (d ≥ 2, s=0) are UNKNOWN.
- For s =  1, nearest wall depth d_wall(1)  = 2  → (1,1) is VISIBLE EMPTY; (d ≥ 2, s=1) are UNKNOWN.
- For s =  2, no wall reported               → (1..FOV_max, 2) are VISIBLE EMPTY unless diagonally blocked.
- For s = -2, no wall reported               → (1..FOV_max, -2) are VISIBLE EMPTY unless diagonally blocked.

Diagonal occlusion rule (corner blocking):
A cell (d,s) is NOT visible if both (d-1,s-1) and (d-1,s+1) are walls. Otherwise it remains visible.

In particular, cells like (3,0), (4,0), (3,-1), (4,-1) lie at depths ≥ the nearest walls on their sides, so they are UNKNOWN (do not list them as empty).
Only the cells strictly in front of the nearest wall on each side, and not corner-blocked, are EMPTY at this step."""
                "The rest of the coordinates in your view are considered empty."
                if unseen_example else ""
            )

            mission = f"""
    Your mission is to return the coordinates of **all EMPTY cells** in the global map.

    ## What counts as EMPTY
    A cell is EMPTY if **either**:
    1) You **stood on it** or **moved through it** at any time (your traversed path), or
    2) It was in your local {agent_view_size}×{agent_view_size} field of view at least once and you clearly saw **no wall** in that cell. Note there's empty spaces you haven't stand on but it was in your view. Please identify them as well.

    ## Visibility and Occlusion
    - You **cannot see through walls**. Any cell *behind* a wall along a line-of-sight ray is **unknown**.
    - Diagonal shadowing: a cell at (d, s) is **not visible** if both its orthogonal predecessors are walls (i.e., blocked by the two adjacent cells).
    - {identify_unseen_example}
    - {seen_clause}
    - {empty_clause}
 

    """
            answer = "empty spaces at coordinates: (1,1), (1,2), (2,1), (2,2)"

        rel_position = (agent_view_size-1) // 2
        return f"""
You are a robot agent inside a grid world. Your task is to build a consistent 2D map of a room based on a {agent_view_size}x{agent_view_size} local view.

### YOUR COORDINATE SYSTEM ###
Your view is described using a special (Depth, Side) coordinate system:
1.  **Depth (0-{agent_view_size-1})**: Represents the front-back direction. Depth 0 is your location. Larger numbers are farther away from you.
2.  **Side ({-rel_position}-{rel_position})**: Represents the left-right direction. Side 0 is your center line. Negative Numbers are to your left; positive numbers are to your right.
3.  **Your Position**: You are ALWAYS at coordinate (0, 0) in this system. The space directly in front of you is (1, 0).
4.  Your movement is governed by your absolute position and direction:
    - 'forward': Changes your absolute position by +1 or -1 in one coordinate depending on your direction.
    - 'turnaround': Reverses your direction (East becomes West, West becomes East).
    - 'left': Turn left by 90 degrees on the spot. For example, if you are facing East, turning left you will now face North.
    - 'right':  Turn right by 90 degrees on the spot. For example, if you are facing East, turning right you will now face South.

### YOUR TASK ###
{mission}
All coordinates in the final summary must be absolute map coordinates (X, Y) in the room frame, not local (Depth, Side).
If you are facing a wall and go forward, you will not be able to move forward, your position will not change.
Your starting absolute position is {startpos}. East is the direction of increasing position numbers for first dimension, and South is the direction of increasing position numbers for second dimension.

{tools_prompt if tools_prompt else ""}
{hint_string if hint else ""}
{corruption_warning if corruption_warning else ""}
### OBSERVATION LOG ###
{obsstring}

### YOUR RECONSTRUCTION ###
First, reason step-by-step about your position and what you see.
Finally, provide the final map summary in a new line, wrapped in "&&&" delimiters. For example:
&&&
{answer}
&&& 
"""
    