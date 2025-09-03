
# a 1*n corridor

from __future__ import annotations
import gymnasium as gym
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Goal, Ball, Key, Wall
from minigrid.core.mission import MissionSpace
from minigrid.core.actions import Actions
from gymnasium.envs.registration import register
import numpy as np
from collections import defaultdict

IDX_TO_OBJECT = {
    0: 'unseen',
    1: 'empty',
    2: 'wall',
    3: 'floor',
    4: 'door',
    5: 'key',
    6: 'ball',
    7: 'box',
    8: 'flag',#goal
    9: 'lava',
    10: 'agent',
}

IDX_TO_COLOR = {
    0: 'red',
    1: 'green',
    2: 'blue',
    3: 'purple',
    4: 'yellow',
    5: 'grey',
}

IDX_TO_STATE = {
    0: 'open',
    1: 'closed',
    2: 'locked',
}

COMMAND_TO_ACTION = {
    "left": Actions.left,
    "right": Actions.right,
    "forward": Actions.forward,
    "pickup": Actions.pickup,
    "drop": Actions.drop,
    "toggle": Actions.toggle,
    "done": Actions.done
}


def describe_observation(obs, walls=True, oneDim=False, dir=False) -> str:
    direction_text = ""
    if dir:
        direction_map = {0: 'East', 1: 'South', 2: 'West', 3: 'North'}
        direction = direction_map.get(obs['direction'], 'unknown')
        direction_text = f"You're facing {direction}."

    image = obs['image']
    grouped_objects = defaultdict(list)
    object_grid, color_grid, state_grid = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            obj_id = object_grid[y, x]
            if obj_id > 1: # 忽略 empty 和 unseen
                obj_name = IDX_TO_OBJECT.get(obj_id)
                if not walls and obj_name == 'wall':
                    continue # 根据参数决定是否忽略墙
                
                color_id, state_id = color_grid[y, x], state_grid[y, x]
                color_name = IDX_TO_COLOR.get(color_id, 'a')
                object_key = (color_name, obj_name)
                if obj_name == 'door':
                    state_name = IDX_TO_STATE.get(state_id, 'unknown_state')
                    object_key = (color_name, obj_name, state_name)
                grouped_objects[object_key].append((x, y))

    object_descriptions = generate_object_descriptions(grouped_objects, image.shape, oneDim)

    if not object_descriptions:
        return f"{direction_text} You see nothing of interest."
    else:
        return f"{direction_text} You can see " + ", ".join(object_descriptions)

def generate_object_descriptions(grouped_objects, view_shape, oneDim=False):

    object_descriptions = []
    view_height, view_width = view_shape[0], view_shape[1]
    
    # AI的中心线 x 坐标是 view_width 的一半 (整数除法)
    agent_center_x = view_width // 2
    
    for object_key, coords in grouped_objects.items():
        color_name, obj_name = object_key[0], object_key[1]
        
        # --- 新的坐标转换逻辑 ---
        coord_strings = []
        for x, y in coords:
            # 深度 = 离AI有多远 (AI在最下面一行，即 view_height - 1)
            depth = (view_height - 1)-x
            
            # 左右偏移 = 离中心线有多远
            side_offset = y - agent_center_x
            
            if oneDim:
                # 如果是一维模式，只报告在中心线上的物体 (side_offset == 0)
                if side_offset == 0:
                    coord_strings.append(str(depth))
            else:
                # 二维模式，报告 (深度, 左右偏移)
                coord_strings.append(f"({depth}, {side_offset})")
        
        if not coord_strings:
            continue # 如果过滤后没有坐标了，就跳过这个物体

        coords_str = ", ".join(coord_strings)
        
        # --- 描述生成逻辑 (保持不变) ---
        if len(coord_strings) == 1:
            description = f"a {color_name} {obj_name} at coordinate {coords_str}"
        else:
            plural_obj_name = obj_name + 's'
            description = f"{color_name} {plural_obj_name} at coordinates: {coords_str}"
        
        if obj_name == 'door':
            description += f" It is {object_key[2]}."
            
        object_descriptions.append(description)
        
    return object_descriptions


import imageio
import random
from Minigrid.utils.trajectory import ExplorationDebugger, MultiRoomPlanner
def build_script(env_name: str, length: int=5, agent_view_size: int=5, easy : bool=True, oneD:bool=True, door_pos: tuple[int, int]=None, agent_start_pos: tuple[int, int]=None, agent_start_dir: int=0) -> list[str]:
    print(f"oneD: {oneD}, easy: {easy}, length: {length}, agent_view_size: {agent_view_size}")
    if oneD:
        if easy:
            go_forward_steps = ['forward'] * (length-1)
            turn_around_step = ['turnaround']
            go_back_steps = ['forward'] * (length-1)
            script = go_forward_steps + turn_around_step + go_back_steps  
        else:
            turn_around_step = ['turnaround']
            go_forward_steps1 = ['forward'] * (random.randint(length/2, length-1))
            go_back_steps1 = ['forward'] * (random.randint(0, length-1))
            go_forward_steps2 = ['forward'] * (random.randint(0, length-1))
            go_back_steps2 = ['forward'] * (random.randint(length/2, length-1))
            script = go_forward_steps1 + turn_around_step + go_back_steps1+ turn_around_step + go_forward_steps2 + turn_around_step + go_back_steps2
    else: # 为MiniGrid-Empty-5x5-v0
        if easy:
            go_forward_steps = ['forward'] * (length-3)
            turn_right_step = ['right']
            script = go_forward_steps + turn_right_step + go_forward_steps + turn_right_step + go_forward_steps + turn_right_step + go_forward_steps  
        elif env_name == "MiniGrid-CustomEmpty-5x5-v0":
            debugger = ExplorationDebugger(
            width=length, 
            height=length,
            fov_depth=agent_view_size,
            fov_width=agent_view_size ,
            use_distance_penalty=False,
            strategy= 'frontier'
        )
            script = debugger.run_exploration(start_pos=agent_start_pos, start_dir_idx=agent_start_dir)
            
        elif env_name == "MiniGrid-TwoRooms-v0":
            debugger = MultiRoomPlanner(
        room_size=length,
        door_pos=door_pos,
        fov_depth=agent_view_size,
        fov_width=agent_view_size,
        use_distance_penalty=True
    )
    

    # 运行探索，调试信息会自动打印
            script = debugger.run_exploration(agent_start_pos=agent_start_pos, agent_start_dir=agent_start_dir)
    return script

import random, re, imageio

def _gauss_skip_count(mu=1.0, sigma=0.6, lo=1, hi=2):
    """离散高斯取样：返回 1 或 2（1 的概率更大）"""
    while True:
        k = int(round(random.gauss(mu, sigma)))
        if lo <= k <= hi:
            return k

def _sample_offset_int(sigma=1.0, kmax=3):
    while True:
        k = int(round(random.gauss(0.0, sigma)))
        if k != 0:
            return max(-kmax, min(kmax, k))

def _noisify_desc_per_flag(desc, per_flag_prob, depth_range=(0,4), side_range=(-2,2), sigma=1.0):
    """
    对文本中的【每个】颜色旗坐标(Depth, Side)独立地以 per_flag_prob 的概率做离散高斯抖动。
    返回: (新文本, 改动清单list)，清单元素: (color, old_d, old_s, new_d, new_s)
    """
    changes = []

    pattern = re.compile(r"\(\s*([-+]?\d+)\s*,\s*([-+]?\d+)\s*\)")

    def repl(m):
        d0 = int(m.group(1)); s0 = int(m.group(2))
        d, s = d0, s0

        # depth 分量是否抖动
        if random.random() < per_flag_prob:
            d = d0 + _sample_offset_int(sigma=sigma, kmax=3)
            d = max(depth_range[0], min(depth_range[1], d))

        # side 分量是否抖动
        if random.random() < per_flag_prob:
            s = s0 + _sample_offset_int(sigma=sigma, kmax=3)
            s = max(side_range[0], min(side_range[1], s))

        if (d, s) != (d0, s0):
            changes.append(((d0, s0), (d, s)))
        return f"({d}, {s})"

    new_desc = pattern.sub(repl, desc)
    return new_desc, changes


def run_scripted_playthrough(
    env: MiniGridEnv, 
    command_sequence: list[str], 
    picfilename="1Dtrace.gif", 
    dir=True, 
    oneDim=True,
    failure_prob: float = 0.0,      # 故障概率：可能触发“幽灵移动”或“多步合并”
    blackout: bool=False,           # 是否启用日志黑洞（5-10 步）
    stale_obs_prob: float = 0.0,    # 观测坐标抖动概率（离散高斯偏移）
    debug: bool = True              # 打印所有故障细节
) -> tuple[tuple, str]:
    print(f"stale_obs_prob: {stale_obs_prob}, failure_prob: {failure_prob}, blackout: {blackout}")

    COMMAND_TO_ACTION = {
        "left": Actions.left, "right": Actions.right, "forward": Actions.forward,
        "pickup": Actions.pickup, "drop": Actions.drop, "toggle": Actions.toggle, "done": Actions.done
    }

    def dprint(*a, **kw):
        if debug:
            print(*a, **kw)

    obs_str = []
    observations_history = []
    frames = []

    obs = env.unwrapped.gen_obs()
    start_pos = get_position_as_tuple(env.unwrapped.agent_pos)

    # 初始观测不加噪，直接记录
    init_desc = describe_observation(obs, walls=False, oneDim=oneDim, dir=True)
    obs_str.append(f"Initial Observation: {init_desc}")
    observations_history.append(obs)
    frame = env.render()
    frames.append(frame)

    step_counter = 1
    last_successful_obs = obs
    last_successful_frame = frame

    # ---------- blackout 窗口（5~10 步） ----------
    blackout_start_index, blackout_end_index = -1, -1
    seq_len = len(command_sequence)
    if blackout and seq_len >= 12:
        blackout = True
        blackout_start_index = random.randint(seq_len // 2, max(seq_len - 10, seq_len // 2 + 1))
        blackout_duration = random.randint(5, 10)
        blackout_end_index = min(seq_len, blackout_start_index + blackout_duration)
        dprint(f"[BLACKOUT] window = [{blackout_start_index}, {blackout_end_index}) (len={blackout_end_index-blackout_start_index})")
    else:
        blackout = False

    failure_start_index = int(seq_len * 0.3)  # 30% 进度后才开始插入故障

    i = 0
    while i < seq_len:
        cmd = command_sequence[i].lower()

        # ===== 故障抽样：幽灵移动 / 多步合并（二选一） =====
        pending_skip_count = 0
        if random.random() < failure_prob:
            if random.random() < 0.5:
                # A) 幽灵移动：只写日志，不动
                ghost_cmd = random.choice(["forward", "left", "right"])
                ghost_desc = describe_observation(last_successful_obs, walls=True, oneDim=oneDim, dir=dir)
                ghost_desc, changes = _noisify_desc_per_flag(ghost_desc, stale_obs_prob, sigma=1.0)
                obs_str.append(f"Step{step_counter}: {ghost_cmd}. {ghost_desc}")
                observations_history.append(last_successful_obs)
                frames.append(last_successful_frame)
                dprint(f"[GHOST] i={i}, ghost_cmd={ghost_cmd}, changes={changes if changes else 'none'}")

                step_counter += 1
            else:
                # B) 多步合并：本条执行后，额外执行后续 1~2 条但不记录
                pending_skip_count = _gauss_skip_count(mu=1.0, sigma=0.6, lo=1, hi=2)
                dprint(f"[MERGE] i={i}, will silently execute next {pending_skip_count} cmd(s)")

        # ===== 执行当前指令（真实动作一定发生） =====
        actions_to_execute = []
        if cmd == 'turnaround':
            actions_to_execute = ['left', 'left']
        elif cmd in COMMAND_TO_ACTION:
            actions_to_execute = [cmd]
        else:
            dprint(f"[WARN] invalid command '{cmd}', skip")
            i += 1
            continue

        final_obs_for_this_step = obs
        current_frame = last_successful_frame
        for sub_action_name in actions_to_execute:
            action_id = COMMAND_TO_ACTION[sub_action_name]
            obs, reward, terminated, truncated, info = env.step(action_id)
            final_obs_for_this_step = obs
            current_frame = env.render()

        # ===== blackout：在窗口内不写日志（但动作都执行了）=====
        if blackout and blackout_start_index <= i < blackout_end_index:
            dprint(f"[BLACKOUT] skip logging at i={i} (cmd='{cmd}')")
            last_successful_obs = final_obs_for_this_step
            last_successful_frame = current_frame

            # 如果还有“多步合并”，把接下来的命令执行掉但不记
            skip_applied = 0
            while pending_skip_count > 0 and (i + 1 + skip_applied) < seq_len:
                j = i + 1 + skip_applied
                next_cmd = command_sequence[j].lower()
                sublist = ['left','left'] if next_cmd == 'turnaround' else ([next_cmd] if next_cmd in COMMAND_TO_ACTION else [])
                for sub in sublist:
                    action_id = COMMAND_TO_ACTION[sub]
                    obs, reward, terminated, truncated, info = env.step(action_id)
                    last_successful_obs = obs
                    last_successful_frame = env.render()
                dprint(f"[MERGE] silently executed cmd index={j}, cmd='{next_cmd}' (in blackout)")
                pending_skip_count -= 1
                skip_applied += 1

            i += 1 + skip_applied
            continue  # blackout 内不记录文本

        # ===== 正常记录当前步（可加“观测抖动”）=====
        desc = describe_observation(final_obs_for_this_step, walls=True, oneDim=oneDim, dir=dir)
        desc, changes = _noisify_desc_per_flag(desc, stale_obs_prob, sigma=1.0)
        if changes:
            dprint(f"[STALE] i={i}, cmd='{cmd}', per-flag jitters={changes}")

        observations_history.append(final_obs_for_this_step)
        frames.append(current_frame)
        obs_str.append(f"Step{step_counter}: {cmd}. {desc}")

        # 更新“上一真实状态”
        last_successful_obs = final_obs_for_this_step
        last_successful_frame = current_frame
        step_counter += 1

        # ===== 若为“多步合并”，执行后续 1~2 条但不记 =====
        skip_applied = 0
        while pending_skip_count > 0 and (i + 1 + skip_applied) < seq_len:
            j = i + 1 + skip_applied
            next_cmd = command_sequence[j].lower()
            sublist = ['left','left'] if next_cmd == 'turnaround' else ([next_cmd] if next_cmd in COMMAND_TO_ACTION else [])
            for sub in sublist:
                action_id = COMMAND_TO_ACTION[sub]
                obs, reward, terminated, truncated, info = env.step(action_id)
                last_successful_obs = obs
                last_successful_frame = env.render()
            dprint(f"[MERGE] silently executed cmd index={j}, cmd='{next_cmd}'")
            pending_skip_count -= 1
            skip_applied += 1

        i += 1 + skip_applied  # 跳过那些被“合并执行”的指令

    imageio.mimsave(picfilename, frames, fps=3)
    return start_pos, ".\n".join(obs_str)

# def run_scripted_playthrough(
#     env: MiniGridEnv, 
#     command_sequence: list[str], 
#     picfilename="1Dtrace.gif", 
#     dir=True, 
#     oneDim=True,
#     failure_prob: float = 0.0,      # “幽灵移动”的概率
#     blackout: bool=False,     # NEW: “日志黑洞”的触发概率
#     stale_obs_prob: float = 0.0     # NEW: “过时观测”的概率
# ) -> tuple[tuple, str]:

#     COMMAND_TO_ACTION = { "left": Actions.left, "right": Actions.right, "forward": Actions.forward, "pickup": Actions.pickup, "drop": Actions.drop, "toggle": Actions.toggle, "done": Actions.done }
#     obs_str = []
#     observations_history = []
#     frames = []
#     obs = env.unwrapped.gen_obs()
#     start_pos = get_position_as_tuple(env.unwrapped.agent_pos)
#     obs_str.append(f"Initial Observation: {describe_observation(obs, walls=False, oneDim=oneDim, dir=True)}")
#     observations_history.append(obs)
#     frame = env.render()
#     frames.append(frame)

#     step_counter = 1
#     last_successful_obs = obs
#     last_successful_frame = frame

#     # --- NEW: 日志黑洞功能的初始化 ---
#     blackout_start_index, blackout_end_index = -1, -1
#     seq_len = len(command_sequence)
#     #  playthrough开始时，一次性决定是否会发生“黑洞”
#     if blackout:
#         blackout = True
#         # 从后半段随机选一个起点
#         blackout_start_index = random.randint(seq_len // 2, seq_len - 10)
#         # 黑洞持续5到10步
#         blackout_duration = random.randint(5, 7)
#         blackout_end_index = blackout_start_index + blackout_duration
#         print(f"DEBUG: Log blackout will occur from command index {blackout_start_index} to {blackout_end_index}")
    
#     failure_start_index = int(seq_len * 0.3) # 从30%的进度开始引入故障
#     # MODIFIED: 使用 enumerate 以获取指令的索引 i
#     for i, cmd in enumerate(command_sequence):
#         cmd = cmd.lower()
        
#         # --- “幽灵移动”注入逻辑 (代码不变) ---
#         if i >= failure_start_index and random.random() < failure_prob:
#             print(f"DEBUG: Ghost command triggered for command {cmd}")
#             ghost_cmd = random.choice(["forward", "left", "right"])
#             obs_str.append(
#                 f"Step{step_counter}: {ghost_cmd}. {describe_observation(last_successful_obs, walls=True, oneDim=oneDim, dir=dir)}"
#             )
#             observations_history.append(last_successful_obs)
#             frames.append(last_successful_frame)
#             step_counter += 1
            
#         # --- 真实动作的执行 ---
#         actions_to_execute = []
#         if cmd == 'turnaround':
#             actions_to_execute = ['left', 'left']
#         elif cmd in COMMAND_TO_ACTION:
#             actions_to_execute = [cmd]
#         else:
#             print(f"Warning: Invalid command '{cmd}' found. Skipping.")
#             continue
            
#         final_obs_for_this_step = obs
#         current_frame = last_successful_frame
#         for sub_action_name in actions_to_execute:
#             action_id = COMMAND_TO_ACTION[sub_action_name]
#             obs, reward, terminated, truncated, info = env.step(action_id) # 机器人【总是】实际移动
#             final_obs_for_this_step = obs
#             current_frame = env.render() # 渲染出【真实的】新画面
#             if terminated or truncated: break

#         # --- NEW: 日志黑洞的核心逻辑 ---
#         if blackout and i >= blackout_start_index and i < blackout_end_index:
#             # 如果当前步骤在黑洞区间内
#             # 机器人已经移动了 (env.step已调用)，但我们不记录任何东西
#             print(f"DEBUG: In blackout, skipping log for command index {i}")
#             # 更新“上一步成功”的状态，以便黑洞结束后能正确衔接
#             last_successful_obs = final_obs_for_this_step
#             last_successful_frame = current_frame
#             continue # 直接跳到下一个循环，不执行下面的日志记录代码
            
#         # --- NEW: 过时观测的核心逻辑 ---
#         obs_to_log = final_obs_for_this_step
#         frame_to_log = current_frame
#         if  i >= failure_start_index and random.random() < stale_obs_prob:
#             print(f"DEBUG: Stale observation triggered for command {cmd}")
#             obs_to_log = last_successful_obs
#             frame_to_log = last_successful_frame
#             # 注意：机器人已经移动到了新位置，但日志欺骗了你

#         # --- MODIFIED: 使用可能被篡改过的数据进行日志记录 ---
#         observations_history.append(obs_to_log)
#         frames.append(frame_to_log)
#         obs_str.append(
#             f"Step{step_counter}: {cmd}. {describe_observation(obs_to_log, walls=True, oneDim=oneDim, dir=dir)}"
#         )
        
#         # 更新上一个成功的状态【总是】使用真实的新状态
#         last_successful_obs = final_obs_for_this_step
#         last_successful_frame = current_frame
#         step_counter += 1
#         # if step on the flags, continue instead of break
#         if terminated or truncated:
#             print("Episode ended before script finished.")
#             continue
            
#     imageio.mimsave(picfilename, frames, fps=3)
#     return start_pos, ".\n".join(obs_str)

def get_position_as_tuple(start_pos):
    return (int(start_pos[0]), int(start_pos[1]))

import logging
import os
def setup_logging(log_dir: str, log_filename: str = "evaluation_log.txt"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, log_filename)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file, mode='w'),
                            logging.StreamHandler()
                        ])
    