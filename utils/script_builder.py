import random
from .common import get_position_as_tuple
# 如果你的 ExplorationDebugger / MultiRoomPlanner 在你自己的模块里，请改成相对导入
from .trajectory import ExplorationDebugger  # 保持与你现有代码一致

def build_script(env_name: str, length: int=5, agent_view_size: int=5,
                 easy: bool=True, oneD: bool=True,
                 door_pos: tuple[int, int]|None=None,
                 agent_start_pos: tuple[int, int]|None=None,
                 agent_start_dir: int=0) -> list[str]:
    # ——把你原来的函数体粘到这里；逻辑保持不变——
    print(f"oneD: {oneD}, easy: {easy}, length: {length}, agent_view_size: {agent_view_size}")
    if oneD:
        if easy:
            go_forward_steps = ['forward'] * (length-1)
            turn_around_step = ['turnaround']
            go_back_steps = ['forward'] * (length-1)
            script = go_forward_steps + turn_around_step + go_back_steps
        else:
            turn_around_step = ['turnaround']
            go_forward_steps1 = ['forward'] * (random.randint(length//2, length-1))
            go_back_steps1 = ['forward'] * (random.randint(0, length-1))
            go_forward_steps2 = ['forward'] * (random.randint(0, length-1))
            go_back_steps2 = ['forward'] * (random.randint(length//2, length-1))
            script = (go_forward_steps1 + turn_around_step + go_back_steps1
                      + turn_around_step + go_forward_steps2 + turn_around_step + go_back_steps2)
    else:
        if easy:
            go_forward_steps = ['forward'] * (length-3)
            turn_right_step = ['right']
            script = go_forward_steps + turn_right_step + go_forward_steps + turn_right_step + go_forward_steps + turn_right_step + go_forward_steps
        elif env_name == "MiniGrid-CustomEmpty-5x5-v0":
            debugger = ExplorationDebugger(
                width=length, height=length, fov_depth=agent_view_size, fov_width=agent_view_size,
                use_distance_penalty=False, strategy='frontier'
            )
            script = debugger.run_exploration(start_pos=agent_start_pos, start_dir_idx=agent_start_dir)

    return script
