import imageio
import random
import numpy as np
from .constants import COMMAND_TO_ACTION
from .noise import _gauss_skip_count, _noisify_desc_per_flag
from .describe import describe_observation
from .common import get_position_as_tuple

def run_scripted_playthrough(
    env,
    command_sequence: list[str],
    picfilename: str = "1Dtrace.gif",
    dir: bool = True,
    oneDim: bool = True,
    failure_prob: float = 0.0,
    blackout: bool = False,
    stale_obs_prob: float = 0.0,
    debug: bool = True,
    *,
    # 新增：路径可视化 / 返回
    mark_path: bool = True,
    return_path: bool = False,
    path_color: tuple[int, int, int] = (255, 0, 0),
    path_alpha: float = 0.35,
    list_unseen: bool = False,
    list_empty: bool = False
) -> tuple:
    """
    若 return_path=False（默认），返回 (start_pos, obs_text)
    若 return_path=True，返回 (start_pos, obs_text, actual_path)
    actual_path 为按时间顺序的坐标列表[(x0,y0),(x1,y1),...]
    """
    print(f"stale_obs_prob: {stale_obs_prob}, failure_prob: {failure_prob}, blackout: {blackout}")

    def dprint(*a, **kw):
        if debug:
            print(*a, **kw)

    # --- 小工具：把路径画到一帧上 ---
    def _overlay_path_on_frame(frame: np.ndarray,
                               path_cells: list[tuple[int, int]],
                               grid_w: int, grid_h: int,
                               color=(255, 0, 0), alpha=0.35) -> np.ndarray:
        if not path_cells:
            return frame
        out = frame.copy()
        H, W = out.shape[0], out.shape[1]
        tile_h = H // grid_h
        tile_w = W // grid_w
        r, g, b = color
        # 半透明叠加：new = alpha*color + (1-alpha)*old
        for (cx, cy) in path_cells:
            x0, x1 = cx * tile_w, (cx + 1) * tile_w
            y0, y1 = cy * tile_h, (cy + 1) * tile_h
            # 防御：坐标越界直接跳过
            if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
                continue
            patch = out[y0:y1, x0:x1, :]
            # 做浮点再回写，避免溢出
            patch_float = patch.astype(np.float32)
            patch_float[..., 0] = alpha * r + (1 - alpha) * patch_float[..., 0]
            patch_float[..., 1] = alpha * g + (1 - alpha) * patch_float[..., 1]
            patch_float[..., 2] = alpha * b + (1 - alpha) * patch_float[..., 2]
            out[y0:y1, x0:x1, :] = np.clip(patch_float, 0, 255).astype(patch.dtype)
        return out

    # --- 初始化观测 / 起点 ---
    obs_str = []
    frames = []
    obs = env.unwrapped.gen_obs()
    start_pos = get_position_as_tuple(env.unwrapped.agent_pos)
    init_desc = describe_observation(obs, walls=False, oneDim=oneDim, dir=True, list_unseen=list_unseen, list_empty=list_empty)
    obs_str.append(f"Initial Observation: {init_desc}")

    # 路径跟踪
    grid_w, grid_h = env.unwrapped.width, env.unwrapped.height
    actual_path: list[tuple[int, int]] = [start_pos]  # 按时间顺序
    last_pos_tuple = start_pos

    # 初始帧
    frame = env.render()
    if mark_path:
        frame = _overlay_path_on_frame(frame, actual_path, grid_w, grid_h,
                                       color=path_color, alpha=path_alpha)
    frames.append(frame)

    step_counter = 1
    last_successful_obs = obs
    last_successful_frame = frame

    # --- blackout 窗口 ---
    blackout_start_index, blackout_end_index = -1, -1
    seq_len = len(command_sequence)
    if blackout and seq_len >= 12:
        blackout_start_index = random.randint(seq_len // 2, max(seq_len - 10, seq_len // 2 + 1))
        blackout_duration = random.randint(5, 10)
        blackout_end_index = min(seq_len, blackout_start_index + blackout_duration)
        dprint(f"[BLACKOUT] window = [{blackout_start_index}, {blackout_end_index}) (len={blackout_end_index-blackout_start_index})")
    else:
        blackout = False

    i = 0
    while i < seq_len:
        cmd = command_sequence[i].lower()

        # ===== 故障插入 =====
        pending_skip_count = 0
        if random.random() < failure_prob:
            if random.random() < 0.5:
                # 幽灵移动：只写日志，不动
                ghost_cmd = random.choice(["forward", "left", "right"])
                ghost_desc = describe_observation(last_successful_obs, walls=True, oneDim=oneDim, dir=dir, list_unseen=list_unseen, list_empty=list_empty)
                ghost_desc, changes = _noisify_desc_per_flag(ghost_desc, stale_obs_prob, sigma=1.0)
                obs_str.append(f"Step{step_counter}: {ghost_cmd}. {ghost_desc}")
                frames.append(last_successful_frame if not mark_path else
                              _overlay_path_on_frame(last_successful_frame, actual_path, grid_w, grid_h,
                                                     color=path_color, alpha=path_alpha))
                dprint(f"[GHOST] i={i}, ghost_cmd={ghost_cmd}, changes={changes if changes else 'none'}")
                step_counter += 1
            else:
                # 多步合并：本条之后额外执行后续 1~2 条但不记
                pending_skip_count = _gauss_skip_count(mu=1.0, sigma=0.6, lo=1, hi=2)
                dprint(f"[MERGE] i={i}, will silently execute next {pending_skip_count} cmd(s)")

        # ===== 解析动作 =====
        if cmd == 'turnaround':
            actions_to_execute = ['left', 'left']
        elif cmd in COMMAND_TO_ACTION:
            actions_to_execute = [cmd]
        else:
            dprint(f"[WARN] invalid command '{cmd}', skip")
            i += 1
            continue

        # ===== 执行当前指令（真实动作）=====
        final_obs_for_this_step = obs
        current_frame = last_successful_frame
        for sub_action_name in actions_to_execute:
            action_id = COMMAND_TO_ACTION[sub_action_name]
            obs, reward, terminated, truncated, info = env.step(action_id)
            final_obs_for_this_step = obs
            current_frame = env.render()

            # 路径记录：仅当位置发生变化时追加
            cur_pos_tuple = get_position_as_tuple(env.unwrapped.agent_pos)
            if cur_pos_tuple != last_pos_tuple:
                actual_path.append(cur_pos_tuple)
                last_pos_tuple = cur_pos_tuple

        # ===== blackout：执行但不记文本 =====
        if blackout and blackout_start_index <= i < blackout_end_index:
            dprint(f"[BLACKOUT] skip logging at i={i} (cmd='{cmd}')")
            last_successful_obs = final_obs_for_this_step
            last_successful_frame = current_frame

            # 在 blackout 期间的帧也可选择画路径（保持 GIF 连贯）
            if mark_path:
                frames.append(_overlay_path_on_frame(current_frame, actual_path, grid_w, grid_h,
                                                     color=path_color, alpha=path_alpha))
            else:
                frames.append(current_frame)

            # 合并执行的后续指令：执行并记录路径，但不写日志
            skip_applied = 0
            while pending_skip_count > 0 and (i + 1 + skip_applied) < seq_len:
                j = i + 1 + skip_applied
                next_cmd = command_sequence[j].lower()
                sublist = ['left','left'] if next_cmd == 'turnaround' \
                         else ([next_cmd] if next_cmd in COMMAND_TO_ACTION else [])
                for sub in sublist:
                    action_id = COMMAND_TO_ACTION[sub]
                    obs, reward, terminated, truncated, info = env.step(action_id)
                    last_successful_obs = obs
                    last_successful_frame = env.render()
                    # 路径追加
                    cur_pos_tuple = get_position_as_tuple(env.unwrapped.agent_pos)
                    if cur_pos_tuple != last_pos_tuple:
                        actual_path.append(cur_pos_tuple)
                        last_pos_tuple = cur_pos_tuple
                    # 也把帧加入，使 GIF 不“跳帧”
                    frame_to_add = last_successful_frame
                    if mark_path:
                        frame_to_add = _overlay_path_on_frame(frame_to_add, actual_path, grid_w, grid_h,
                                                              color=path_color, alpha=path_alpha)
                    frames.append(frame_to_add)

                dprint(f"[MERGE] silently executed cmd index={j}, cmd='{next_cmd}' (in blackout)")
                pending_skip_count -= 1
                skip_applied += 1

            i += 1 + skip_applied
            continue

        # ===== 正常记录文本与帧 =====
        desc = describe_observation(final_obs_for_this_step, walls=True, oneDim=oneDim, dir=dir, list_unseen=list_unseen, list_empty=list_empty)
        desc, changes = _noisify_desc_per_flag(desc, stale_obs_prob, sigma=1.0)
        if changes:
            dprint(f"[STALE] i={i}, cmd='{cmd}', per-flag jitters={changes}")

        obs_str.append(f"Step{step_counter}: {cmd}. {desc}")

        # 帧加入（可选路径叠加）
        frame_to_add = current_frame
        if mark_path:
            frame_to_add = _overlay_path_on_frame(frame_to_add, actual_path, grid_w, grid_h,
                                                  color=path_color, alpha=path_alpha)
        frames.append(frame_to_add)

        # 更新“上一真实状态”
        last_successful_obs = final_obs_for_this_step
        last_successful_frame = current_frame
        step_counter += 1

        # ===== 若为“多步合并”，执行后续 1~2 条但不记文本 =====
        skip_applied = 0
        while pending_skip_count > 0 and (i + 1 + skip_applied) < seq_len:
            j = i + 1 + skip_applied
            next_cmd = command_sequence[j].lower()
            sublist = ['left','left'] if next_cmd == 'turnaround' \
                     else ([next_cmd] if next_cmd in COMMAND_TO_ACTION else [])
            for sub in sublist:
                action_id = COMMAND_TO_ACTION[sub]
                obs, reward, terminated, truncated, info = env.step(action_id)
                last_successful_obs = obs
                last_successful_frame = env.render()
                # 路径追加
                cur_pos_tuple = get_position_as_tuple(env.unwrapped.agent_pos)
                if cur_pos_tuple != last_pos_tuple:
                    actual_path.append(cur_pos_tuple)
                    last_pos_tuple = cur_pos_tuple
                # 也把帧加入，使 GIF 连贯
                frame_to_add = last_successful_frame
                if mark_path:
                    frame_to_add = _overlay_path_on_frame(frame_to_add, actual_path, grid_w, grid_h,
                                                          color=path_color, alpha=path_alpha)
                frames.append(frame_to_add)

            dprint(f"[MERGE] silently executed cmd index={j}, cmd='{next_cmd}'")
            pending_skip_count -= 1
            skip_applied += 1

        i += 1 + skip_applied

    # --- 输出 GIF ---
    imageio.mimsave(picfilename, frames, fps=3)

    # --- 返回 ---
    obs_text = ".\n".join(obs_str)
    if return_path:
        return start_pos, obs_text, actual_path
    else:
        return start_pos, obs_text
