from __future__ import annotations

import math
import random
from collections import deque
from typing import Iterable, List, Optional, Set, Tuple

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Wall
from minigrid.minigrid_env import MiniGridEnv

# 方向：0=东，1=南，2=西，3=北
DIRS: List[Tuple[int, int]] = [(1, 0), (0, 1), (-1, 0), (0, -1)]


# ------------------------------
# 基础：脚本展开与路径模拟
# ------------------------------
def _expand_script(script: Iterable[str]) -> List[str]:
    out: List[str] = []
    for cmd in script:
        c = cmd.lower()
        if c == "turnaround":
            out.extend(["left", "left"])
        elif c in ("left", "right", "forward"):
            out.append(c)
        # 其他动作不影响“开路”，忽略
    return out


def _simulate_path(
    start_pos: Tuple[int, int],
    start_dir: int,
    script: Iterable[str],
    width: int,
    height: int,
) -> tuple[Set[Tuple[int, int]], List[Tuple[int, int]], List[int]]:
    """
    返回：
      - path_set：走过的格子集合（含起点）
      - path_seq：按行走顺序的格子序列（含起点与转向姿态节点）
      - dir_seq ：与 path_seq 对齐的朝向序列
    """
    path_set: Set[Tuple[int, int]] = set()
    path_seq: List[Tuple[int, int]] = []
    dir_seq: List[int] = []

    x, y = start_pos
    d = start_dir % 4

    def record():
        path_set.add((x, y))
        path_seq.append((x, y))
        dir_seq.append(d)

    record()  # 起点

    for cmd in _expand_script(script):
        if cmd == "left":
            d = (d - 1) % 4
            record()  # 记录姿态节点（位置不变，朝向变化）
        elif cmd == "right":
            d = (d + 1) % 4
            record()
        elif cmd == "forward":
            dx, dy = DIRS[d]
            nx, ny = x + dx, y + dy
            # 只在内层移动
            if 1 <= nx < width - 1 and 1 <= ny < height - 1:
                x, y = nx, ny
                record()
            # 越界则丢弃该步
    return path_set, path_seq, dir_seq


# ------------------------------
# 宽度曲线：低频起伏
# ------------------------------
def _moving_average(vals: List[float], k: int) -> List[float]:
    if k <= 1:
        return vals[:]
    n = len(vals)
    pre = [0.0]
    for v in vals:
        pre.append(pre[-1] + v)
    out: List[float] = []
    half = k // 2
    for i in range(n):
        L = max(0, i - half)
        R = min(n, i + half + 1)
        out.append((pre[R] - pre[L]) / (R - L))
    return out


def _widen_by_width_curve(
    base_open: Set[Tuple[int, int]],
    path_seq: List[Tuple[int, int]],
    dir_seq: List[int],
    width: int,
    height: int,
    *,
    base_width: float = 0.0,
    noise_amp: float = 1.0,
    max_band: int = 1,
    smooth_win: int = 7,
    corner_shrink: float = 0.6,
    prob_two_sided: float = 0.25,
    prob_left_when_one: float = 0.5,
    target_fill_ratio: Optional[float] = 0.22,
    rng: Optional[random.Random] = None,
) -> Set[Tuple[int, int]]:
    """
    稳定拓宽：不做 0/1 量化；为每个中心点生成 ceil(width_f) 层法向候选，
    用分数排序后加入，直到达到目标占用率。
    """
    rng = rng or random

    def in_bounds(px: int, py: int) -> bool:
        return 1 <= px < width - 1 and 1 <= py < height - 1

    n = len(path_seq)
    if n == 0:
        return set(base_open)

    # 1) 低频半宽
    white = [rng.uniform(-1.0, 1.0) for _ in range(n)]
    smooth = _moving_average(white, smooth_win)
    widths_f = [max(0.0, base_width + noise_amp * v) for v in smooth]

    # 2) 拐角收敛
    for i in range(1, n - 1):
        if dir_seq[i - 1] != dir_seq[i + 1]:
            widths_f[i] *= corner_shrink

    # 3) 收集候选（按优先级排序）
    open_cells = set(base_open)
    candidates: List[Tuple[float, Tuple[int, int]]] = []

    for i, (x, y) in enumerate(path_seq):
        # 跳过“仅转向”的重复坐标
        if i > 0 and path_seq[i] == path_seq[i - 1]:
            continue

        d = dir_seq[i]
        w = widths_f[i]
        if w <= 0.0:
            continue

        W = min(max_band, int(math.ceil(w)))  # 层数上限

        # 单/双侧
        two = (rng.random() < prob_two_sided)
        sides = [-1, +1] if two else ([-1] if rng.random() < prob_left_when_one else [+1])

        # 分数：w 为主，tie-break 用很小的 -i 和微噪声，前段略优先
        base_score = w
        tie_break = -1e-6 * i
        for side in sides:
            sx, sy = DIRS[(d - 1) % 4] if side < 0 else DIRS[(d + 1) % 4]
            for bw in range(1, W + 1):
                wx, wy = x + sx * bw, y + sy * bw
                if in_bounds(wx, wy) and (wx, wy) not in open_cells:
                    score = base_score + tie_break + 1e-8 * rng.random()
                    candidates.append((score, (wx, wy)))

    if not candidates:
        return open_cells

    candidates.sort(reverse=True)

    # 4) 加到目标为止（若未设置目标则全加）
    if target_fill_ratio is None:
        for _, cell in candidates:
            open_cells.add(cell)
        return open_cells

    inner_total = (width - 2) * (height - 2)
    cap = max(len(base_open), int(target_fill_ratio * inner_total))
    room = cap - len(open_cells)
    if room <= 0:
        return open_cells

    for _, cell in candidates:
        if room <= 0:
            break
        if cell not in open_cells:
            open_cells.add(cell)
            room -= 1
    return open_cells


# ------------------------------
# 迭代填充：≥3 空邻的墙改为空
# ------------------------------
def _iterative_wall_fill(
    open_cells: Set[Tuple[int, int]],
    width: int,
    height: int,
    *,
    min_open_neighbors: int = 3,
    cap: Optional[int] = None,
) -> Set[Tuple[int, int]]:
    """
    把“墙且四邻中空格数 ≥ min_open_neighbors”的格子改为空格，并迭代传播。
    cap 用于限制最终总空格数（可选）。
    """

    def in_bounds(px: int, py: int) -> bool:
        return 1 <= px < width - 1 and 1 <= py < height - 1

    def neighbors4(px: int, py: int):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            qx, qy = px + dx, py + dy
            if in_bounds(qx, qy):
                yield (qx, qy)

    def open_count(px: int, py: int) -> int:
        return sum((qx, qy) in open_cells for (qx, qy) in neighbors4(px, py))

    q: deque[Tuple[int, int]] = deque()

    # 初始队列：所有满足条件的墙
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            if (x, y) not in open_cells and open_count(x, y) >= min_open_neighbors:
                q.append((x, y))

    # 迭代传播
    while q:
        x, y = q.popleft()
        if (x, y) in open_cells:
            continue
        if open_count(x, y) >= min_open_neighbors:
            if cap is not None and len(open_cells) >= cap:
                break
            open_cells.add((x, y))
            for nb in neighbors4(x, y):
                if nb not in open_cells:
                    q.append(nb)

    return open_cells

def _segments_from_path(path_seq, dir_seq):
    """
    把 (path_seq, dir_seq) 切成若干段，每段朝向不变且相邻格子相差 1。
    返回: [(cells, d), ...]，cells 是该段的坐标列表。
    """
    if not path_seq:
        return []

    segs = []
    cur_cells = [path_seq[0]]
    cur_dir = dir_seq[0]

    for i in range(1, len(path_seq)):
        p_prev, p = path_seq[i-1], path_seq[i]
        d_prev, d = dir_seq[i-1], dir_seq[i]

        # 只把“真正前进”的格子算进段里（转向时位置不动，会被这里自动打断）
        is_step = (abs(p[0]-p_prev[0]) + abs(p[1]-p_prev[1]) == 1)
        if is_step and d == cur_dir:
            cur_cells.append(p)
        else:
            if len(cur_cells) >= 1:
                segs.append((cur_cells, cur_dir))
            cur_cells = [p]
            cur_dir = d

    if len(cur_cells) >= 1:
        segs.append((cur_cells, cur_dir))
    return segs
import math
import random

DIRS = [(1,0),(0,1),(-1,0),(0,-1)]  # E,S,W,N

def _widen_by_segments(
    base_open: set[tuple[int,int]],
    path_seq: list[tuple[int,int]],
    dir_seq: list[int],
    width: int,
    height: int,
    *,
    prob_left: float = 0.25,
    prob_right: float = 0.25,
    band: int = 1,            # 每侧最多拓一格
    end_trim: int = 1,        # 端点收缩，避免在拐角处鼓包
    skip_prob: float = 0.0,   # 段内随机跳过一些点，产生轻微不规则
    rng: random.Random | None = None,
) -> set[tuple[int,int]]:
    """
    对每一段整体决定是否向左/向右拓宽，再把该段的中间部分整体外扩 band 层。
    """
    rng = rng or random
    open_cells = set(base_open)

    def in_bounds(x,y):
        return 1 <= x < width-1 and 1 <= y < height-1

    segs = _segments_from_path(path_seq, dir_seq)
    for cells, d in segs:
        if len(cells) <= 1:
            continue

        # 段两端收缩，避免在转角处外鼓
        L = max(0, end_trim)
        R = max(0, end_trim)
        work = cells[L: len(cells)-R] if len(cells) > (L+R) else []

        if not work:
            continue

        # 该段一次性抽样“拓左/拓右”
        do_left  = (rng.random() < prob_left)
        do_right = (rng.random() < prob_right)

        if not (do_left or do_right):
            continue

        # 法向
        left_vec  = DIRS[(d-1) % 4]
        right_vec = DIRS[(d+1) % 4]

        for x,y in work:
            if skip_prob > 0.0 and rng.random() < skip_prob:
                continue
            if do_left:
                for k in range(1, band+1):
                    nx, ny = x + left_vec[0]*k, y + left_vec[1]*k
                    if in_bounds(nx, ny):
                        open_cells.add((nx, ny))
            if do_right:
                for k in range(1, band+1):
                    nx, ny = x + right_vec[0]*k, y + right_vec[1]*k
                    if in_bounds(nx, ny):
                        open_cells.add((nx, ny))

    return open_cells

# ------------------------------
# 环境：按脚本开路 + 拓宽 + 迭代填充
# ------------------------------
class CarvedPathRoomEnv(MiniGridEnv):
    """
    与 CustomEmptyEnv 同尺寸，有外墙；按脚本在内层开路。
    然后用宽度曲线做稳定拓宽，并可做“≥3 空邻转空”的迭代填充。
    无物体。
    """

    def __init__(
        self,
        length: int = 7,
        agent_view_size: int = 5,
        carve_script: Optional[List[str]] = None,
        start_pos: Optional[Tuple[int, int]] = None,
        start_dir: Optional[int] = None,
        max_steps: Optional[int] = None,
        *,
        # 拓宽参数
        widen_mode: str = "segments",   # "segments" 或 "curve"

        widen_enable: bool = True,
        widen_base_width: float = 0.0,
        widen_noise_amp: float = 1.0,
        widen_band_width: int = 1,
        widen_smooth_win: int = 7,
        widen_corner_shrink: float = 0.6,
        widen_prob_two_sided: float = 0.25,
        widen_prob_left_when_one: float = 0.5,
        widen_target_ratio: Optional[float] = 0.22,
        seg_prob_left: float = 0.25,
        seg_prob_right: float = 0.25,
        seg_band: int = 1,
        seg_end_trim: int = 1,
        seg_skip_prob: float = 0.0,        post_fill_enable: bool = True,
        post_fill_min_open_neighbors: int = 3,
        post_fill_keep_ratio: Optional[float] = None,  # 若限制最终占用率，设为比例；否则 None
        # 随机种子（可复现）
        rng_seed: Optional[int] = None,
        **kwargs,
    ):
        self.room_size = length
        self._carve_script = carve_script
        self._given_start_pos = start_pos
        self._given_start_dir = start_dir
        self._widen_mode = widen_mode

        # 拓宽设置
        self._widen_enable = widen_enable
        self._widen_base_width = widen_base_width
        self._widen_noise_amp = widen_noise_amp
        self._widen_band_width = widen_band_width
        self._widen_smooth_win = widen_smooth_win
        self._widen_corner_shrink = widen_corner_shrink
        self._widen_prob_two_sided = widen_prob_two_sided
        self._widen_prob_left_when_one = widen_prob_left_when_one
        self._widen_target_ratio = widen_target_ratio

        self._seg_prob_left = seg_prob_left
        self._seg_prob_right = seg_prob_right
        self._seg_band = seg_band
        self._seg_end_trim = seg_end_trim
        self._seg_skip_prob = seg_skip_prob
        # 迭代填充设置
        self._post_fill_enable = post_fill_enable
        self._post_fill_min_open_neighbors = post_fill_min_open_neighbors
        self._post_fill_keep_ratio = post_fill_keep_ratio

        # 随机种子
        self._rng_seed = rng_seed

        if max_steps is None:
            max_steps = 8 * length * length

        mission_space = MissionSpace(mission_func=lambda: "follow the carved corridor")

        super().__init__(
            mission_space=mission_space,
            width=length,
            height=length,
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            see_through_walls=False,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        # 外墙
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # 起点与朝向（未给定则随机）
        if self._given_start_pos is None:
            sx = self._rand_int(1, width - 1)
            sy = self._rand_int(1, height - 1)
            start_pos = (sx, sy)
        else:
            start_pos = self._given_start_pos
            if not (1 <= start_pos[0] < width - 1 and 1 <= start_pos[1] < height - 1):
                raise ValueError(f"start_pos {start_pos} out of inner bounds")

        start_dir = self._rand_int(0, 4) if self._given_start_dir is None else (self._given_start_dir % 4)

        # 开路路径（若没给脚本，则只开起点）
        script = self._carve_script or []
        corridor_base, path_seq, dir_seq = _simulate_path(start_pos, start_dir, script, width, height)

        # 拓宽（稳定版）
        corridor_open = set(corridor_base)
        if len(path_seq) > 0:
            rng = random.Random(self._rng_seed) if self._rng_seed is not None else None
            if self._widen_mode == "segments":
                corridor_open = _widen_by_segments(
                    base_open=corridor_base,
                    path_seq=path_seq,
                    dir_seq=dir_seq,
                    width=width,
                    height=height,
                    prob_left=self._seg_prob_left,
                    prob_right=self._seg_prob_right,
                    band=self._seg_band,
                    end_trim=self._seg_end_trim,
                    skip_prob=self._seg_skip_prob,
                    rng=rng,
                )
            else:
                corridor_open = _widen_by_width_curve(
                    base_open=corridor_base,
                    path_seq=path_seq,
                    dir_seq=dir_seq,
                    width=width,
                    height=height,
                    base_width=self._widen_base_width,
                    noise_amp=self._widen_noise_amp,
                    max_band=self._widen_band_width,
                    smooth_win=self._widen_smooth_win,
                    corner_shrink=self._widen_corner_shrink,
                    prob_two_sided=self._widen_prob_two_sided,
                    prob_left_when_one=self._widen_prob_left_when_one,
                    target_fill_ratio=self._widen_target_ratio,
                    rng=rng,
                )


        # 迭代填充（≥3 空邻转空）
        if self._post_fill_enable:
            cap = None
            if self._post_fill_keep_ratio is not None:
                inner_total = (width - 2) * (height - 2)
                cap = max(len(corridor_base), int(self._post_fill_keep_ratio * inner_total))
            corridor_open = _iterative_wall_fill(
                open_cells=corridor_open,
                width=width,
                height=height,
                min_open_neighbors=self._post_fill_min_open_neighbors,
                cap=cap,
            )

        # 写回网格：先全墙，再把开路单元清空
        for ix in range(1, width - 1):
            for iy in range(1, height - 1):
                self.grid.set(ix, iy, Wall())
        for (ix, iy) in corridor_open:
            self.grid.set(ix, iy, None)

        # 放置智能体（与脚本一致）
        self.place_agent(top=start_pos, size=(1, 1), rand_dir=False)
        self.agent_dir = start_dir

        self.mission = "follow the carved corridor"
