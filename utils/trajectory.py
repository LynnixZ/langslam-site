import random
from collections import deque
import math

import math
import random
from collections import deque

class ExplorationDebugger:
    """
    目标：
    1) 路径只拐一次：先直行，再一次左/右转，再直行到目标（L 形路径）。
    2) 途中不频繁转向；只在必要时转（起步对齐一次可选、拐角一次、到达后对齐观察方向一次）。
    """

    def __init__(self, width, height, fov_depth, fov_width, 
                 strategy: str = 'frontier', use_distance_penalty=True):
        self.width, self.height = width, height
        self.fov_depth, self.fov_width = fov_depth, fov_width
        self.strategy, self.use_distance_penalty = strategy, use_distance_penalty
        if self.strategy not in ['frontier', 'room_boundary']:
            raise ValueError("Strategy must be 'frontier' or 'room_boundary'")

        # NEW: 定义墙壁单元格集合（外边界）
        self.wall_cells = set()
        for x in range(width):
            self.wall_cells.add((x, 0))
            self.wall_cells.add((x, height - 1))
        for y in range(1, height - 1):
            self.wall_cells.add((0, y))
            self.wall_cells.add((width - 1, y))

        # MODIFIED: unseen_cells 现在是墙壁内部的可探索区域
        self.unseen_cells = set(
            (x, y) for x in range(1, width - 1) for y in range(1, height - 1)
        )
        
        self.script, self.robot_pos, self.robot_dir_idx = [], None, None

        # MODIFIED: 房间边界策略的目标点现在是内壁旁边的点
        self.room_boundary_points = set()
        for x in range(1, width - 1):
            self.room_boundary_points.add((x, 1))
            self.room_boundary_points.add((x, height - 2))
        for y in range(1, height - 1):
            self.room_boundary_points.add((1, y))
            self.room_boundary_points.add((width - 2, y))

        # 方向：0右(E)、1下(S)、2左(W)、3上(N)
        self.dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.dir_symbols = {0: '>', 1: 'v', 2: '<', 3: '^'}

    # ========== NEW: 基础动作 ==========
    def _turn_to(self, desired_idx):
        """最小转向：右/左/调头。更新 self.robot_dir_idx 并写入 script。"""
        diff = (desired_idx - self.robot_dir_idx) % 4
        if diff == 1:
            self.script.append('right'); self.robot_dir_idx = (self.robot_dir_idx + 1) % 4
        elif diff == 3:
            self.script.append('left');  self.robot_dir_idx = (self.robot_dir_idx - 1) % 4
        elif diff == 2:
            self.script.extend(['right','right']); self.robot_dir_idx = (self.robot_dir_idx + 2) % 4
        # diff == 0 不转

    def _forward_steps(self, steps):
        """前进 steps 步；每步更新位置与可见单元。"""
        for _ in range(max(0, int(steps))):
            fx, fy = self.dirs[self.robot_dir_idx]
            nx, ny = self.robot_pos[0] + fx, self.robot_pos[1] + fy
            # 保证不穿墙
            if 1 <= nx < self.width - 1 and 1 <= ny < self.height - 1:
                self.script.append('forward')
                self.robot_pos = (nx, ny)
                self.unseen_cells -= self._get_visible_cells(self.robot_pos, self.robot_dir_idx)
            else:
                break

    def _plan_L_path(self, start, end):
        """
        生成一条 L 形路径（含起点与终点格），用于调试渲染。
        选择“先沿当前朝向所在轴走直线，再拐一次”的方案；若该轴方向为 0 位移，则改用另一轴。
        """
        sx, sy = start; tx, ty = end
        dx, dy = tx - sx, ty - sy
        path = [start]

        # 先选首段轴：若当前朝向在 x 轴(0/2)，优先走 x；在 y 轴(1/3)，优先走 y。
        prefer_x_first = self.robot_dir_idx in (0, 2)
        if prefer_x_first and dx == 0: prefer_x_first = False
        if (not prefer_x_first) and dy == 0: prefer_x_first = True

        if prefer_x_first:
            # 首段：沿 x
            if dx != 0:
                step_x = 1 if dx > 0 else -1
                for x in range(sx + step_x, tx + step_x, step_x):
                    path.append((x, sy))
            # 次段：沿 y
            if dy != 0:
                step_y = 1 if dy > 0 else -1
                for y in range(sy + step_y, ty + step_y, step_y):
                    path.append((tx, y))
        else:
            # 首段：沿 y
            if dy != 0:
                step_y = 1 if dy > 0 else -1
                for y in range(sy + step_y, ty + step_y, step_y):
                    path.append((sx, y))
            # 次段：沿 x
            if dx != 0:
                step_x = 1 if dx > 0 else -1
                for x in range(sx + step_x, tx + step_x, step_x):
                    path.append((x, ty))
        return path

    # ========== 视野、路径查找、渲染 ==========
    def _get_visible_cells(self, pos, direction_idx):
        # 注意：这个基础的视野函数会“看穿”墙壁。
        # 一个更真实的实现需要光线投射算法来处理遮挡。
        visible = set()
        px, py = pos
        f_dx, f_dy = self.dirs[direction_idx]
        s_dx, s_dy = self.dirs[(direction_idx - 1 + 4) % 4]  # 左侧方向
        for depth in range(self.fov_depth):
            side_offset_range = self.fov_width // 2
            for side_offset in range(-side_offset_range, side_offset_range + 1):
                cell_x = px + depth * f_dx + side_offset * s_dx
                cell_y = py + depth * f_dy + side_offset * s_dy
                # 只添加在网格内且不是墙壁的单元格
                if 0 <= cell_x < self.width and 0 <= cell_y < self.height and (cell_x, cell_y) not in self.wall_cells:
                    visible.add((cell_x, cell_y))
        return visible

    def _find_path_bfs(self, start, end):
        # 保留：如果后续需要无障碍网格内的最短路调试，可以使用
        queue = deque([([start], start)])
        visited_path = {start}
        while queue:
            path, current = queue.popleft()
            if current == end: return path
            cx, cy = current
            neighbor_indices = list(range(4)); random.shuffle(neighbor_indices)
            for i in neighbor_indices:
                dx, dy = self.dirs[i]
                nx, ny = cx + dx, cy + dy
                # MODIFIED: 确保下一步不在墙壁里
                if (1 <= nx < self.width - 1 and 1 <= ny < self.height - 1 and 
                        (nx, ny) not in visited_path):
                    visited_path.add((nx, ny))
                    queue.append((list(path) + [(nx, ny)], (nx, ny)))
        return None

    def _render_debug_map(self, path, target_pos, candidate_points=None):
        # MODIFIED: 渲染时区分墙壁
        grid = [['.' for _ in range(self.width)] for _ in range(self.height)]
        for x, y in self.wall_cells:
            grid[y][x] = 'W'  # NEW: 'W' 代表墙壁
        
        for x, y in self.unseen_cells:
            grid[y][x] = '#'
        if candidate_points:
            for x, y in candidate_points:
                if grid[y][x] == '.':
                    grid[y][x] = 'c'
        if path:
            for x, y in path:
                if grid[y][x] in ['.', 'c']:
                    grid[y][x] = '*'
        if target_pos:
            grid[target_pos[1]][target_pos[0]] = 'T'
        rx, ry = self.robot_pos
        grid[ry][rx] = self.dir_symbols.get(self.robot_dir_idx, 'R')
        map_str = f" MAP DEBUG (Strategy: {self.strategy}) "
        map_str = f"{map_str:-^{self.width+4}}\n"
        for r_idx, row in enumerate(grid):
            map_str += f"{r_idx:02}|{''.join(row)}|\n"
        map_str += "-" * (self.width + 4) + "\n"
        map_str += f"Legend: W=Wall, #=Unseen, .=Seen, c=Candidate, {grid[ry][rx]}=Robot, T=Target, *=Path\n"
        return map_str

    # ========== 主流程 ==========
    def run_exploration(self, start_pos=(1, 1), start_dir_idx=0):
        # 合法性检查
        if start_pos in self.wall_cells:
            raise ValueError(f"Start position {start_pos} cannot be inside a wall.")
        
        self.robot_pos = start_pos
        self.robot_dir_idx = start_dir_idx
        # 初始观察，从 unseen_cells 中移除可见部分
        self.unseen_cells -= self._get_visible_cells(self.robot_pos, self.robot_dir_idx)
        
        step_count = 0
        while self.unseen_cells:
            step_count += 1
            
            # --- a. 根据策略确定候选点集合 ---
            candidate_points_to_evaluate = set()
            if self.strategy == 'frontier':
                # MODIFIED: seen_cells 是所有可探索区域减去未见区域
                explorable_area = set((x, y) for x in range(1, self.width - 1) for y in range(1, self.height - 1))
                seen_cells = explorable_area - self.unseen_cells
                for sx, sy in seen_cells:
                    for dx, dy in self.dirs:
                        if (sx + dx, sy + dy) in self.unseen_cells:
                            candidate_points_to_evaluate.add((sx, sy))
                            break
            elif self.strategy == 'room_boundary':
                candidate_points_to_evaluate = self.room_boundary_points

            if not candidate_points_to_evaluate:
                break

            # (评估和决策部分)
            max_score = -1.0
            best_vantage_states = []
            for vx, vy in candidate_points_to_evaluate:
                for d_idx in range(4):
                    visible = self._get_visible_cells((vx, vy), d_idx)
                    newly_seen_count = len(self.unseen_cells.intersection(visible))
                    if newly_seen_count == 0:
                        continue
                    score = float(newly_seen_count)
                    if self.use_distance_penalty:
                        distance = math.hypot(vx - self.robot_pos[0], vy - self.robot_pos[1])
                        score /= (distance + 1)
                    if score > max_score:
                        max_score = score
                        best_vantage_states = [{'pos': (vx, vy), 'dir': d_idx}]
                    elif abs(score - max_score) < 1e-9:
                        best_vantage_states.append({'pos': (vx, vy), 'dir': d_idx})

            if not best_vantage_states:
                print(f"DEBUG STEP {step_count}: No vantage state found. Exploration might be complete or stuck.")
                break
            
            target_state = random.choice(best_vantage_states)
            target_pos = target_state['pos']
            target_dir_at_dest = target_state['dir']

            # 用 L 形路径替代逐格 BFS：只在中间拐一次
            l_path = self._plan_L_path(self.robot_pos, target_pos)
            print(f"\n{'='*25} DEBUG REPORT: STEP {step_count} {'='*25}")
            print(f"Chosen Target State: Go to {target_pos} and face direction {target_dir_at_dest}")
            print(self._render_debug_map(l_path, target_pos, candidate_points_to_evaluate))

            # 执行 L 形运动：先沿首段轴直走，再左/右一次，再沿次段轴直走
            sx, sy = self.robot_pos; tx, ty = target_pos
            dx, dy = tx - sx, ty - sy

            # 判定首段轴：尽量让起步不转向（沿当前朝向所在轴直走）
            first_x = self.robot_dir_idx in (0, 2)
            if first_x and dx == 0:
                first_x = False
            if (not first_x) and dy == 0:
                first_x = True

            if first_x:
                # 起步对齐到东西向（若已对齐则不转），直走 |dx| 步
                if dx != 0:
                    want = 0 if dx > 0 else 2
                    self._turn_to(want)
                    self._forward_steps(abs(dx))
                # 拐一次到南北向并直走 |dy| 步
                if dy != 0:
                    want = 1 if dy > 0 else 3
                    self._turn_to(want)
                    self._forward_steps(abs(dy))
            else:
                # 起步对齐到南北向并直走 |dy| 步
                if dy != 0:
                    want = 1 if dy > 0 else 3
                    self._turn_to(want)
                    self._forward_steps(abs(dy))
                # 拐一次到东西向并直走 |dx| 步
                if dx != 0:
                    want = 0 if dx > 0 else 2
                    self._turn_to(want)
                    self._forward_steps(abs(dx))

            # 仅在到达目标点后对齐到观察方向；路途中不再频繁转向
            final_dir = target_dir_at_dest
            self._turn_to(final_dir)
            self.unseen_cells -= self._get_visible_cells(self.robot_pos, self.robot_dir_idx)

            # MODIFIED: 'room_boundary' 策略的撞墙逻辑调整（只追加动作，不改变位置）
            if self.strategy == 'room_boundary' and target_pos in self.room_boundary_points:
                print("DEBUG: Target is on inner boundary, performing wall bump.")
                wall_facing_dir = -1
                tx, ty = target_pos
                if tx == 1: wall_facing_dir = 2  # 西内墙
                elif tx == self.width - 2: wall_facing_dir = 0  # 东内墙
                elif ty == 1: wall_facing_dir = 3  # 北内墙
                elif ty == self.height - 2: wall_facing_dir = 1  # 南内墙
                
                if wall_facing_dir != -1:
                    self._turn_to(wall_facing_dir)
                    bump_count = random.randint(0, 3)
                    print(f"DEBUG: Bumping wall {bump_count} time(s).")
                    self.script.extend(['forward'] * bump_count)

        print(f"\n{'='*25} EXPLORATION COMPLETE {'='*25}")
        return self.script

# class ExplorationDebugger:
#     """
#     """
#     def __init__(self, width, height, fov_depth, fov_width, 
#                  strategy: str = 'frontier', use_distance_penalty=True):
#         self.width, self.height = width, height
#         self.fov_depth, self.fov_width = fov_depth, fov_width
#         self.strategy, self.use_distance_penalty = strategy, use_distance_penalty
#         if self.strategy not in ['frontier', 'room_boundary']:
#             raise ValueError("Strategy must be 'frontier' or 'room_boundary'")

#         # NEW: 定义墙壁单元格集合
#         self.wall_cells = set()
#         for x in range(width):
#             self.wall_cells.add((x, 0))
#             self.wall_cells.add((x, height - 1))
#         for y in range(1, height - 1):
#             self.wall_cells.add((0, y))
#             self.wall_cells.add((width - 1, y))

#         # MODIFIED: unseen_cells 现在是墙壁内部的可探索区域
#         self.unseen_cells = set(
#             (x, y) for x in range(1, width - 1) for y in range(1, height - 1)
#         )
        
#         self.script, self.robot_pos, self.robot_dir_idx = [], None, None

#         # MODIFIED: 房间边界策略的目标点现在是内壁旁边的点
#         self.room_boundary_points = set()
#         for x in range(1, width - 1):
#             self.room_boundary_points.add((x, 1))
#             self.room_boundary_points.add((x, height - 2))
#         for y in range(1, height - 1):
#             self.room_boundary_points.add((1, y))
#             self.room_boundary_points.add((width - 2, y))

#         self.dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
#         self.dir_symbols = {0: '>', 1: 'v', 2: '<', 3: '^'}

#     def _get_visible_cells(self, pos, direction_idx):
#         # 注意：这个基础的视野函数会“看穿”墙壁。
#         # 一个更真实的实现需要光线投射算法来处理遮挡。
#         visible = set()
#         px, py = pos
#         f_dx, f_dy = self.dirs[direction_idx]
#         s_dx, s_dy = self.dirs[(direction_idx - 1 + 4) % 4]
#         for depth in range(self.fov_depth):
#             side_offset_range = self.fov_width // 2
#             for side_offset in range(-side_offset_range, side_offset_range + 1):
#                 cell_x = px + depth * f_dx + side_offset * s_dx
#                 cell_y = py + depth * f_dy + side_offset * s_dy
#                 # 只添加在网格内且不是墙壁的单元格
#                 if 0 <= cell_x < self.width and 0 <= cell_y < self.height and (cell_x, cell_y) not in self.wall_cells:
#                     visible.add((cell_x, cell_y))
#         return visible

#     def _find_path_bfs(self, start, end):
#         queue = deque([([start], start)])
#         visited_path = {start}
#         while queue:
#             path, current = queue.popleft()
#             if current == end: return path
#             cx, cy = current
#             neighbor_indices = list(range(4)); random.shuffle(neighbor_indices)
#             for i in neighbor_indices:
#                 dx, dy = self.dirs[i]
#                 nx, ny = cx + dx, cy + dy
#                 # MODIFIED: 确保下一步不在墙壁里
#                 if (1 <= nx < self.width - 1 and 1 <= ny < self.height - 1 and 
#                         (nx, ny) not in visited_path):
#                     visited_path.add((nx, ny))
#                     queue.append((list(path) + [(nx, ny)], (nx, ny)))
#         return None

#     def _render_debug_map(self, path, target_pos, candidate_points=None):
#         # MODIFIED: 渲染时区分墙壁
#         grid = [['.' for _ in range(self.width)] for _ in range(self.height)]
#         for x,y in self.wall_cells: grid[y][x] = 'W' # NEW: 'W' 代表墙壁
        
#         for x, y in self.unseen_cells: grid[y][x] = '#'
#         if candidate_points:
#             for x,y in candidate_points:
#                 if grid[y][x] == '.': grid[y][x] = 'c'
#         if path:
#             for x, y in path:
#                 if grid[y][x] in ['.', 'c']: grid[y][x] = '*'
#         if target_pos: grid[target_pos[1]][target_pos[0]] = 'T'
#         rx, ry = self.robot_pos
#         grid[ry][rx] = self.dir_symbols.get(self.robot_dir_idx, 'R')
#         map_str = f" MAP DEBUG (Strategy: {self.strategy}) "
#         map_str = f"{map_str:-^{self.width+4}}\n"
#         for r_idx, row in enumerate(grid):
#             map_str += f"{r_idx:02}|{''.join(row)}|\n"
#         map_str += "-" * (self.width + 4) + "\n"
#         map_str += f"Legend: W=Wall, #=Unseen, .=Seen, c=Candidate, {grid[ry][rx]}=Robot, T=Target, *=Path\n"
#         return map_str

#     # MODIFIED: 更改了默认起始位置
#     def run_exploration(self, start_pos=(1, 1), start_dir_idx=0):
#         # 合法性检查
#         if start_pos in self.wall_cells:
#             raise ValueError(f"Start position {start_pos} cannot be inside a wall.")
        
#         self.robot_pos = start_pos
#         self.robot_dir_idx = start_dir_idx
#         # 初始观察，从 unseen_cells 中移除可见部分
#         self.unseen_cells -= self._get_visible_cells(self.robot_pos, self.robot_dir_idx)
        
#         step_count = 0
#         while self.unseen_cells:
#             step_count += 1
            
#             # --- a. 根据策略确定候选点集合 ---
#             candidate_points_to_evaluate = set()
#             if self.strategy == 'frontier':
#                 # MODIFIED: seen_cells 是所有可探索区域减去未见区域
#                 explorable_area = set((x, y) for x in range(1, self.width - 1) for y in range(1, self.height - 1))
#                 seen_cells = explorable_area - self.unseen_cells
#                 for sx, sy in seen_cells:
#                     for dx, dy in self.dirs:
#                         if (sx + dx, sy + dy) in self.unseen_cells:
#                             candidate_points_to_evaluate.add((sx, sy)); break
#             elif self.strategy == 'room_boundary':
#                 candidate_points_to_evaluate = self.room_boundary_points

#             if not candidate_points_to_evaluate: break

#             # (评估和决策部分无需修改，逻辑是通用的)
#             max_score = -1.0
#             best_vantage_states = []
#             for vx, vy in candidate_points_to_evaluate:
#                 for d_idx in range(4):
#                     visible = self._get_visible_cells((vx, vy), d_idx)
#                     newly_seen_count = len(self.unseen_cells.intersection(visible))
#                     if newly_seen_count == 0: continue
#                     score = float(newly_seen_count)
#                     if self.use_distance_penalty:
#                         distance = math.hypot(vx - self.robot_pos[0], vy - self.robot_pos[1])
#                         score /= (distance + 1)
#                     if score > max_score:
#                         max_score = score
#                         best_vantage_states = [{'pos': (vx, vy), 'dir': d_idx}]
#                     elif abs(score - max_score) < 1e-9:
#                         best_vantage_states.append({'pos': (vx, vy), 'dir': d_idx})

#             if not best_vantage_states:
#                 print(f"DEBUG STEP {step_count}: No vantage state found. Exploration might be complete or stuck.")
#                 break
            
#             target_state = random.choice(best_vantage_states)
#             target_pos = target_state['pos']
#             target_dir_at_dest = target_state['dir']

#             # (路径规划和执行移动部分无需修改)
#             path = self._find_path_bfs(self.robot_pos, target_pos) if self.robot_pos != target_pos else [self.robot_pos]
#             print(f"\n{'='*25} DEBUG REPORT: STEP {step_count} {'='*25}")
#             print(f"Chosen Target State: Go to {target_pos} and face direction {target_dir_at_dest}")
#             print(self._render_debug_map(path, target_pos, candidate_points_to_evaluate))
#             if not path: continue
#             if len(path) > 1:
#                 for i in range(len(path) - 1):
#                     current_pos = path[i]; next_pos = path[i+1]
#                     dx, dy = next_pos[0] - current_pos[0], next_pos[1] - current_pos[1]
#                     target_dir = self.dirs.index((dx, dy))
#                     turn_diff = target_dir - self.robot_dir_idx
#                     if turn_diff in [1, -3]: self.script.append('right'); self.robot_dir_idx = (self.robot_dir_idx + 1) % 4
#                     elif turn_diff in [-1, 3]: self.script.append('left'); self.robot_dir_idx = (self.robot_dir_idx - 1 + 4) % 4
#                     elif abs(turn_diff) == 2: self.script.extend(['right', 'right']); self.robot_dir_idx = (self.robot_dir_idx + 2) % 4
#                     self.unseen_cells -= self._get_visible_cells(current_pos, self.robot_dir_idx)
#                     self.script.append('forward')
#                 self.robot_pos = path[-1]

#             final_dir = target_dir_at_dest
#             turn_diff = final_dir - self.robot_dir_idx
#             if turn_diff in [1, -3]: self.script.append('right'); self.robot_dir_idx = (self.robot_dir_idx + 1) % 4
#             elif turn_diff in [-1, 3]: self.script.append('left'); self.robot_dir_idx = (self.robot_dir_idx - 1 + 4) % 4
#             elif abs(turn_diff) == 2: self.script.extend(['right', 'right']); self.robot_dir_idx = (self.robot_dir_idx + 2) % 4
#             self.unseen_cells -= self._get_visible_cells(self.robot_pos, self.robot_dir_idx)

#             # MODIFIED: 'room_boundary' 策略的撞墙逻辑调整
#             if self.strategy == 'room_boundary' and target_pos in self.room_boundary_points:
#                 print("DEBUG: Target is on inner boundary, performing wall bump.")
#                 wall_facing_dir = -1
#                 tx, ty = target_pos
#                 if tx == 1: wall_facing_dir = 2 # 西内墙
#                 elif tx == self.width - 2: wall_facing_dir = 0 # 东内墙
#                 elif ty == 1: wall_facing_dir = 3 # 北内墙
#                 elif ty == self.height - 2: wall_facing_dir = 1 # 南内墙
                
#                 if wall_facing_dir != -1:
#                     turn_diff = wall_facing_dir - self.robot_dir_idx
#                     if turn_diff in [1, -3]: self.script.append('right'); self.robot_dir_idx = (self.robot_dir_idx + 1) % 4
#                     elif turn_diff in [-1, 3]: self.script.append('left'); self.robot_dir_idx = (self.robot_dir_idx - 1 + 4) % 4
#                     elif abs(turn_diff) == 2: self.script.extend(['right', 'right']); self.robot_dir_idx = (self.robot_dir_idx + 2) % 4
                    
#                     bump_count = random.randint(0, 3)
#                     print(f"DEBUG: Bumping wall {bump_count} time(s).")
#                     self.script.extend(['forward'] * bump_count)

#         print(f"\n{'='*25} EXPLORATION COMPLETE {'='*25}")
#         return self.script
    
# class MultiRoomPlanner:
#     """
#     一个用于两房间环境的高级任务规划器。
#     修正版: 每个房间的探索都是一次独立的、自包含的任务。
#     """
#     def __init__(self, room_size, door_pos, fov_depth, fov_width, use_distance_penalty=True):
#         # ... (构造函数和之前一样) ...
#         self.width = room_size * 2 + 1
#         self.height = room_size
#         self.room_size = room_size
#         self.door_pos = door_pos
#         self.fov_depth = fov_depth
#         self.fov_width = fov_width
#         self.use_distance_penalty = use_distance_penalty
#         self.dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
#         self.dir_symbols = {0: '>', 1: 'v', 2: '<', 3: '^'}

#     def _get_visible_cells(self, pos, direction_idx, room_bounds):
#         # (这个辅助函数现在也需要知道房间边界，以防视野“看穿”到另一个房间)
#         visible = set()
#         px, py = pos
#         f_dx, f_dy = self.dirs[direction_idx]
#         s_dx, s_dy = self.dirs[(direction_idx - 1 + 4) % 4]
#         for depth in range(self.fov_depth):
#             side_offset_range = self.fov_width // 2
#             for side_offset in range(-side_offset_range, side_offset_range + 1):
#                 cell_x = px + depth * f_dx + side_offset * s_dx
#                 cell_y = py + depth * f_dy + side_offset * s_dy
#                 if (room_bounds['x_min'] <= cell_x < room_bounds['x_max'] and
#                     room_bounds['y_min'] <= cell_y < room_bounds['y_max']):
#                     visible.add((cell_x, cell_y))
#         return visible

#     def _find_path_bfs(self, start, end, room_bounds):
#         # ... (和之前一样) ...
#         queue = deque([([start], start)])
#         visited_path = {start}
#         while queue:
#             path, current = queue.popleft()
#             if current == end: return path
#             cx, cy = current
#             neighbor_indices = list(range(4)); random.shuffle(neighbor_indices)
#             for i in neighbor_indices:
#                 dx, dy = self.dirs[i]
#                 nx, ny = cx + dx, cy + dy
#                 if (room_bounds['x_min'] <= nx < room_bounds['x_max'] and 
#                     room_bounds['y_min'] <= ny < room_bounds['y_max'] and 
#                     (nx, ny) not in visited_path):
#                     visited_path.add((nx, ny))
#                     queue.append((list(path) + [(nx, ny)], (nx, ny)))
#         return None

#     def _convert_path_to_script(self, path, start_dir_idx):
#         # ... (和之前一样) ...
#         script = []
#         direction_idx = start_dir_idx

#         if len(path) > 1:
#             for i in range(len(path) - 1):
#                 current_pos, next_pos = path[i], path[i+1]
#                 dx, dy = next_pos[0] - current_pos[0], next_pos[1] - current_pos[1]
#                 target_dir = self.dirs.index((dx, dy))
#                 turn_diff = target_dir - direction_idx
#                 if turn_diff in [1, -3]: script.append('right'); direction_idx = (direction_idx + 1) % 4
#                 elif turn_diff in [-1, 3]: script.append('left'); direction_idx = (direction_idx - 1 + 4) % 4
#                 elif abs(turn_diff) == 2: script.extend(['right', 'right']); direction_idx = (direction_idx + 2) % 4
#                 script.append('forward')
#         return script, direction_idx

#     def _explore_single_room(self, start_pos, start_dir_idx, room_bounds):
#         """
#         在单个限定区域内执行完整的探索算法。
#         这个函数现在是完全独立的，不依赖任何外部状态。
#         """
#         script = []
#         robot_pos, robot_dir_idx = start_pos, start_dir_idx
        
#         # 1. 【核心修改】只创建当前房间的未见格子集合
#         unseen_cells = set((x, y) for x in range(room_bounds['x_min'], room_bounds['x_max']) 
#                                 for y in range(room_bounds['y_min'], room_bounds['y_max']))
        
#         # 初始观察
#         unseen_cells -= self._get_visible_cells(robot_pos, robot_dir_idx, room_bounds)

#         while unseen_cells:
#             # a. 寻找边缘点 (只在当前房间内)
#             seen_cells = set((x,y) for x in range(room_bounds['x_min'], room_bounds['x_max']) for y in range(room_bounds['y_min'], room_bounds['y_max'])) - unseen_cells
#             frontier_points = set(p for p in seen_cells if any((p[0]+d[0], p[1]+d[1]) in unseen_cells for d in self.dirs))
#             if not frontier_points: break

#             # b. 寻找最佳观测状态
#             max_score, best_vantage_states = -1.0, []
#             for vx, vy in frontier_points:
#                 for d_idx in range(4):
#                     newly_seen_count = len(unseen_cells.intersection(self._get_visible_cells((vx, vy), d_idx, room_bounds)))
#                     if newly_seen_count == 0: continue
#                     score = float(newly_seen_count)
#                     if self.use_distance_penalty:
#                         distance = math.hypot(vx - robot_pos[0], vy - robot_pos[1])
#                         score /= (distance + 1)
#                     if score > max_score: max_score, best_vantage_states = score, [{'pos': (vx, vy), 'dir': d_idx}]
#                     elif abs(score - max_score) < 1e-9: best_vantage_states.append({'pos': (vx, vy), 'dir': d_idx})
            
#             if not best_vantage_states: break
            
#             target_state = random.choice(best_vantage_states)
#             target_pos, target_dir_at_dest = target_state['pos'], target_state['dir']

#             # c. 规划并执行路径
#             path = self._find_path_bfs(robot_pos, target_pos, room_bounds) if robot_pos != target_pos else [robot_pos]
#             if not path: continue

#             if len(path) > 1:
#                 for i in range(len(path) - 1):
#                     current_pos = path[i]; next_pos = path[i+1]
#                     dx, dy = next_pos[0] - current_pos[0], next_pos[1] - current_pos[1]
#                     target_dir = self.dirs.index((dx, dy))
#                     turn_diff = target_dir - robot_dir_idx
#                     if turn_diff in [1, -3]: script.append('right'); robot_dir_idx = (robot_dir_idx + 1) % 4
#                     elif turn_diff in [-1, 3]: script.append('left'); robot_dir_idx = (robot_dir_idx - 1 + 4) % 4
#                     elif abs(turn_diff) == 2: script.extend(['right', 'right']); robot_dir_idx = (robot_dir_idx + 2) % 4
#                     unseen_cells -= self._get_visible_cells(current_pos, robot_dir_idx, room_bounds)
#                     script.append('forward')
            
#             robot_pos = path[-1]
            
#             # d. 到达后最终观察
#             final_dir = target_dir_at_dest
#             turn_diff = final_dir - robot_dir_idx
#             if turn_diff in [1, -3]: script.append('right'); robot_dir_idx = (robot_dir_idx + 1) % 4
#             elif turn_diff in [-1, 3]: script.append('left'); robot_dir_idx = (robot_dir_idx - 1 + 4) % 4
#             elif abs(turn_diff) == 2: script.extend(['right', 'right']); robot_dir_idx = (robot_dir_idx + 2) % 4
#             unseen_cells -= self._get_visible_cells(robot_pos, robot_dir_idx, room_bounds)
        
#         return script, robot_pos, robot_dir_idx

#     def run_exploration(self, agent_start_pos, agent_start_dir):
#         """
#         生成覆盖两个房间的完整任务轨迹。
#         现在每个房间的探索都是独立的。
#         """
#         full_script = []
        
#         # --- 阶段一: 探索左侧房间 ---
#         print("--- Phase 1: Exploring Room 1 ---")
#         print(self.door_pos,  self.height)
#         room1_bounds = {'x_min': 1, 'x_max': self.door_pos[0]+1, 'y_min': 1, 'y_max': self.height+1}
#         script1, pos1, dir1 = self._explore_single_room(agent_start_pos, agent_start_dir, room1_bounds)
#         full_script.extend(script1)
        
#         # --- 阶段二: 移动到门口并穿过 ---
#         print("\n--- Phase 2: Traversing Door ---")
#         target_before_door = (self.door_pos[0], self.door_pos[1])#needs debbuging! self.door_pos[0] 
#         path_to_door = self._find_path_bfs(pos1, target_before_door, room1_bounds)
        
#         path_script, dir2 = self._convert_path_to_script(path_to_door, dir1)
#         full_script.extend(path_script)
        
#         # 确保面朝门 (东方, dir=0)
#         turn_diff = 0 - dir2
#         if turn_diff in [1, -3]: full_script.append('right'); dir2 = (dir2 + 1) % 4
#         elif turn_diff in [-1, 3]: full_script.append('left'); dir2 = (dir2 - 1 + 4) % 4
#         elif abs(turn_diff) == 2: full_script.extend(['right', 'right']); dir2 = (dir2 + 2) % 4

#         # 穿门动作: 开门，前进，再前进
#         full_script.extend(['toggle', 'forward', 'forward'])
#         pos2, dir2 = (1, self.door_pos[1]), 0
        
#         # --- 阶段三: 探索右侧房间 ---
#         print("\n--- Phase 3: Exploring Room 2 ---")
#         room2_bounds = {'x_min': 1, 'x_max': self.door_pos[0]+1, 'y_min': 1, 'y_max': self.height+1}
#         # 【核心修改】调用同样的函数，但传入第二个房间的边界和新的起点
#         script3, _, _ = self._explore_single_room(pos2, dir2, room2_bounds)
#         full_script.extend(script3)

#         print("\n--- Full Trajectory Generation Complete! ---")
#         return full_script