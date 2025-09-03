from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Goal, Wall

class CorridorEnv(MiniGridEnv):
    """
    一个 1×N 的走廊。中间一行可放置若干旗帜。触碰旗帜给奖励但不结束。
    """

    def __init__(self, length: int = 10, num_objects: int = 2, max_steps: int | None = None, **kwargs):
        self.length = length
        self.num_objects = num_objects
        width = length + 2
        height = 3
        if max_steps is None:
            max_steps = 5 * length

        mission_space = MissionSpace(mission_func=lambda: "touch all goals")

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=False,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # 顶底两行设墙，左右两端封口
        for x in range(width):
            self.grid.set(x, 0, Wall())
            self.grid.set(x, height - 1, Wall())
        self.grid.set(0, 1, Wall())
        self.grid.set(width - 1, 1, Wall())

        # 清空中间通道
        for x in range(1, width - 1):
            self.grid.set(x, 1, None)

        # 随机放置旗帜（不重复）
        goal_pos = self.np_random.choice(range(2, width - 2), self.num_objects, replace=False)
        for i in range(self.num_objects):
            self.put_obj(Goal(self._rand_color()), int(goal_pos[i]), 1)

        # 放置智能体，朝东
        self.place_agent(top=(1, 1), size=(1, 1), rand_dir=False)
        self.agent_dir = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        current_cell = self.grid.get(*self.agent_pos)
        if current_cell is not None and current_cell.type == 'goal':
            reward = 0.5
            terminated = False
            print(f"Agent touched a goal at {self.agent_pos}! Game continues.")
        return obs, reward, terminated, truncated, info
