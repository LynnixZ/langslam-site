from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Goal, Wall, Door

class TwoRoomsEnv(MiniGridEnv):
    """
    两个房间，中间一堵墙，墙上有门。目标分布在两边房间。
    """

    def __init__(
        self,
        length: int = 7,
        max_steps: int | None = None,
        agent_view_size: int = 5,
        num_objects: int = 1,
        door_pos: tuple | None = None,
        **kwargs,
    ):
        self.room_size = length
        width = length * 2 + 3
        height = length + 2
        self.door_pos = door_pos
        self.num_goals = num_objects

        if max_steps is None:
            max_steps = 10 * length**2

        mission_space = MissionSpace(mission_func=lambda: "go through the door and get to the goal")

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # 中间分隔墙
        divider_x = self.room_size + 1
        for y in range(1, height - 1):
            self.grid.set(divider_x, y, Wall())

        # 门位置
        if self.door_pos is None:
            door_y = self._rand_int(1, height - 2)
        else:
            door_y = self.door_pos[1]
            if not (1 <= door_y < height - 1):
                raise ValueError(f"Door y-coordinate {door_y} is invalid for room height {height}.")
        self.put_obj(Door('yellow', is_locked=False), divider_x, door_y)

        # 左房间放 agent
        self.place_agent(top=(1, 1), size=(self.room_size - 1, self.room_size - 2), rand_dir=True)

        # 右房间放一部分目标
        num_goals_right = self._rand_int(1, self.num_goals)
        for _ in range(num_goals_right):
            self.place_obj(
                obj=Goal(color=self._rand_color()),
                top=(self.room_size + 1, 1),
                size=(self.room_size - 2, self.room_size - 2),
            )

        # 左房间放剩余目标
        num_goals_left = self.num_goals - num_goals_right
        for _ in range(num_goals_left):
            self.place_obj(
                obj=Goal(color=self._rand_color()),
                top=(1, 1),
                size=(self.room_size - 2, self.room_size - 2),
            )

        self.mission = "go through the door and get to the green goal square"
