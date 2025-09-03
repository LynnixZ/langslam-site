from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Goal, Wall

class CustomEmptyEnv(MiniGridEnv):
    """
    一个空的 length×length 房间，可放置多个随机旗帜。
    参数 same_color_goals=True 时所有旗帜同色；否则尽量取不同色。
    """

    def __init__(
        self,
        num_objects: int = 1,
        max_steps: int | None = None,
        length: int = 5,
        agent_view_size: int = 5,
        same_color_goals: bool = False,
        **kwargs,
    ):
        self.num_goals = num_objects
        self.same_color_goals = same_color_goals
        mission_space = MissionSpace(mission_func=lambda: "get to any flags in the room")

        if max_steps is None:
            max_steps = 4 * length**2

        super().__init__(
            mission_space=mission_space,
            grid_size=length,
            see_through_walls=True,
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.place_agent(
            top=(1, 1),
            size=(self.width - 1, self.height - 1),
            rand_dir=True,
        )

        if self.same_color_goals:
            goal_color = self._rand_color()
            for _ in range(self.num_goals):
                self.place_obj(Goal(color=goal_color))
            self.mission = f"get to the {goal_color} flag"
        else:
            final_colors = self.get_unique_colors()
            last_color = ""
            for color in final_colors:
                self.place_obj(Goal(color=color))
                last_color = color
            self.mission = f"get to the {last_color} flag"

    def get_unique_colors(self):
        chosen_colors = set()
        max_tries = 100
        tries = 0
        while len(chosen_colors) < self.num_goals and tries < max_tries:
            random_color = self._rand_color()
            chosen_colors.add(random_color)
            tries += 1
        if len(chosen_colors) < self.num_goals:
            raise Exception(
                f"尝试 {max_tries} 次后仍未得到 {self.num_goals} 个不同颜色；可减小 num_objects 或扩展颜色集合。"
            )
        final_colors = sorted(list(chosen_colors))
        return final_colors
