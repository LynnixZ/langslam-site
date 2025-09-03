from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.minigrid_env import MiniGridEnv
from gymnasium.envs.registration import register
import gymnasium as gym
from minigrid.core.world_object import Goal, Ball, Key, Wall, Door


class CustomEmptyEnv(MiniGridEnv):
    """
    ## Description

    An empty 5x5 room where the agent must reach one or more randomly placed
    goal squares.

    ## Mission Space

    "get to the green goal square"

    """

    def __init__(
        self,
        num_objects: int = 1,
        max_steps: int | None = None,
        length: int = 5,
        agent_view_size: int = 5, 
        same_color_goals: bool = False, # NEW: Add a boolean flag to control goal colors

        **kwargs,
    ):
        self.num_goals = num_objects

        mission_space = MissionSpace(mission_func=lambda: "get to any flags in the room")
        self.same_color_goals = same_color_goals # NEW: Store the flag

        if max_steps is None:
            max_steps = 4 * length**2

        super().__init__(
            mission_space=mission_space,
            grid_size=length,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            **kwargs,
        )


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent
        self.place_agent(
            top=(1, 1),
            size=(self.width - 1, self.height - 1),
            rand_dir=True
        )
        # MODIFIED: Logic to handle goal colors
        if self.same_color_goals:
            # New logic: All goals have the same color
            goal_color = self._rand_color()
            for _ in range(self.num_goals):
                self.place_obj(Goal(color=goal_color))
            self.mission = f"get to the {goal_color} flag"
        else:
            # Original logic: Goals have different random colors

            # 添加一个安全计数器，以防止在 _rand_color() 无法提供足够多的
            # 独特颜色时，程序陷入无限循环。
            final_colors = self.get_unique_colors()

            # 现在，使用这个保证唯一的颜色列表来放置物体
            last_color = ""
            for color in final_colors:
                self.place_obj(Goal(color=color))
                last_color = color # 这里的 last_color 仍然可以用于简单的 mission
            self.mission = f"get to the {last_color} flag"

    def get_unique_colors(self):
        chosen_colors = set()
        max_tries = 100
        tries = 0

            # 循环，直到我们收集到足够数量的独特颜色
        while len(chosen_colors) < self.num_goals and tries < max_tries:
            random_color = self._rand_color()
            chosen_colors.add(random_color) # 集合的add方法会自动处理重复
                                                # 如果颜色已存在，则什么也不会发生
            tries += 1

            # 循环结束后，检查我们是否成功收集到了足够的颜色
        if len(chosen_colors) < self.num_goals:
            raise Exception(
                    f"在尝试 {max_tries} 次后，未能生成 {self.num_goals} 个独特的颜色。"
                    f"self._rand_color() 函数可能无法提供足够多的颜色种类。"
                )

        final_colors = sorted(list(chosen_colors))
        return final_colors


# It's good practice to register your custom environment
# so you can create it with gym.make()
try:
    register(
        id='MiniGrid-CustomEmpty-5x5-v0',
        entry_point='create_env:CustomEmptyEnv'
    )
except Exception:
    # This will fail if you run the script multiple times, which is fine.
    pass



class CorridorEnv(MiniGridEnv):
    """
    一个 1xN 的一维走廊环境。
    AI需要从左端走到右端的终点，途中可能会有随机放置的物体。
    """
    def __init__(self, length: int = 10, num_objects: int = 2, max_steps: int | None = None, **kwargs):
        self.length = length
        self.num_objects = num_objects
        width = length + 2
        height = 3
        if max_steps is None:
            max_steps = 5 * length
        # 修改了任务描述以反映新的游戏规则
        mission_space = MissionSpace(
            mission_func=lambda: "touch all goals"
        )
        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=False,
            **kwargs
        )

    def _gen_grid(self, width, height):
        self.grid.wall_rect(0, 0, width, height)
        for x in range(0, width):
            self.grid.set(x, 0, Wall())
            self.grid.set(x, 2, Wall())
        self.put_obj(Wall(), 0, 1)
        self.put_obj(Wall(), width - 1, 1)
        for x in range(1, width - 1):
            self.grid.set(x, 1, None)
            
        # 在走廊内随机放置两个Goal
        # 注意：这里的 place_obj 逻辑被简化了以确保能放下两个目标
        goal_pos = self.np_random.choice(range(2, width - 2), self.num_objects, replace=False)
        for i in range(self.num_objects):
            self.put_obj(Goal(self._rand_color()), goal_pos[i], 1)

        self.place_agent(top=(1, 1), size=(1, 1), rand_dir=False)
        self.agent_dir = 0
    
    # --- 新增：重写 step 方法 ---
    def step(self, action):
        # 1. 首先，调用父类的step方法，让它处理所有基本动作
        obs, reward, terminated, truncated, info = super().step(action)

        # 2. 检查我们当前所在的格子是什么
        current_cell = self.grid.get(*self.agent_pos)

        # 3. 我们的新规则：如果走到了Goal上...
        if current_cell is not None and current_cell.type == 'goal':
            
            # 给予一个正奖励来鼓励AI
            reward = 0.5  # 你可以设置任何你想要的奖励值
            
            # 关键：把父类设置的 terminated 标志覆盖回 False，让游戏继续
            terminated = False
            
            # 我们可以让这个Goal“消失”，避免重复得分
            #self.grid.set(*self.agent_pos, None)
            print(f"Agent touched a goal at {self.agent_pos}! Game continues.")

        return obs, reward, terminated, truncated, info
try:
    register(id='MiniGrid-MyCorridor-v0', entry_point='create_env:CorridorEnv')
except gym.error.Error:
    pass # 如果已经注册过，就跳过



class TwoRoomsEnv(MiniGridEnv):
    """
    ## 描述

    一个由两个房间组成的环境。两个房间之间由一面墙隔开，墙上有一扇门。
    智能体必须从左边房间出发，穿过门到达右边房间的目标点。

    ## 任务空间

    "go through the door and get to the goal"
    """

    def __init__(
        self,
        length: int = 7,
        max_steps: int | None = None,
        agent_view_size: int = 5,
        num_objects: int = 1,
        door_pos: tuple | None = None, # <--- 新增参数

        **kwargs,
    ):
        self.room_size = length
        width = length * 2 + 3 # 两个房间宽度 + 一面分割墙
        height = length + 2
        self.door_pos = door_pos # <--- 保存门的位置
        self.num_goals = num_objects
        if max_steps is None:
            max_steps = 10 * length**2

        mission_space = MissionSpace(
            mission_func=lambda: "go through the door and get to the goal"
        )

        super().__init__(
            mission_space=mission_space,
            width=width,
            height=height,
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            **kwargs,
        )

    def _gen_grid(self, width, height):
        # 创建一个空的网格
        self.grid = Grid(width, height)

        # 生成周围的墙
        self.grid.wall_rect(0, 0, width, height)

        # 创建分割两个房间的墙
        divider_x = self.room_size+1
        for y in range(1, height - 1):
            self.grid.set(divider_x, y, Wall())
        
        # 在分割墙上随机放置一扇门
        if self.door_pos is None:
            # 如果未指定，则随机放置
            door_y = self._rand_int(1, height - 2)
        else:
            # 如果已指定，使用指定的y坐标
            # 我们假设x坐标总是中间的墙，只关心y
            door_y = self.door_pos[1]
            # 安全检查
            if not (1 <= door_y < height - 1):
                raise ValueError(f"Door y-coordinate {door_y} is invalid for room height {height}.")
        
        self.put_obj(Door('yellow', is_locked=False), divider_x, door_y)

        # 在左边房间放置智能体
        self.place_agent(
            top=(1, 1),
            size=(self.room_size - 1, self.room_size - 2),
            rand_dir=True
        )
        # 1. 决定在右房间放置多少个目标 (至少1个，最多不超过总数)
        num_goals_right = self._rand_int(1, self.num_goals)
        
        # 2. 在右房间放置目标
        for _ in range(num_goals_right):
            self.place_obj(
                obj=Goal(color=self._rand_color()),
                top=(self.room_size + 1, 1), # 右房间的左上角
                size=(self.room_size - 2, self.room_size - 2) # 右房间的内部区域
            )

        # 3. 在左房间放置剩余的目标
        num_goals_left = self.num_goals - num_goals_right
        for _ in range(num_goals_left):
            self.place_obj(
                obj=Goal(color=self._rand_color()),
                top=(1, 1), # 左房间的左上角
                size=(self.room_size - 2, self.room_size - 2) # 左房间的内部区域
            )
        self.mission = "go through the door and get to the green goal square"
try:
    register(
        id='MiniGrid-TwoRooms-v0',
        entry_point='create_env:TwoRoomsEnv' # 或者 'your_filename:TwoRoomsEnv'
    )
except gym.error.Error:
    pass
# Example of how to use your new environment
if __name__ == "__main__":
    # You can now specify the number of goals when creating the environment
    env = gym.make('MiniGrid-CustomEmpty-5x5-v0', num_goals=3, size=5, render_mode="human")

    # Reset the environment and render it
    obs, info = env.reset()
    env.render()

    input("A 5x5 room with 3 random goals has been generated. Press Enter to close...")
    env.close()