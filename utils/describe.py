from collections import defaultdict
from .constants import IDX_TO_OBJECT, IDX_TO_COLOR, IDX_TO_STATE

def describe_observation(
    obs,
    walls: bool = True,
    oneDim: bool = False,
    dir: bool = False,
    list_unseen: bool = True,
    list_empty: bool = True,
) -> str:
    # 方向前缀
    direction_text = ""
    if dir:
        direction_map = {0: 'East', 1: 'South', 2: 'West', 3: 'North'}
        direction = direction_map.get(obs['direction'], 'unknown')
        direction_text = f"You're facing {direction}. "

    image = obs['image']
    object_grid, color_grid, state_grid = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    view_h, view_w = image.shape[0], image.shape[1]
    center_x = view_w // 2

    # 统一的分组键设计：
    #   ('obj', color, name)         -> 一般物体（包括 wall, key, ball, box, goal 等）
    #   ('door', color, state)       -> 门，带状态
    #   'empty'                      -> 可见空格
    #   'unseen'                     -> 观测中的不可见格（被墙遮挡/未进入视野）
    grouped = defaultdict(list)

    def rel_coord(ix, iy):
        # 完全复用你现有的相对坐标定义
        depth = (view_h - 1) - ix
        side  = iy - center_x
        if oneDim:
            return str(depth) if side == 0 else None  # 1D 只报中心线
        else:
            return f"({depth}, {side})"

    for y in range(view_h):       # 行
        for x in range(view_w):   # 列
            obj_id = object_grid[y, x]

            # 1) 可见物体（忽略 unseen(0) 与 empty(1)）
            if obj_id > 1:
                obj_name = IDX_TO_OBJECT.get(obj_id, 'unknown')
                if not walls and obj_name == 'wall':
                    continue
                color_id, state_id = color_grid[y, x], state_grid[y, x]
                color_name = IDX_TO_COLOR.get(color_id, 'a')

                if obj_name == 'door':
                    state_name = IDX_TO_STATE.get(state_id, 'unknown_state')
                    key = ('door', color_name, state_name)
                else:
                    key = ('obj', color_name, obj_name)

                rc = rel_coord(x, y)
                if rc is not None:
                    grouped[key].append(rc)

            # 2) 不可见（unseen = 0）
            elif obj_id == 0 and list_unseen:
                rc = rel_coord(x, y)
                if rc is not None:
                    grouped['unseen'].append(rc)

            # 3) 可见空格（empty = 1）
            elif obj_id == 1 and list_empty:
                rc = rel_coord(x, y)
                if rc is not None:
                    grouped['empty'].append(rc)

    # 生成文字
    desc = generate_object_descriptions(grouped, oneDim=oneDim)

    if not desc:
        return f"{direction_text}You see nothing of interest."
    else:
        return f"{direction_text}" + " ".join(desc)


def generate_object_descriptions(grouped_objects, oneDim: bool = False):
    """
    grouped_objects 的键是：
      ('obj', color, name)
      ('door', color, state)
      'empty'
      'unseen'
    值是**已经转成相对坐标字符串**的列表（例如 "3" 或 "(2, -1)"）
    """
    out = []

    # 为了可读性，坐标做一次去重+排序
    def _sort_rel(coords):
        if not coords:
            return []
        if oneDim:
            return sorted(set(coords), key=lambda z: int(z))
        else:
            def keyfn(s):
                d, s2 = s.strip("()").split(",")
                return (int(d.strip()), int(s2.strip()))
            return sorted(set(coords), key=keyfn)

    for key, coords in grouped_objects.items():
        coords = _sort_rel(coords)
        if not coords:
            continue

        # empty
        if key == 'empty':
            if len(coords) == 1:
                out.append(f"Coordinate {coords[0]} is empty.")
            else:
                out.append(f"Coordinates {', '.join(coords)} are empty.")
            continue

        # unseen
        if key == 'unseen':
            if len(coords) == 1:
                out.append(f"You cannot see coordinate {coords[0]}.")
            else:
                out.append(f"You cannot see coordinates {', '.join(coords)}.")
            continue

        # 一般物体 / 门
        kind = key[0]
        if kind == 'door':
            _, color, state = key
            if len(coords) == 1:
                out.append(f"You can see a {color} door at coordinate {coords[0]}. It is {state}.")
            else:
                out.append(f"You can see {color} doors at coordinates {', '.join(coords)}.")
        elif kind == 'obj':
            _, color, name = key
            if len(coords) == 1:
                out.append(f"You can see a {color} {name} at coordinate {coords[0]}.")
            else:
                out.append(f"You can see {color} {name}s at coordinates {', '.join(coords)}.")

    return out
