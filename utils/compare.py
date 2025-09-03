import re
import math
from typing import Dict, Tuple, Union, Set
from minigrid.minigrid_env import MiniGridEnv

Coord2D = Tuple[int, int]

# ----------------------
# ground truth 按 env_name
# ----------------------
def generate_ground_truth(env: MiniGridEnv, env_name: str) -> Dict[Union[int, Coord2D], Union[Tuple[str, str], str]]:
    """
    - MiniGrid-CustomEmpty-5x5-v0: flags 模式，返回 {(x,y): (color, type)}，含 goal/door（与原逻辑一致）
    - MiniGrid-CarvedPathRoom-v0: empty 模式，返回 {(x,y): 'empty'}（grid.get(x,y) is None）
    """
    grid = env.unwrapped.grid
    W, H = grid.width, grid.height

    if env_name == "MiniGrid-CarvedPathRoom-v0":
        # empty 模式：收集所有空格
        truth_map: Dict[Coord2D, str] = {}
        for x in range(W):
            for y in range(H):
                if grid.get(x, y) is None:
                    truth_map[(x, y)] = "empty"
        return truth_map

    # flags 模式（默认）
    truth_map: Dict[Coord2D, Tuple[str, str]] = {}
    for x in range(W):
        for y in range(H):
            cell = grid.get(x, y)
            if not cell:
                continue
            if cell.type == 'goal':
                truth_map[(x, y)] = (cell.color, "flag")
            elif cell.type == 'door':
                truth_map[(x, y)] = (cell.color, "door")
            elif cell.type == 'wall':
                truth_map[(x, y)] = (getattr(cell, "color", "grey"), "wall")
    return truth_map


# ----------------------
# LLM 输出解析按 env_name
# ----------------------
def parse_llm_output(response_text: str, env_name: str) -> Dict[Union[int, Coord2D], Union[Tuple[str, str], str]]:
    """
    - MiniGrid-CustomEmpty-5x5-v0: flags 解析（保持原有格式）
      a {color} (wall|flag|door) at coordinate (x, y)
    - MiniGrid-CarvedPathRoom-v0: empty 解析，支持：
      A) empty spaces at coordinates (x1,y1) (x2,y2) ...
      B) an empty cell/space at coordinate (x, y), ...
    """
    m = re.search(r'&&&([\s\S]*?)&&&', response_text)
    if not m:
        print("Warning: summary block &&&...&&& not found.")
        return {}
    text = m.group(1).lower()

    if env_name == "MiniGrid-CarvedPathRoom-v0":
        out: Dict[Coord2D, str] = {}

        # A) 抓取所有括号坐标
        for xs, ys in re.findall(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", text):
            out[(int(xs), int(ys))] = "empty"

        # B) 兼容 “an empty cell/space at coordinate (x,y)”
        pat2 = re.compile(
            r"(?:an\s+)?empty\s+(?:cell|space)\s+at\s+coordinate\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
        )
        for xs, ys in pat2.findall(text):
            out[(int(xs), int(ys))] = "empty"
        return out

    # flags（默认）
    out: Dict[Coord2D, Tuple[str, str]] = {}
    pat = re.compile(
        r"a\s+(\w+)\s+(wall|flag|door)\s+at\s+coordinate\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
    )
    for color, typ, xs, ys in pat.findall(text):
        out[(int(xs), int(ys))] = (color, typ)
    return out

Coord = Tuple[int, int]


# ----------------------
# 对比分发
# ----------------------
def compare_maps(ground_truth: Dict, llm_map: Dict, env_name: str) -> Dict[str, Union[bool, float, str]]:
    if env_name == "MiniGrid-CustomEmpty-5x5-v0":
        return _compare_maps_flags(ground_truth, llm_map)
    elif env_name == "MiniGrid-CarvedPathRoom-v0":
        return _compare_maps_empty(ground_truth, llm_map)
    else:
        # 未列出的环境，按 flags 处理
        return _compare_maps_flags(ground_truth, llm_map)



def _dilate_set(points: Set[Coord], r: int) -> Set[Coord]:
    """对点集做 L1 半径 r 的膨胀（菱形邻域）。"""
    if r <= 0 or not points:
        return set(points)
    out: Set[Coord] = set()
    for (x, y) in points:
        for dx in range(-r, r + 1):
            rem = r - abs(dx)
            for dy in range(-rem, rem + 1):
                out.add((x + dx, y + dy))
    return out

def _iou(a: Set[Coord], b: Set[Coord]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return 1.0 if union == 0 else inter / union

def _compare_maps_empty(ground_truth: Dict[Coord, str],
                        llm_map: Dict[Coord, str],
                        tolerance_r: int = 1) -> Dict[str, Union[bool, float, str]]:
    gt = set(ground_truth.keys())
    pr = set(llm_map.keys())

    # 原始 IoU（r=0）
    iou_raw = _iou(gt, pr)

    # 容忍 IoU：对双方各做一次半径 r 的膨胀，再算 IoU
    if tolerance_r > 0:
        gt_d = _dilate_set(gt, tolerance_r)
        pr_d = _dilate_set(pr, tolerance_r)
        iou_tol = _iou(gt_d, pr_d)
    else:
        iou_tol = iou_raw

    return {
        "is_perfect_match": gt == pr,
        # 主指标：容忍 IoU（把偏移≤r格的情况记成“基本对齐”）
        "overall_accuracy": iou_tol,
        # 为兼容你现有汇总，我把 goal_accuracy 也设成同一个数
        "goal_accuracy": iou_tol,
        "euclidean_distance_error": 0.0,
        "details": (
            f"tolerant IoU (r={tolerance_r}): {iou_tol:.2%}; "
            f"raw IoU: {iou_raw:.2%}  "
            f"(|GT|={len(gt)}, |Pred|={len(pr)})"
        ),
    }




# ----------------------
# flags: 保持你原来的定义（Jaccard + “按键完全匹配的比例” + 距离）
# ----------------------
def _compare_maps_flags(ground_truth: Dict[Union[int, Coord2D], Tuple[str, str]],
                        llm_map: Dict[Union[int, Coord2D], Tuple[str, str]]) -> Dict[str, Union[bool, float, str]]:
    gt_items = set(ground_truth.items())
    llm_items = set(llm_map.items())
    gt_keys = set(ground_truth.keys())
    llm_keys = set(llm_map.keys())

    inter = len(gt_items & llm_items)
    union = len(gt_items | llm_items)
    overall = 1.0 if union == 0 else inter / union

    if not gt_keys:
        goal_acc = 1.0
    else:
        correct = sum(1 for k, v in ground_truth.items() if k in llm_map and llm_map[k] == v)
        goal_acc = correct / len(gt_keys)

    # 若键是坐标，给一个距离误差；否则置 0
    common_keys = gt_keys & llm_keys
    dists = []
    for k in common_keys:
        gt_val = ground_truth[k]
        pr_val = llm_map[k]
        # flags 场景通常不需要坐标距离，这里留个兜底
        if isinstance(k, tuple):
            # 如果键本身就是坐标，距离为 0（键已同一）；若需比较值里的坐标，可在此扩展
            dists.append(0.0)
        elif isinstance(k, int):
            dists.append(0.0)
    avg_dist = sum(dists) / len(dists) if dists else 0.0

    details = (
        f"Ground Truth: {ground_truth}\n"
        f"LLM's Map   : {llm_map}\n"
        f"----------------------------------------\n"
        f"Overall Accuracy (Jaccard): {overall:.2%}\n"
        f"Goal Accuracy (Avg. Correctness): {goal_acc:.2%}\n"
        f"Euclidean Distance Error: {avg_dist:.2f}"
    )
    return {
        "is_perfect_match": gt_items == llm_items,
        "overall_accuracy": overall,
        "goal_accuracy": goal_acc,
        "euclidean_distance_error": avg_dist,
        "details": details,
    }

def map_to_records(m):
    """
    把形如 {(x,y): value} 或 {(x,y): (color,type)} 的地图
    转成可 JSON 序列化的记录列表：
      [{"x":x, "y":y, "value":...}] 或 [{"x":x,"y":y,"color":...,"type":...}]
    也兼容 1D/其他键：{"key": k, "value": v}
    """
    if m is None:
        return None
    out = []
    for k, v in m.items():
        rec = {}
        if isinstance(k, tuple) and len(k) == 2:
            rec["x"] = int(k[0])
            rec["y"] = int(k[1])
        else:
            rec["key"] = k  # 兼容 1D 或其他情况

        if isinstance(v, tuple):
            # 常见于 flags: (color, type)
            if len(v) == 2 and all(isinstance(x, str) for x in v):
                rec["color"] = v[0]
                rec["type"]  = v[1]
            else:
                rec["value"] = v
        else:
            rec["value"] = v
        out.append(rec)

    # 排个序，方便肉眼比对（按 y,x，再按 key）
    out.sort(key=lambda r: (r.get("y", -10**9), r.get("x", -10**9), str(r.get("key", ""))))
    return out
