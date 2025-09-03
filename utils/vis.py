# filename: utils/vis.py
from __future__ import annotations
import os
import numpy as np
import imageio.v2 as imageio
from typing import Dict, Tuple, Any, Iterable

def save_empty_prediction_image(
    llm_map: Dict[Tuple[int, int], Any],
    width: int,
    height: int,
    out_path: str,
    *,
    cell: int = 14,
    gray_unknown: int = 140,  # 默认未知/未声明用浅灰
    gray_border: int = 100,   # 外墙/边框用深灰（更贴近 MiniGrid 视觉）
    black_empty: int = 0      # LLM 说 empty 的格子画成黑色
) -> None:
    """
    将 { (x,y): 'empty' } 或 { (x,y): ('empty', ...)} 的字典渲染成图片。
    - 整张图先铺满浅灰(unknown)；
    - 外圈边框(0 与 width-1，0 与 height-1)画稍深的灰；
    - 出现在 llm_map 并标注为 'empty' 的格子填充黑色。
    """
    H, W = height, width
    hpx, wpx = H * cell, W * cell

    # 背景：浅灰
    img = np.full((hpx, wpx, 3), gray_unknown, dtype=np.uint8)

    # 外墙/边界：深灰（可选，纯视觉）
    def fill_cell(cx: int, cy: int, val: int):
        ys, ye = cy * cell, (cy + 1) * cell
        xs, xe = cx * cell, (cx + 1) * cell
        img[ys:ye, xs:xe, :] = val

    for x in range(W):
        fill_cell(x, 0, gray_border)
        fill_cell(x, H - 1, gray_border)
    for y in range(H):
        fill_cell(0, y, gray_border)
        fill_cell(W - 1, y, gray_border)

    # 解析 empty 坐标集合
    empties: Iterable[Tuple[int, int]] = []
    # 兼容两种格式： (x,y)->'empty' 或 (x,y)->('empty', 'anything')
    for (x, y), tag in llm_map.items():
        if isinstance(tag, str):
            if tag.lower().startswith("empty"):
                empties = (*empties, (x, y))
        elif isinstance(tag, tuple) and len(tag) >= 1:
            if isinstance(tag[0], str) and tag[0].lower().startswith("empty"):
                empties = (*empties, (x, y))

    # 绘制 empty
    for (x, y) in empties:
        if 0 <= x < W and 0 <= y < H:
            fill_cell(x, y, black_empty)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imageio.imwrite(out_path, img)
