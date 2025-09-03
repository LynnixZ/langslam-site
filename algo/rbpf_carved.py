# rbpf_occ.py
# Python 3.10+
# 用于 10x10 网格（0..9），5x5 视野（depth=0..4, side=-2..2），朝向 ∈ {E,S,W,N}

from __future__ import annotations
import math, re, random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable, Set

# ========= 基本常量 =========
GRID_W = GRID_H = 10
DEPTH_MAX = 4
SIDE_MIN, SIDE_MAX = -2, 2
HEADINGS = ["E", "S", "W", "N"]  # 0:E,1:S,2:W,3:N

# ========= 实用函数 =========
def clamp(v:int, lo:int, hi:int)->int:
    return max(lo, min(hi, v))

def heading_index(name:str)->int:
    name = name.strip().upper()
    return {"EAST":0, "SOUTH":1, "WEST":2, "NORTH":3}[name]

def rot_local_to_world(d:int, s:int, h:int)->Tuple[int,int]:
    # 离散 90° 旋转：local(depth,side) -> world Δ(x,y)
    if h==0:   # E
        return (d, s)
    if h==1:   # S
        return (-s, d)
    if h==2:   # W
        return (-d, -s)
    if h==3:   # N
        return (s, -d)
    raise ValueError("invalid heading")

def compose_pose(x:int,y:int,h:int, action:str, step_len:int)->Tuple[int,int,int]:
    # 根据动作推进姿态；左/右/掉头只改朝向；forward 改位置（遇边界则停）
    if action == "forward":
        dx, dy = rot_local_to_world(step_len, 0, h)
        nx, ny = clamp(x+dx, 0, GRID_W-1), clamp(y+dy, 0, GRID_H-1)
        return (nx, ny, h)
    if action == "left":
        return (x,y,(h+1)%4)
    if action == "right":
        return (x,y,(h+3)%4)
    if action == "turnaround":
        return (x,y,(h+2)%4)
    # 其他视作停留
    return (x,y,h)

def bresenham_cells(x0:int,y0:int, x1:int,y1:int)->List[Tuple[int,int]]:
    # 直线离散化，包含起点和终点（小网格下足够）
    cells = []
    dx = abs(x1-x0); dy = abs(y1-y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x,y = x0,y0
    while True:
        cells.append((x,y))
        if x==x1 and y==y1: break
        e2 = 2*err
        if e2 > -dy:
            err -= dy; x += sx
        if e2 < dx:
            err += dx; y += sy
    return cells

def sigmoid(l:float)->float:
    return 1.0/(1.0+math.exp(-l))

# ========= 观测解析 =========
@dataclass
class StepObs:
    action: str                      # 'forward'/'left'/'right'/'turnaround'/...
    walls_local: List[Tuple[int,int]]  # 本步上报的墙（local）
    cannot_see: Optional[Set[Tuple[int,int]]] = None  # 显式不可见（local）

@dataclass
class ParsedLog:
    x0:int; y0:int; h0:int
    steps: List[StepObs]

def parse_log(txt:str)->ParsedLog:
    # 起点与朝向
    mpos = re.search(r"starting absolute position is\s*\((\d+)\s*,\s*(\d+)\)", txt, re.I)
    x0, y0 = int(mpos.group(1)), int(mpos.group(2))
    mh = re.search(r"Initial Observation:\s*You['’]re facing\s+(East|South|West|North)", txt, re.I)
    h0 = heading_index(mh.group(1))

    # 逐步解析
    parts = re.split(r"\bStep\s*\d+\s*:\s*", txt)
    # parts[0] 是前导，引导后每个 parts[i] 对应一步
    steps: List[StepObs] = []
    # 用一个正则抓局部坐标列表
    def parse_pairs(chunk:str, keyphrase:str)->List[Tuple[int,int]]:
        pat = keyphrase + r".*?coordinates\s*\(([^)]+)\)"
        res: List[Tuple[int,int]] = []
        # 可能有多段 "coordinates (...)"，逐个抓
        for m in re.finditer(pat, chunk, re.I|re.S):
            coords = m.group(1)
            for p in coords.split("),"):
                nums = re.findall(r"[-]?\d+", p)
                if len(nums)>=2:
                    d,s = int(nums[0]), int(nums[1])
                    # 合法范围裁剪（日志已裁剪到 5x5；此处再次保护）
                    if 0<=d<=DEPTH_MAX and SIDE_MIN<=s<=SIDE_MAX:
                        res.append((d,s))
        return res

    # 每个步骤块的首词就是 action（形如 "left. ..."）
    step_chunks = re.findall(r"Step\s*\d+\s*:\s*(.*?)\n(?=Step|\Z)", txt, re.I|re.S)
    for chunk in step_chunks:
        mact = re.match(r"\s*([A-Za-z]+)\s*\.", chunk)
        action = mact.group(1).lower() if mact else "noop"
        walls = parse_pairs(chunk, r"You can see grey walls at")
        cant = parse_pairs(chunk, r"You cannot see")
        cant_set = set(cant) if len(cant)>0 else None
        steps.append(StepObs(action=action, walls_local=walls, cannot_see=cant_set))
    return ParsedLog(x0=x0,y0=y0,h0=h0, steps=steps)

# ========= 抖动核 =========
def chebyshev_kernel(radius:int=1, phi=(1.0, 0.55, 0.25, 0.10))->Dict[Tuple[int,int], float]:
    """
    返回 Δd,Δs -> 权重 的字典；支持半径 1..3。
    phi[k] 是环 k 的权重（k=0..3）。更远为 0。
    """
    assert 0 < radius <= 3
    out: Dict[Tuple[int,int], float] = {}
    for dd in range(-radius, radius+1):
        for ds in range(-radius, radius+1):
            r = max(abs(dd), abs(ds))
            if r<=radius:
                w = phi[r]
                if w>0: out[(dd,ds)] = w
    # 归一化
    s = sum(out.values())
    for k in list(out.keys()):
        out[k] /= (s if s>0 else 1.0)
    return out

# ========= 粒子与地图 =========
@dataclass
class Particle:
    x:int; y:int; h:int
    w:float = 1.0
    logodds: Dict[Tuple[int,int], float] = field(default_factory=dict)
    visited: Set[Tuple[int,int]] = field(default_factory=set)

    def copy_shallow(self)->"Particle":
        return Particle(self.x, self.y, self.h, self.w,
                        dict(self.logodds), set(self.visited))

def nearest_occ_distance(p:Particle, q:Tuple[int,int])->float:
    # 返回 q 到最近“已知占用格”的欧氏距离；若当前无占用，则给一个常数（弱信息）
    occ_cells = [cell for cell,l in p.logodds.items() if l > 2.0]  # 阈值可调
    if not occ_cells: 
        return 2.5  # 空图时给弱信息
    qx,qy = q
    best = 999.0
    for ox,oy in occ_cells:
        d = math.hypot(ox-qx, oy-qy)
        if d < best: best = d
    return best

def visible_set_from_walls(walls_local:Set[Tuple[int,int]])->Set[Tuple[int,int]]:
    """
    B 类：不带 cannot_see 的情况，用局部上报的墙端点推断 5x5 中的可见集：
    - 对每个 (d,s) 作为“最近命中”，沿 (0,0)->(d,s) 的离散线段，命中之前的格子都是可见 free 候选；
    - 命中之后的格子视作遮挡（不可见）。
    - 对整个 5x5 取并集。
    """
    vis: Set[Tuple[int,int]] = set()
    # 所有可能的目标（深度优先，近端优先）
    targets = [(d,s) for d in range(1,DEPTH_MAX+1) for s in range(SIDE_MIN,SIDE_MAX+1)]
    # 将墙端点放入集合便于查询
    wallset = set(walls_local)
    for d,s in targets:
        # 如果该射线上存在最近墙端点（就是它自己）则可见到它；否则可见到末端
        # 我们用 Bresenham 判定路径
        # 先构造 local -> world 的离散路径（在 local 平面上也可用同一算法）
        # 这里直接用 “local 网格” 下的 Bresenham（起点 0,0）
        path = bresenham_cells(0,0, d, s)
        # 去掉起点 (0,0)
        body = path[1:]
        blocked = False
        for idx,(dd,ss) in enumerate(body):
            vis.add((dd,ss))
            if (dd,ss) in wallset:
                blocked = True
                break
        # 如果没被命中，剩余的自然也算“可见”（直到 5×5 边界）
        # 上面已把全路径加入 vis，无需额外处理
    return vis

# ========= RBPF 主循环 =========
@dataclass
class RBPFParams:
    N_particles:int = 64
    resample_thresh:float = 0.5  # N_eff/N 阈值
    jitter_radius:int = 1        # 抖动半径（1=3x3；可设到 3=7x7）
    phi:Tuple[float,float,float,float] = (1.0,0.55,0.25,0.10)  # 环权重
    sigma_hit:float = 0.9        # L_hit 的尺度（格）
    dL_occ:float = 2.0           # log-odds 增量（占用）
    dL_free:float = -1.0         # log-odds 增量（空）
    L_clip:Tuple[float,float] = (-6.0, 6.0)
    init_exact_frac:float = 0.9  # 初始粒子放在精确起点的比例
    # 运动模型（前进步长 0/1/2 的概率；转向“成功/未转/转错”）
    p_step:Tuple[float,float,float] = (0.10, 0.80, 0.10)
    p_turn:Tuple[float,float,float] = (0.85, 0.10, 0.05)  # (intended, stay, wrong)

class RBPF:
    def __init__(self, params:RBPFParams):
        self.P = params
        self.kernel = chebyshev_kernel(self.P.jitter_radius, self.P.phi)

    def init_particles(self, x0:int,y0:int,h0:int)->List[Particle]:
        N = self.P.N_particles
        M = int(round(N*self.P.init_exact_frac))
        parts = []
        for _ in range(M):
            p = Particle(x0,y0,h0, w=1.0/N)
            p.visited.add((x0,y0))  # 起点计入路径
            parts.append(p)
        for _ in range(N-M):
            dx = random.choice([-1,0,1]); dy = random.choice([-1,0,1])
            hh = (h0 + random.choice([0,0,1,3]))%4
            px,py = clamp(x0+dx,0,GRID_W-1), clamp(y0+dy,0,GRID_H-1)
            p = Particle(px,py,hh, w=1.0/N)
            p.visited.add((px,py))
            parts.append(p)
        return parts
    def sample_step_len(self)->int:
        r = random.random()
        p0,p1,p2 = self.P.p_step
        return 0 if r<p0 else (1 if r<p0+p1 else 2)

    def sample_turn_outcome(self, action:str, h:int)->int:
        # 返回新的朝向索引
        intended = {"left":(h+1)%4, "right":(h+3)%4, "turnaround":(h+2)%4}[action]
        stay = h
        wrong = {"left":(h+3)%4, "right":(h+1)%4, "turnaround":h}[action]
        r = random.random()
        a,b,c = self.P.p_turn
        return intended if r<a else (stay if r<a+b else wrong)

    def weight_update(self, p:Particle, sensor_local_walls:List[Tuple[int,int]])->float:
        # 观测似然（对数域）：端点独立、候选边缘化
        if not sensor_local_walls:
            return 0.0  # 没有新信息
        log_like = 0.0
        for (d,s) in sensor_local_walls:
            # 候选集合（抖动核）
            # 计算边缘化： sum_alpha alpha * L_hit(q)
            accum = 0.0
            for (dd,ds), alpha in self.kernel.items():
                d2, s2 = d+dd, s+ds
                if not (0<=d2<=DEPTH_MAX and SIDE_MIN<=s2<=SIDE_MAX):
                    continue
                # local -> world
                dx,dy = rot_local_to_world(d2, s2, p.h)
                qx,qy = clamp(p.x+dx,0,GRID_W-1), clamp(p.y+dy,0,GRID_H-1)
                r = nearest_occ_distance(p, (qx,qy))
                Lhit = math.exp(-0.5 * (r/self.P.sigma_hit)**2)
                accum += alpha * Lhit
            # 防止数值为 0
            accum = max(accum, 1e-6)
            log_like += math.log(accum)
        return log_like
    def map_update(self, p:Particle, sensor_local_walls:Set[Tuple[int,int]], visible_set:Optional[Set[Tuple[int,int]]]):
        Lmin,Lmax = self.P.L_clip

        def world_of(d:int,s:int)->Tuple[int,int]:
            dx,dy = rot_local_to_world(d, s, p.h)
            return (clamp(p.x+dx,0,GRID_W-1), clamp(p.y+dy,0,GRID_H-1))

        # 1) 先收集端点在抖动核下的候选占用格 cand_occ（用于排除）
        cand_occ: Set[Tuple[int,int]] = set()
        endpoints = sorted(set(sensor_local_walls))
        for (d,s) in endpoints:
            for (dd,ds), alpha in self.kernel.items():
                d2,s2 = d+dd, s+ds
                if 0<=d2<=DEPTH_MAX and SIDE_MIN<=s2<=SIDE_MAX:
                    cand_occ.add(world_of(d2,s2))

        # 2) 射线更新：候选端点的射线“端点之前” free，端点邻域 occ
        for (d,s) in endpoints:
            # 2a) free（端点之前）
            for (dd,ds), alpha in self.kernel.items():
                d2,s2 = d+dd, s+ds
                if not (0<=d2<=DEPTH_MAX and SIDE_MIN<=s2<=SIDE_MAX): 
                    continue
                qx,qy = world_of(d2,s2)
                ray = bresenham_cells(p.x,p.y, qx,qy)
                body = ray[1:-1]
                for cx,cy in body:
                    if visible_set is not None:
                        # world->local 反投影，确保只在可见集里写 free
                        dx,dy = cx-p.x, cy-p.y
                        if p.h==0:   dloc,sloc = dx, dy
                        elif p.h==1: dloc,sloc = dy, -dx
                        elif p.h==2: dloc,sloc = -dx, -dy
                        else:        dloc,sloc = -dy, dx
                        if (dloc,sloc) not in visible_set:
                            continue
                    key = (cx,cy)
                    p.logodds[key] = clamp_float(p.logodds.get(key,0.0) + alpha*self.P.dL_free, Lmin, Lmax)
            # 2b) occ（端点邻域）
            for (dd,ds), alpha in self.kernel.items():
                d2,s2 = d+dd, s+ds
                if not (0<=d2<=DEPTH_MAX and SIDE_MIN<=s2<=SIDE_MAX): 
                    continue
                q = world_of(d2,s2)
                p.logodds[q] = clamp_float(p.logodds.get(q,0.0) + alpha*self.P.dL_occ, Lmin, Lmax)

        # 3) **可见但非墙** 一律标 free（补齐“视野里除了墙全是 empty”）
        if visible_set is not None:
            # 可选：如果担心抖动，把 cand_occ 一圈邻域也排除；不需要就把 margin 设 0
            margin = 0  # 设为 1 可排除 cand_occ 的一圈邻域
            excl: Set[Tuple[int,int]] = set(cand_occ)
            if margin >= 1:
                for (ox,oy) in list(cand_occ):
                    for dx in (-1,0,1):
                        for dy in (-1,0,1):
                            excl.add((clamp(ox+dx,0,GRID_W-1), clamp(oy+dy,0,GRID_H-1)))

            for (d,s) in visible_set:
                if d == 0 and s == 0:
                    continue  # 当前位置已在 visited 里另外加 free
                wx,wy = world_of(d,s)
                if (wx,wy) in excl:
                    continue  # 端点（或其邻域）不当作 free
                key = (wx,wy)
                p.logodds[key] = clamp_float(p.logodds.get(key,0.0) + self.P.dL_free, Lmin, Lmax)

        # 4) 当前位置作为路径 free 证据
        p.visited.add((p.x,p.y))
        p.logodds[(p.x,p.y)] = clamp_float(p.logodds.get((p.x,p.y),0.0) + self.P.dL_free, Lmin, Lmax)
        
    def step(self, parts:List[Particle], step:StepObs):
        # 1) 运动提议（含失败/多走）
        new_parts: List[Particle] = []
        for p in parts:
            np = p.copy_shallow()
            if step.action == "forward":
                ell = self.sample_step_len()
                np.x, np.y, np.h = compose_pose(p.x,p.y,p.h, "forward", ell)
            elif step.action in ("left","right","turnaround"):
                # 三分支：按 p_turn 采样朝向结果
                np.h = self.sample_turn_outcome(step.action, p.h)
            # 其他动作不变
            new_parts.append(np)

        # 2) 观测似然（对数）
        walls_local = step.walls_local
        for np in new_parts:
            logL = self.weight_update(np, walls_local)
            np.w *= math.exp(logL)

        # 3) 归一化 + 重采样
        norm = sum(p.w for p in new_parts)
        if norm <= 0:  # 极端数值防护
            for p in new_parts: p.w = 1.0/len(new_parts)
        else:
            for p in new_parts: p.w /= norm
        Neff = 1.0 / sum(p.w*p.w for p in new_parts)
        if Neff / len(new_parts) < self.P.resample_thresh:
            new_parts = systematic_resample(new_parts)

        # 4) 地图更新（逆模型）
        # 可见集：A 类直接用 step.cannot_see；B 类用上报墙推断
        visible: Optional[Set[Tuple[int,int]]] = None
        if step.cannot_see is not None:
            # A 类：可见 = 5x5 中的格 - cannot_see
            all_local = {(d,s) for d in range(1,DEPTH_MAX+1) for s in range(SIDE_MIN,SIDE_MAX+1)}
            visible = all_local.difference(step.cannot_see)
        else:
            # B 类：基于墙端点推断
            visible = visible_set_from_walls(set(walls_local))

        for np in new_parts:
            self.map_update(np, set(walls_local), visible)

        return new_parts

def clamp_float(v:float, lo:float, hi:float)->float:
    return max(lo, min(hi, v))

def systematic_resample(parts:List[Particle])->List[Particle]:
    N = len(parts)
    ws = [p.w for p in parts]
    cumsum = [0.0]
    for w in ws: cumsum.append(cumsum[-1]+w)
    step = 1.0/N
    u0 = random.random()*step
    out: List[Particle] = []
    i = 0
    for m in range(N):
        u = u0 + m*step
        while u > cumsum[i+1]:
            i += 1
        # 复制第 i 个
        p = parts[i]
        cp = p.copy_shallow()
        cp.w = 1.0/N
        # 微小姿态抖动（不越界）
        dx = random.choice([0,0,1,-1]); dy = random.choice([0,0,1,-1])
        cp.x, cp.y = clamp(cp.x+dx,0,GRID_W-1), clamp(cp.y+dy,0,GRID_H-1)
        out.append(cp)
    return out

# ========= 最终汇总 =========
def merge_map(parts:List[Particle])->Dict[Tuple[int,int], float]:
    # 加权合并 log-odds（再转回 log-odds；简单做法是权重平均 log-odds）
    # 注：严格做法应在概率域平均；此处 N 小问题不大
    wsum = sum(p.w for p in parts)
    agg: Dict[Tuple[int,int], float] = {}
    for p in parts:
        w = p.w / (wsum if wsum>0 else 1.0)
        for cell, L in p.logodds.items():
            agg[cell] = agg.get(cell, 0.0) + w*L
    return agg

def best_particle(parts:List[Particle])->Particle:
    return max(parts, key=lambda p: p.w)

def empty_cells_output(parts:List[Particle], tau_free:float=-1.0)->List[Tuple[int,int]]:
    # 取加权合并图作为判据；并把最佳粒子的路径格子并入 empty
    agg = merge_map(parts)
    empty = {cell for cell,L in agg.items() if L < tau_free}
    bp = best_particle(parts)
    empty.update(bp.visited)
    # 只输出在 0..9 内的格子
    return sorted([(x,y) for (x,y) in empty if 0<=x<GRID_W and 0<=y<GRID_H])

from typing import Optional, Tuple

def _phi_from_noise(rho: float) -> Tuple[int, Tuple[float,float,float,float]]:
    """
    根据观测抖动强度 rho ∈ [0,1] 生成：抖动半径、各环权重 phi。
    - rho 小：更集中的核（半径 1，中心权重大）
    - rho 大：更分散的核（半径 2/3，外环权重更大）
    """
    rho = max(0.0, min(1.0, rho))
    # 基础环权重模板
    c0, c1, c2, c3 = 1.0 - rho, 0.55*rho, 0.30*rho, 0.15*rho
    # 归一化到 (0..1) 比例，由 chebyshev_kernel 再做精确归一化
    phi = (max(1e-6,c0), c1, c2, c3)
    # 半径按 rho 分段
    if rho <= 0.25:
        radius = 1
    elif rho <= 0.55:
        radius = 2
    else:
        radius = 3
    return radius, phi

def rbpf_carved(
    log_text: str,
    params: Optional[RBPFParams] = None,
    *,
    num_particles: Optional[int] = None,
    failure_prob: Optional[float] = None,
    stale_obs_prob: Optional[float] = None,
) -> str:
    """
    运行 RBPF + 占用栅格。
    - num_particles: 粒子数（覆盖 RBPFParams.N_particles）
    - failure_prob: 动作失败总概率（对应 “未动 + 多走” 之和）
        映射为步长分布 p_step=(p0,p1,p2)：
            p1 = 1 - failure_prob
            p0 = 0.6 * failure_prob   # 未动更常见
            p2 = 0.4 * failure_prob   # 多走较少
    - stale_obs_prob: 观测坐标抖动强度 ∈ [0,1]
        映射为离散抖动核（半径与环权重）与 L_hit 的尺度：
            jitter_radius, phi = _phi_from_noise(stale_obs_prob)
            sigma_hit = 0.8 + 0.7 * stale_obs_prob
    若不传这些关键字参数，则使用 params 或默认 RBPFParams。
    """
    P = params or RBPFParams()

    if num_particles is not None:
        P.N_particles = int(num_particles)

    if failure_prob is not None:
        fp = max(0.0, min(1.0, float(failure_prob)))
        p1 = max(1e-6, 1.0 - fp)
        p0 = max(1e-6, 0.6 * fp)
        p2 = max(1e-6, 0.4 * fp)
        # 归一化，避免极端输入造成总和≠1
        s = p0 + p1 + p2
        P.p_step = (p0/s, p1/s, p2/s)

    if stale_obs_prob is not None:
        rho = max(0.0, min(1.0, float(stale_obs_prob)))
        radius, phi = _phi_from_noise(rho)
        P.jitter_radius = radius
        P.phi = phi
        P.sigma_hit = 0.8 + 0.7 * rho

    parsed = parse_log(log_text)
    rbpf = RBPF(P)
    parts = rbpf.init_particles(parsed.x0, parsed.y0, parsed.h0)
    for step in parsed.steps:
        parts = rbpf.step(parts, step)
    empties = empty_cells_output(parts, tau_free=-1.0)
    body = ", ".join(f"({x},{y})" for (x,y) in empties)
    return f"&&& empty spaces at coordinates: {body} &&&"