#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml


BASE_DIR = Path(__file__).resolve().parents[1]
SITE_DIR = BASE_DIR / "site"
MAPDATA_DIR = BASE_DIR / "mapdata"
GENERATED_DIR = SITE_DIR / "assets" / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def _safe_out_dir(raw: str) -> Path:
    out_dir = (BASE_DIR / raw).resolve()
    mapdata_root = MAPDATA_DIR.resolve()
    if out_dir != mapdata_root and mapdata_root not in out_dir.parents:
        raise ValueError("out_dir must be under mapdata/")
    return out_dir


def _auto_dir(env_name: str, length: int, view: int, episodes: int, flags: int) -> str:
    safe = env_name.replace("/", "-")
    return f"{safe}-L{length}-V{view}-E{episodes}-F{flags}"


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_answer(data: dict) -> str:
    flags = [it for it in data.get("ground_truth", []) if it.get("type") == "flag"]
    flags_sorted = sorted(flags, key=lambda x: (x.get("color", ""), x.get("x", 0), x.get("y", 0)))
    if not flags_sorted:
        return "No flags in this map."
    return ", ".join([f"{f['color']} flag at ({f['x']}, {f['y']})" for f in flags_sorted])


def _properties(data: dict) -> dict:
    env_name = data.get("env_name", "-")
    length = data.get("length", "-")
    view = data.get("prompt_kwargs", {}).get("agent_view_size", "-")
    flags = data.get("prompt_kwargs", {}).get("flags", 0)
    env_kwargs = data.get("env_kwargs", {}) or {}
    movable = 0
    if "movable_goal_specs" in env_kwargs:
        movable = len(env_kwargs.get("movable_goal_specs") or [])
    elif "movable_goals" in env_kwargs:
        movable = env_kwargs.get("movable_goals", 0)

    run_kwargs = data.get("run_kwargs", {}) or {}
    return {
        "environment": env_name,
        "map_size": f"{length} × {length}",
        "view_size": f"{view} × {view}",
        "flags": flags,
        "movable_objects": movable,
        "p_jump": run_kwargs.get("p_jump", 0.0),
        "p_fail": run_kwargs.get("p_fail", 0.0),
        "mu_d": run_kwargs.get("mu_d", 0.0),
        "mu_s": run_kwargs.get("mu_s", 0.0),
        "blackout": run_kwargs.get("blackout", False),
    }


def _run_generation(payload: dict) -> dict:
    corridor = payload.get("mapType") == "corridor"
    env_name = "MiniGrid-CarvedPathRoom-v0" if corridor else "MiniGrid-FlagsTightRoom-v0"
    prompt_env = env_name

    length = int(payload.get("length", 10))
    view = int(payload.get("viewSize", 5))
    episodes = int(payload.get("episodes", 1))
    flags = int(payload.get("flags", 0))
    movable = int(payload.get("movable", 0))
    out_dir = _safe_out_dir(payload.get("outDir", "mapdata/custom_runs"))

    p_jump = float(payload.get("pJump", 0))
    p_fail = float(payload.get("pFail", 0))
    mu_d = float(payload.get("muD", 0))
    mu_s = float(payload.get("muS", 0))
    blackout = bool(payload.get("blackout", False))

    cmd = [
        sys.executable, "-m", "map_main.make_episodes",
        "--env_name", env_name,
        "--prompt_env_names", prompt_env,
        "--num_episodes", str(episodes),
        "--length", str(length),
        "--agent_view_size", str(view),
        "--flags", str(flags),
        "--noise_log",
        "--p_jump", str(p_jump),
        "--p_fail", str(p_fail),
        "--mu_d", str(mu_d),
        "--mu_s", str(mu_s),
        "--movable_goals", str(movable),
        "--out_dir", str(out_dir.relative_to(BASE_DIR)),
    ]
    if blackout:
        cmd.append("--blackout")

    if corridor:
        cmd += [
            "--seg_band", str(payload.get("segBand", 1)),
            "--seg_prob_left", str(payload.get("segLeft", 0.2)),
            "--seg_prob_right", str(payload.get("segRight", 0.2)),
            "--seg_end_trim", str(payload.get("segTrim", 1)),
            "--seg_skip_prob", str(payload.get("segSkip", 0.0)),
        ]
        if payload.get("postFillDisable", True):
            cmd.append("--post_fill_disable")

    subprocess.run(cmd, cwd=BASE_DIR, check=True)

    auto_dir = _auto_dir(prompt_env, length, view, episodes, flags)
    base_dir = out_dir / auto_dir

    if env_name == "MiniGrid-FlagsTightRoom-v0":
        yaml_path = base_dir / "flags" / "ep_00000_flags_prompt.yaml"
        gif_path = base_dir / "flags" / "ep_00000_flags_trace.gif"
    else:
        yaml_path = base_dir / "ep_00000_main_prompt.yaml"
        gif_path = base_dir / "ep_00000_main_trace.gif"

    data = _load_yaml(yaml_path)
    answer = _build_answer(data)

    stamp = int(time.time())
    out_name = f"generated-{stamp}.gif"
    out_gif = GENERATED_DIR / out_name
    out_gif.write_bytes(gif_path.read_bytes())

    return {
        "gif_url": f"assets/generated/{out_name}",
        "prompt": data.get("prompt", ""),
        "answer": answer,
        "properties": _properties(data),
        "paths": {
            "gif": str(gif_path),
            "yaml": str(yaml_path),
            "manifest": str(base_dir / "manifest_prompt.json"),
        },
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SITE_DIR), **kwargs)

    def do_POST(self):
        if self.path != "/api/generate":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body.decode("utf-8"))
            result = _run_generation(payload)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
        except Exception as exc:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))


def main():
    host = "127.0.0.1"
    port = 8000
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving LangSLAM site at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
