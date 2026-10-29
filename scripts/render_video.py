"""Shared helpers for the video renderers (render_video_v3.py, render_video_v4.py): slide capture with headless Chrome and the ffmpeg
join with crossfades. The first silent skeleton cut (video/skeleton.mp4 from video/scenes.json) was retired on Oct 29 (cloud plan v2,
item 7); its files are in git history."""
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT, SHEET = REPO / "video/out.mp4", REPO / "video/out_sheet.png"      # set by the caller before assemble()
CHROME = "/usr/bin/google-chrome-stable"


def total_seconds(spec: dict) -> float:
    s = spec["scenes"]
    return sum(x["seconds"] for x in s) - spec["crossfade_s"] * (len(s) - 1)


def capture(html_path: Path, png: Path, work: Path, size) -> None:
    env = {**os.environ, "HOME": str(work / "home"), "XDG_CACHE_HOME": str(work / "cache"), "XDG_CONFIG_HOME": str(work / "config")}
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1", "--no-first-run",
                    "--allow-file-access-from-files", f"--user-data-dir={work / 'profile'}", f"--disk-cache-dir={work / 'cache'}",
                    f"--window-size={size[0]},{size[1]}", f"--screenshot={png}", html_path.as_uri()],
                   check=True, capture_output=True, env=env, timeout=120)


def assemble(pngs: list, spec: dict) -> None:
    fps, xf, sc = spec["fps"], spec["crossfade_s"], spec["scenes"]
    args = ["ffmpeg", "-y", "-loglevel", "error"]
    for p, s in zip(pngs, sc):
        args += ["-loop", "1", "-framerate", str(fps), "-t", str(s["seconds"]), "-i", str(p)]
    f = [f"[{i}:v]fps={fps},format=yuv420p,settb=AVTB[v{i}]" for i in range(len(pngs))]
    prev, acc = "v0", 0.0
    for i in range(1, len(pngs)):
        acc += sc[i - 1]["seconds"]
        f.append(f"[{prev}][v{i}]xfade=transition=fade:duration={xf}:offset={acc - i * xf:.3f}[x{i}]"); prev = f"x{i}"
    args += ["-filter_complex", ";".join(f), "-map", f"[{prev}]", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
             "-pix_fmt", "yuv420p", "-r", str(fps), "-movflags", "+faststart", str(OUT)]
    subprocess.run(args, check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(Path(pngs[0]).parent / "scene_%02d.png"), "-vf",
                    "scale=640:360,tile=4x3:padding=8:color=white", "-frames:v", "1", str(SHEET)], check=True)
