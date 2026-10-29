"""Renders an asciicast v3 terminal recording (asciinema rec) to an MP4 at the speed it was recorded: every frame shows the terminal as
it was at that moment, with the elapsed time in the corner. Plain line output only (ANSI codes are stripped). Needs Pillow and ffmpeg.
usage: python scripts/render_cast.py CAST OUT_MP4 [--label TEXT] [--hold S]"""
import argparse
import json
import re
import subprocess

from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SIZE, ROWS = 1920, 1080, 10, 26, 30
FONT, BOLD = "/usr/share/fonts/noto/NotoSansMono-Regular.ttf", "/usr/share/fonts/noto/NotoSansMono-Bold.ttf"
BG, FG, DIM, REQ, ACC, DEC, BAR = (17, 20, 24), (214, 218, 224), (128, 136, 148), (255, 255, 255), (120, 210, 140), (240, 190, 90), (34, 39, 46)
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07")


def events(path: str) -> list:
    """(seconds from start, output text) for every output event; v3 stores the interval since the previous event."""
    lines = open(path).read().splitlines()
    assert json.loads(lines[0]).get("version") == 3, "asciicast v3 only"
    t, out = 0.0, []
    for ln in lines[1:]:
        if ln.strip():
            dt, kind, data = json.loads(ln)
            t += dt
            if kind == "o":
                out.append((t, ANSI.sub("", data)))
    return out


def screen(text: str) -> list:
    rows = [""]
    for part in re.split(r"(\r\n|\n|\r)", text):
        if part in ("\r\n", "\n"):
            rows.append("")
        elif part == "\r":
            rows[-1] = ""
        else:
            rows[-1] += part
    return rows[-ROWS:]


def styled(rows: list) -> list:
    out, style = [], FG
    for r in rows:
        if r.startswith("Request"):
            style = REQ
        elif r.startswith("..."):
            style = DIM
        elif r.startswith("  decision: accept"):
            style = ACC
        elif r.startswith("  decision: decline"):
            style = DEC
        elif not (r.startswith("  ") and style in (REQ, ACC, DEC)):
            style = FG
        out.append((r, style))
    return out


def frame(rows: list, t: float, label: str, fonts) -> bytes:
    font, bold = fonts
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 56], fill=BAR)
    d.text((40, 14), label, font=bold, fill=FG)
    clock = f"real time, not sped up   {int(t) // 60:02d}:{int(t) % 60:02d}"
    d.text((W - 40 - d.textlength(clock, font=font), 14), clock, font=font, fill=DIM)
    y = 80
    for r, style in styled(rows):
        d.text((60, y), r, font=bold if style in (REQ, ACC, DEC) else font, fill=style)
        y += 32
    return im.tobytes()


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("cast"); ap.add_argument("out")
    ap.add_argument("--label", default="StillHere Pilot, live run"); ap.add_argument("--hold", type=float, default=3.0); A = ap.parse_args()
    ev = events(A.cast); end = ev[-1][0] + A.hold if ev else A.hold
    fonts = (ImageFont.truetype(FONT, SIZE), ImageFont.truetype(BOLD, SIZE))
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                           "-i", "-", "-vf", "fps=30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", A.out],
                          stdin=subprocess.PIPE)
    text, i, last = "", 0, None
    for n in range(int(end * FPS) + 1):
        t = n / FPS
        while i < len(ev) and ev[i][0] <= t:
            text += ev[i][1]; i += 1
        key = (len(text), int(t))
        if key != (last or (None,))[0]:
            last = (key, frame(screen(text), t, A.label, fonts))
        ff.stdin.write(last[1])
    ff.stdin.close()
    assert ff.wait() == 0, "ffmpeg failed"
    print(f"{A.out}: {end:.1f} s")


if __name__ == "__main__":
    main()
