"""SAM 3.1 on one allowlisted public film frame (Tears of Steel, CC BY 3.0) through Meta's Model API: the frame goes as a one-frame
H.264 clip at the bridge cut's 640x360 letterbox geometry (egress.PublicFrame checks its SHA-256), the concept as a scanned
egress.PublicQuestion, to api.meta.ai only; the uploaded file is deleted after. The stream format and parser follow the SAM probe
(pilot/probes/sam31/run_sam31_api.py)."""
import json
import os
import re

import httpx

from stillhere import egress, meta_client as M

REC = re.compile(r"(\d+)<\|box;x1=(\d+);y1=(\d+);x2=(\d+);y2=(\d+);w=(\d+);h=(\d+)(?:;c=([0-9.eE+-]+))?\|>"
                 r"<\|mask;x=0;y=0(?:;c=[0-9.eE+-]+)?;data=(\d+),(\d+),(.)([^|]*)\|>")


def parse_line(line: str) -> tuple:
    m = re.match(r"<(\d+)f>", line)
    if not m:
        raise ValueError("line without a frame marker")
    objs = []
    for r in REC.finditer(line, m.end()):
        x1, y1, x2, y2 = (int(r.group(i)) for i in (2, 3, 4, 5))
        objs.append({"id": int(r.group(1)), "score": float(r.group(8)) if r.group(8) else None, "box_xyxy": [x1, y1, x2, y2],
                     "mask": {"H": int(r.group(9)), "W": int(r.group(10)), "enc": r.group(11), "payload": r.group(12)}})
    return int(m.group(1)), objs


def segment(clip_path: str, concept: egress.PublicQuestion, timeout: float = 120.0) -> dict:
    """{"objects": [...], "usage": {...}, "status": str} for frame 0 of an allowlisted one-frame clip."""
    path = egress.PublicFrame(path=clip_path).path
    if not isinstance(concept, egress.PublicQuestion):
        raise egress.EgressRefused("the SAM concept must be a scanned egress.PublicQuestion")
    if httpx.URL(M.API).host not in M.HOSTS:
        raise egress.EgressRefused("host not on the allowlist")
    auth = {"Authorization": f"Bearer {M._key()}"}
    up = httpx.post(M.API + "/files", headers=auth, data={"purpose": "user_data"},
                    files={"file": (os.path.basename(path), open(path, "rb"), "video/mp4")}, timeout=timeout)
    up.raise_for_status(); fid = up.json()["id"]
    body = {"model": "sam-3.1", "stream": True, "metadata": {"include_confidence": "true"},
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": concept.text},
                                                                      {"type": "input_video", "file_id": fid}]}]}
    lines, usage, status = [], None, None
    try:
        with httpx.stream("POST", M.API + "/responses", headers=auth, json=body, timeout=httpx.Timeout(timeout, read=timeout)) as r:
            r.raise_for_status()
            for raw in r.iter_lines():
                data = raw[5:].strip() if raw.startswith("data:") else ""
                if not data.startswith("{"):
                    continue
                ev = json.loads(data)
                if ev.get("type") == "response.output_text.delta":
                    lines.extend(l for l in ev["delta"].splitlines() if l.strip())
                elif ev.get("type") in ("response.completed", "response.failed", "response.incomplete"):
                    status = ev["type"]; usage = (ev.get("response") or {}).get("usage")
    finally:
        httpx.delete(f"{M.API}/files/{fid}", headers=auth, timeout=60)
    objs = []
    for line in lines:
        k, o = parse_line(line)
        if k == 0:
            objs.extend(o)
    return {"objects": objs, "usage": usage, "status": status}
