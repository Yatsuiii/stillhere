"""The judge scripts' refusals and the local GPU launcher's restart (docs/RUN_ON_YOUR_GPU.md), with stand-in pythons: no GPU, no model,
no network. scripts/judge_gpu_setup.sh, scripts/run_local_gpu.sh and scripts/reproduce_paranoia.sh."""
import os
import shutil
import stat
import subprocess
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("bash") is None or shutil.which("realpath") is None, reason="needs bash and coreutils")


def run(script, *args, env=None, timeout=60):
    e = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/nonexistent"), **(env or {})}
    return subprocess.run(["bash", str(REPO / "scripts" / script), *map(str, args)], cwd=REPO, env=e, capture_output=True, text=True,
                          timeout=timeout)


def stub(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env bash\n" + body); path.chmod(0o755); return path


def test_setup_refuses_a_models_dir_inside_the_repository_and_creates_nothing():
    p = run("judge_gpu_setup.sh")
    assert p.returncode == 2 and "usage" in p.stdout
    inside = REPO / f"models-{uuid.uuid4().hex[:8]}"
    p = run("judge_gpu_setup.sh", inside)
    assert p.returncode == 2 and "outside the repository" in p.stdout and not inside.exists()


def test_reproduce_refuses_inside_the_repository_and_checks_the_scorers_packages_before_any_network(tmp_path):
    p = run("reproduce_paranoia.sh", REPO / "data_here")
    assert p.returncode == 2 and "outside this repository" in p.stderr
    bad = stub(tmp_path / "py", "exit 1\n")                                          # a python without the scorers' packages
    p = run("reproduce_paranoia.sh", "--no-gpu", tmp_path / "data", env={"STILLHERE_SCORE_PYTHON": str(bad), "http_proxy": "http://0.0.0.0:1",
                                                                           "https_proxy": "http://0.0.0.0:1"})
    assert p.returncode == 2 and "refused before any download" in p.stderr and "requirements-score.txt" in p.stderr
    assert not (tmp_path / "data").exists()


def test_local_gpu_launcher_needs_the_setup_and_keeps_the_token_out_of_tmp(tmp_path):
    p = run("run_local_gpu.sh")
    assert p.returncode != 0 and "judge_gpu_setup.sh first" in p.stderr
    env = {"STILLHERE_TRIBE_PYTHON": "x", "STILLHERE_MODELS_DIR": "x", "STILLHERE_STATE_DIR": "/tmp/stillhere-state-test"}
    p = run("run_local_gpu.sh", env=env)
    assert p.returncode == 2 and "out of /tmp" in p.stdout and not Path("/tmp/stillhere-state-test").exists()


def test_local_gpu_launcher_restarts_the_worker_on_exit_75_and_stops_on_its_end(tmp_path):
    state = Path.home() / ".cache" / f"stillhere-test-{uuid.uuid4().hex[:8]}"           # not /tmp: the launcher refuses it there
    count = tmp_path / "count"
    web = stub(tmp_path / "web", 'case "$1" in\n  -c) case "$2" in *secrets*) echo ' + "ab" * 32 + ';; *) exit 0;; esac;;\n'
                                 "  web/serve.py) exec sleep 30;;\nesac\n")
    tribe = stub(tmp_path / "tribe", f'n=$(cat {count} 2>/dev/null || echo 0); n=$((n+1)); echo $n > {count}\n'
                                     '[ "$n" -lt 2 ] && exit 75 || exit 0\n')
    try:
        p = run("run_local_gpu.sh", env={"STILLHERE_TRIBE_PYTHON": str(tribe), "STILLHERE_MODELS_DIR": str(tmp_path),
                                         "STILLHERE_WEB_PYTHON": str(web), "STILLHERE_STATE_DIR": str(state)})
        assert p.returncode == 0, p.stdout + p.stderr
        assert count.read_text().strip() == "2" and "starting a fresh worker" in p.stdout     # exit 75 once, then a clean end
        assert stat.S_IMODE((state / "job_token").stat().st_mode) == 0o600 and stat.S_IMODE(state.stat().st_mode) == 0o700
        assert "ab" * 32 not in p.stdout + p.stderr                                            # the token is never printed
    finally:
        shutil.rmtree(state, ignore_errors=True)
