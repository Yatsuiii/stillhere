#!/bin/bash
# Installs the StillHere live worker as a sandboxed systemd service (docs/worker.md). Run once, from your own account, with sudo:
#   sudo bash worker/install_worker.sh https://<page host>
# What it does, in order (it stops at the first failure and changes nothing it has not announced):
#  1. checks the inputs exist and audits the three folders the service will see: the worker's model folder holds only model weights,
#     configs and tokenizers; site-packages and the spaCy model hold no audio, video or brain-imaging file that a published wheel does
#     not ship, nothing with a private-data mark, and no symlink leaving the folder;
#  2. copies the Python 3.11 interpreter (from your uv folder, which the sandbox hides with the rest of /home) and the worker's own code
#     and clips (worker/, stillhere/conditions.py, data/live_clips*) into /opt/stillhere-worker, owned by root, read-only;
#  3. creates the job token in /etc/stillhere-worker/job_token (root, 0600) unless one exists; the service receives it as a credential;
#  4. renders stillhere-worker.service and stillhere-worker-selftest.service from worker/stillhere-worker.service.in;
#  5. runs the self-test INSIDE the sandbox and refuses to finish unless it passes (hidden paths cannot be listed, the bound folders
#     are read-only, the GPU is visible).
#  6. enables ONLY stillhere-worker.timer (10:00 America/New_York, Nov 4-9, Persistent=true); the worker stops itself at 14:00 ET, and
#     stillhere-worker-awake.service holds a sleep inhibitor while it runs.
# It does not start the worker; the commands for that are printed at the end. Undo everything: sudo bash worker/uninstall_worker.sh
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
URL=${1:-}
[[ $URL =~ ^https://[A-Za-z0-9.-]+(/[A-Za-z0-9._/-]*)?$ ]] || { echo "usage: sudo bash worker/install_worker.sh https://<page host>"; exit 1; }
U=${SUDO_USER:?run this with sudo from your own account}
UHOME=$(getent passwd "$U" | cut -d: -f6)
REPO=$(cd "$(dirname "$0")/.." && pwd -P); WORK=$(dirname "$REPO")
PYSRC=$(ls -d "$UHOME"/.local/share/uv/python/cpython-3.11.*-linux-x86_64-gnu 2>/dev/null | head -1)
MODELS=$WORK/.worker-models; SITE=$WORK/.venvs/tribe/lib/python3.11/site-packages; SPACY=$WORK/.venvs/spacy-en-site
OPT=/opt/stillhere-worker; ETC=/etc/stillhere-worker; UNIT=/etc/systemd/system
HIDDEN="$WORK/stillhere/d21/ds003171:$WORK:$(dirname "$WORK"):$UHOME:/home:/run/media/$U"

echo "1/5 checks and audit"
for p in "$MODELS/hub" "$SITE/tribev2" "$SPACY/en_core_web_lg" "$PYSRC/bin/python3.11" "$REPO/data/live_clips.json" /dev/nvidia0 /dev/nvidiactl; do
  [ -e "$p" ] || { echo "missing: $p"; exit 1; }
done
AUDIT=$(sudo -u "$U" python3 "$REPO/worker/link_models.py" --audit "$MODELS" "$SITE" "$SPACY") || { echo "audit failed:"; echo "$AUDIT"; exit 1; }
echo "   audit passed (models, site-packages, spaCy)"

echo "2/5 interpreter and worker code into $OPT (root, read-only)"
rm -rf "$OPT"
install -d -m 0755 "$OPT" "$OPT/models" "$OPT/site-packages" "$OPT/spacy" "$OPT/app" "$OPT/app/worker" "$OPT/app/stillhere" "$OPT/app/data"
cp -a "$PYSRC" "$OPT/python"
install -m 0644 "$REPO"/worker/{recipe.py,engine.py,tribe_worker.py,labels.py,regions.json} "$OPT/app/worker/"
install -m 0644 "$REPO/stillhere/__init__.py" "$REPO/stillhere/conditions.py" "$OPT/app/stillhere/"
install -m 0644 "$REPO/data/live_clips.json" "$OPT/app/data/"
cp -r "$REPO/data/live_clips" "$OPT/app/data/"
chown -R root:root "$OPT"; chmod -R u+rwX,go+rX,go-w "$OPT"

echo "3/5 job token"
install -d -m 0700 "$ETC"
if [ ! -s "$ETC/job_token" ]; then (umask 077; head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$ETC/job_token"); fi
chown root:root "$ETC/job_token"; chmod 0600 "$ETC/job_token"

echo "4/5 units"
render() {   # role type command restart wants -> stdout
  sed -e "s#@ROLE@#$1#" -e "s#@TYPE@#$2#" -e "s#@COMMAND@#$3#" -e "s#@RESTART@#$4#" -e "s#@WANTS@#$5#" -e "s#@MODELS@#$MODELS#" \
      -e "s#@SITE@#$SITE#" -e "s#@SPACY@#$SPACY#" -e "s#@JOBS_URL@#$URL#" -e "s#@HIDDEN@#$HIDDEN#" "$REPO/worker/stillhere-worker.service.in"
}
render "serve" simple serve on-failure stillhere-worker-awake.service > "$UNIT/stillhere-worker.service"
render "self-test" oneshot self-test no "" > "$UNIT/stillhere-worker-selftest.service"
install -m 0644 "$REPO/worker/stillhere-worker-awake.service" "$REPO/worker/stillhere-worker.timer" "$UNIT/"
if grep -q "@[A-Z_]*@" "$UNIT/stillhere-worker.service" "$UNIT/stillhere-worker-selftest.service"; then echo "unrendered field"; exit 1; fi
systemd-analyze verify "$UNIT/stillhere-worker.service" "$UNIT/stillhere-worker-selftest.service" "$UNIT/stillhere-worker-awake.service" \
  "$UNIT/stillhere-worker.timer"
systemctl daemon-reload

echo "5/5 self-test inside the sandbox"
if systemctl start stillhere-worker-selftest.service; then st=0; else st=1; fi
journalctl -u stillhere-worker-selftest.service -n 30 --no-pager -o cat | grep -E "PASS|FAIL|info:" || true
[ $st = 0 ] || { echo "SELF-TEST FAILED: the worker is installed but must not be started; fix it or run worker/uninstall_worker.sh"; exit 1; }

echo "enabling the timer only (10:00 ET, Nov 4-9; the worker stops itself at 14:00 ET)"
systemctl enable --now stillhere-worker.timer
systemctl list-timers stillhere-worker.timer --no-pager | head -3

cat <<EOF

Installed. Next steps (the worker itself is not running; the timer starts it at 10:00 ET on Nov 4-9):
  - put the job token into the page's server secrets as STILLHERE_WORKER_TOKEN yourself:  sudo cat $ETC/job_token
  - start by hand (inside 10:00-14:00 ET):   sudo systemctl start stillhere-worker      stop:  sudo systemctl stop stillhere-worker
  - outside the hours (e.g. filming): echo STILLHERE_HOURS_ET=00:00-24:00 | sudo tee $ETC/worker.env, start, then sudo rm $ETC/worker.env
  - logs (counts and timings only):   journalctl -u stillhere-worker -f
  - rerun the self-test any time:  sudo systemctl start stillhere-worker-selftest && journalctl -u stillhere-worker-selftest -n 20 -o cat
  - undo everything:  sudo bash $REPO/worker/uninstall_worker.sh
EOF
