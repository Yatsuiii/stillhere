#!/bin/bash
# Removes everything worker/install_worker.sh created: the timer and the three units, /opt/stillhere-worker, /etc/stillhere-worker (the job token) and the
# service's state directory. Leaves the repository, the model folder and the Python environments alone. Run: sudo bash worker/uninstall_worker.sh
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
for u in stillhere-worker.timer stillhere-worker.service stillhere-worker-awake.service stillhere-worker-selftest.service; do
  systemctl stop "$u" 2>/dev/null || true; systemctl disable "$u" 2>/dev/null || true; rm -f "/etc/systemd/system/$u"
done
systemctl daemon-reload
rm -rf /opt/stillhere-worker /etc/stillhere-worker /var/lib/private/stillhere-worker /var/lib/stillhere-worker
echo "removed: the timer and units, /opt/stillhere-worker, /etc/stillhere-worker, the state directory"
echo "remember to delete STILLHERE_WORKER_TOKEN from the page's server secrets"
