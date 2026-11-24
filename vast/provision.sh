#!/bin/bash
# Vast.ai PROVISIONING_SCRIPT for the vastai/pytorch image: clones MoDE into
# /workspace and runs marimo (per-notebook PEP 723 sandboxes) under supervisor.
# The template's PORTAL_CONFIG needs "localhost:2718:12718:/:marimo" so the
# Instance Portal proxies it with Vast's token auth.
set -eo pipefail

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

cd /workspace
[ -d mode ] || git clone https://github.com/wenbang24/mode.git

cat > /opt/supervisor-scripts/marimo.sh <<'SH'
#!/bin/bash
utils=/opt/supervisor-scripts/utils
. "${utils}/logging.sh"
. "${utils}/cleanup_generic.sh"
. "${utils}/environment.sh"
. "${utils}/exit_portal.sh" "marimo"

export PATH="$HOME/.local/bin:$PATH"
# Directory-mode sandboxes ignore requires-python; most notebooks pin 3.12.
# The >=3.14 ones (evaluate_piguard.py, select_*.py) need single-file mode:
#   uvx marimo@0.24.0 edit --sandbox evaluate_piguard.py
export UV_PYTHON=3.12
cd /workspace/mode
pty uvx marimo@0.24.0 edit --sandbox --headless --no-token --host 127.0.0.1 --port 12718 2>&1
SH
chmod +x /opt/supervisor-scripts/marimo.sh

cat > /etc/supervisor/conf.d/marimo.conf <<'CONF'
[program:marimo]
environment=PROC_NAME="%(program_name)s"
command=/opt/supervisor-scripts/marimo.sh
autostart=true
autorestart=unexpected
exitcodes=0
startsecs=5
stopasgroup=true
killasgroup=true
stopsignal=TERM
stopwaitsecs=10
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_events_enabled=true
stdout_logfile_maxbytes=0
stdout_logfile_backups=0
CONF

supervisorctl reread
supervisorctl update
