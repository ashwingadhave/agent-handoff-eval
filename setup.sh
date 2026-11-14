#!/usr/bin/env bash
# Creates .venv with τ²-bench pinned to the commit used for the results.
# τ²-bench reads its task data from its own source folder, so it is installed
# from a clone in vendor/ rather than from a wheel.
set -euo pipefail

TAU2_COMMIT=c06b07dd7c4d4ec61ba6ff9bcdf0b15dd7ef36d4
PYTHON="${PYTHON:-python3.12}"  # τ²-bench needs Python 3.12 or 3.13

cd "$(dirname "$0")"
if [ ! -d vendor/tau2-bench ]; then
  git clone https://github.com/sierra-research/tau2-bench.git vendor/tau2-bench
fi
git -C vendor/tau2-bench fetch --quiet origin "$TAU2_COMMIT" 2>/dev/null || true
git -C vendor/tau2-bench checkout --quiet "$TAU2_COMMIT"

"$PYTHON" -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -e vendor/tau2-bench -r requirements.txt
echo "Ready. Activate with: source .venv/bin/activate"
