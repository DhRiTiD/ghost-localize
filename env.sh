#!/usr/bin/env bash
# usage: source env.sh
cd "$(dirname "${BASH_SOURCE[0]}")"
unset PYTHONPATH
source .venv/bin/activate
echo "ghost-localize env ready (ROS paths cleared)"
which python
