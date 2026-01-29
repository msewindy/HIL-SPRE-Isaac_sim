#!/bin/bash
# Inject venv site-packages into PYTHONPATH
export PYTHONPATH=$(pwd):$(pwd)/serl_robot_infra:$(pwd)/serl_launcher:$(pwd)/examples:$(pwd)/.venv/lib/python3.11/site-packages:$PYTHONPATH
exec /home/lfw/isaacsim/python.sh "$@"
