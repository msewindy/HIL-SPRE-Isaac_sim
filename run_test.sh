#!/bin/bash
# Source Isaac Sim environment setup (handles LD_LIBRARY_PATH etc)
source /home/lfw/isaacsim/setup_python_env.sh

# Activate our venv
source .venv/bin/activate

# Echo python path to verify
echo "Python: $(which python)"

# Execute the command
exec python "$@"
