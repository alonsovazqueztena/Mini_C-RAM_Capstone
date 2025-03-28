#!/bin/bash
# Change directory to the project’s src folder
cd "$(dirname "$0")/.."

# Start qlcplus with the specified workspace file.
# The workspace file is located in the hardware directory.
qlcplus -w -o qlight_workspace.qxw &

# Run the main Python script.
python3 run.py

