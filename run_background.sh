#!/bin/bash
# Run experiments in background, safe from terminal hangup

VENV_PYTHON="/Users/pankajtakawale/llm/reproduce/dl/dl-reproducibility-activations/llm-reproducibility-activations/venv/bin/python"
SCRIPT_DIR="/Users/pankajtakawale/llm/reproduce/dl/dl-reproducibility-activations/llm-reproducibility-activations"

cd "$SCRIPT_DIR"

# Run with nohup - continues even if terminal closes
nohup $VENV_PYTHON run_all_experiments.py > experiments_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Save the process ID
echo $! > experiment.pid

echo "Experiments started in background!"
echo "Process ID: $(cat experiment.pid)"
echo "Log file: experiments_*.log"
echo ""
echo "To check progress:"
echo "  tail -f experiments_*.log"
echo ""
echo "To check if still running:"
echo "  ps -p \$(cat experiment.pid)"
echo ""
echo "To stop:"
echo "  kill \$(cat experiment.pid)"
