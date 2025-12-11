#!/bin/bash
# Monitor training progress in real-time

echo "Monitoring training progress..."
echo "Press Ctrl+C to stop monitoring (training will continue)"
echo ""

tail -f training.log 2>/dev/null &
TAIL_PID=$!

# Also show GPU usage if available
while true; do
    sleep 10
    clear
    echo "=== Latest Training Progress ==="
    tail -20 training.log 2>/dev/null || echo "No log file yet..."
    echo ""
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null || echo "GPU info not available"
    echo ""
    echo "=== Results Files ==="
    ls -lh results/*.json 2>/dev/null | tail -5 || echo "No results yet..."
done

kill $TAIL_PID 2>/dev/null
