#!/bin/bash

echo "Start aikit_main.py at $(date)" >> /home/er/aikit_log.txt 2>&1
sleep 20
mate-terminal -- bash -c "python3 /home/er/convenient_aikit/aikit_main.py; exec bash"
echo "Launched mate-terminal at $(date)" >> /home/er/aikit_log.txt 2>&1

