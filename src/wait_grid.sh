#!/bin/bash
# Block until at least $1 tags have finished, or the queue reports empty.
until [ "$(wc -l < logs/jobs/done.txt 2>/dev/null || echo 0)" -ge "$1" ] || grep -q "QUEUE EMPTY" logs/queue.log; do sleep 30; done
