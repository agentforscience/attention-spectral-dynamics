#!/bin/bash
# Fetch CIFAR-10 parquet files (not in the local HF cache).
set -e
for s in train test; do
  curl -sL -o datasets/cifar10/$s.parquet "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/$s-00000-of-00001.parquet"
done
