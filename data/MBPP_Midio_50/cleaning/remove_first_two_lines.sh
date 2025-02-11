#!/usr/bin/env bash
for f in /root/Thesis_project/data/MBPP_Midio_50/includes_tests/*.midio; do
  sed -i '1,2d' "$f"
done