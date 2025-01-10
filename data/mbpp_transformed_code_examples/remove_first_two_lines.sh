#!/usr/bin/env bash
for f in /root/Thesis_project/data/mbpp_transformed_code_examples/includes_tests/*.midio; do
  sed -i '1,2d' "$f"
done