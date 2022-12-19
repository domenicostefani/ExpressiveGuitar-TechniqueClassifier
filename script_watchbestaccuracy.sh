#!/bin/bash
watch -n60  echo Best Accuracy yet: $(ls -l output/ | grep -o -P '[.]*_acc[_]?\d\.\d\d\d\d' | grep -o -P '\d\.\d\d\d\d' | tail -1)
