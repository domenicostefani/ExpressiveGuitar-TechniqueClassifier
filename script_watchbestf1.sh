#!/bin/bash
watch -n60  echo Best Macro Average F1-Score yet: $(ls -l output/ | grep -o -P '[.]*_maf1_\d\.\d\d\d\d' | grep -o -P '\d\.\d\d\d\d' | tail -1)
