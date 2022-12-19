echo Best Accuracy yet: $(ls -la output/ | grep -o -P "_acc[_]?\d\.\d\d\d\d" | grep -o -P "\d\.\d\d\d\d" | tail -1)
