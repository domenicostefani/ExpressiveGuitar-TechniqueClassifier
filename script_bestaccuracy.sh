echo Best Accuracy yet: $(ls -la output/ | grep -o -P "full_c_acc\d\.\d\d\d\d" | grep -o -P "\d\.\d\d\d\d" | tail -1)
