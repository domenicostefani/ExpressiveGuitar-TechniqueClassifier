#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from glob import glob
infofiles = glob('output/c_acc*/info.txt')

# print(infofiles[:5])


# In[ ]:


metrics = []
for iff in infofiles:
    with open(iff) as ifile:
        lines = ifile.readlines()
        accuracy, f1_macroavg,f1_weightedavg = (0,0,0)
        found_accuracy = False
        found_f1_macroavg = False
        found_f1_weightedavg = False
        
        for line in lines:
            if not found_accuracy and 'accuracy    ' in line:
                accuracy = float(line.split()[-2])
                found_accuracy = True
            elif not found_f1_macroavg and 'macro avg  ' in line:
                f1_macroavg = float(line.split()[-2])
                found_f1_macroavg = True
            elif not found_f1_weightedavg and 'weighted avg  ' in line:
                f1_weightedavg = float(line.split()[-2])
                found_f1_weightedavg = True

            if found_accuracy == True and found_f1_macroavg == True and found_f1_weightedavg == True:
                break 
        
        metrics.append({'file':iff,'accuracy':accuracy,'f1_macroavg':f1_macroavg,'f1_weightedavg':f1_weightedavg})


# In[ ]:


best_accuracy = max(metrics,key=lambda x: x['accuracy'])
best_f1_macroavg = max(metrics,key=lambda x: x['f1_macroavg'])
best_f1_weightedavg = max(metrics,key=lambda x: x['f1_weightedavg'])

print('Best accuracy:',best_accuracy['accuracy'],'\t',best_accuracy['file'])
print('Best f1_weightedavg:',best_f1_weightedavg['f1_weightedavg'],'\t',best_f1_weightedavg['file'])
print()
print('Best f1_macroavg:',best_f1_macroavg['f1_macroavg'],'\t',best_f1_macroavg['file'])
print()

with open(best_f1_macroavg['file']) as ifile:
    lines = ifile.readlines()
    # Search and print the classification report
    for i,line in enumerate(lines):
        if 'avg_classification_report:' in line:
            print(''.join(lines[i+1:i+12]))

