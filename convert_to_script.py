#!/usr/bin/env python3

# This script converts our Colab/Jupyter notebook to a python script
# It is used to run the notebook via tmux
#
# It uses jupyter nbconvert and then removes all the ipython magic

import os
import re
import json
def escape_double_quotes(instring):
    # print(instring)
    res = instring #json.dumps(string)[1:-1]

    return res.replace('\\\\\'','<QUOTE>').replace('\\n','<SPACE>').replace('\\t','<TAB>').replace('\\','').replace('<SPACE>','\n').replace('<TAB>','\t').replace('<QUOTE>','\'')

NOTEBOOK_PATH = "expressive-technique-classifier-phase3.ipynb"

command = 'jupyter nbconvert --to script '+NOTEBOOK_PATH
print('running command: '+command)
os.system(command)


script_lines = []
with open(os.path.splitext(NOTEBOOK_PATH)[0]+'.py') as sf:
    for e in sf.readlines():
        if 'run_cell_magic' in e:
            # e = e.replace('get_ipython().run_cell_magic(', '')
            # e = e.replace(re.findall('^\'\w+\',[ ]*\'[\w._]+\',[ ]*\'',e)[0], '')
            # e = e.replace(re.findall('\'\)$',e)[0], '')
            # print(escape_double_quotes(e))
            e = escape_double_quotes(e)
            e = e.replace('get_ipython().run_cell_magic(\'write_and_run\', \'feature_selection.txt\', \'', '')
            e = e.replace("get_ipython().run_cell_magic('write_and_run', 'model_architecture_code.txt', '","")
            assert e.rstrip()[-2:] == r"')"
            e = e.rstrip()[:-2]
            script_lines.append(escape_double_quotes(e))
        elif 'COLAB = ' in e:
            script_lines.append('COLAB = False')
        elif 'ipython' in e.lower():
            script_lines.append('#<redacted ipython line>'+e)
            script_lines.append(' '*(len(e)-len(e.lstrip()))+'None\n')
        elif e.strip().startswith('@'):
            script_lines.append('#<redacted decorator line>'+e)
            script_lines.append(' '*(len(e)-len(e.lstrip()))+'None\n')
        elif e.strip().startswith('get_ipython'):
            script_lines.append('#<redacted ipython line>'+e)
            script_lines.append(' '*(len(e)-len(e.lstrip()))+'None\n')
        elif e.strip().startswith('!') or  e.strip().startswith('%'):
            script_lines.append('#<redacted command>'+e)
            script_lines.append(' '*(len(e)-len(e.lstrip()))+'None\n')
        else:
            script_lines.append(e)
    
with open(os.path.splitext(NOTEBOOK_PATH)[0]+'.py','w') as sf:
    for e in script_lines:
        sf.write(e)