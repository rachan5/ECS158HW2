#! /usr/bin/env python

import subprocess
import sys
import re

#subprocess.call("ls -l", shell=True)

trial_num = 1
#for trial_num in range (88,101):
num_leaves = [10]*6
for i in range(1,6):
    num_leaves[i] = 10**i

for x in num_leaves:
    proc = subprocess.Popen(['./ms', str(x), '1', '-T'], stdout=subprocess.PIPE) # generate test file using ms
    print ('generating test file with ' + str(x) + ' leaves\n')
    #proc.communicate()
    f = open('input' + str(x) + '.tre', 'w')                                     # create file to store output
    line = proc.stdout.readlines()
    line_count = 0
    for lines in line:                                                           # parse output
        line_count +=1
        if (line_count == 5):
            f.write(lines)                                                  # store in file
    f.close()
    #print("running gus on " + str(x) + " leaves...\n")
    proc = subprocess.call(['Rscript', 'testmaker.R', 'input' + str(x) + '.tre', 'test' + str(x) + '.tre'] ) 
    #subprocess.call(['gurobi_cl', 'resultfile=' + "gus_" + str(x) + '_trial_' + str(trial_num) + ".sol", 'trial' + str(trial_num) + 'input' + str(x) + '.lp'])    # run gurobi 
