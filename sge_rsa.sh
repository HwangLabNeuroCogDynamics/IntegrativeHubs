#####Set Scheduler Configuration Directives#####
#$ -N thalhistrsa
#$ -o /Users/kahwang/sge_logs
#$ -e /Users/kahwang/sge_logs
#$ -q SEASHORE
#$ -l mem_free=10G
#$ -t 1-59
#$ -pe smp 2
##### 1-4740
#!/bin/bash
source activate py3.8
subjects=(128 112 108 110 120 98 86 82 115 94 76 91 80 95 121 114 125 70 107 111 88 113 131 130 135 140 167 145 146 138 147 176 122 118 103 142)

# select subject
echo ${SGE_TASK_ID}
sub=${subjects[${SGE_TASK_ID}-1]}

#### run script
#echo $sub | python /Users/kahwang/bin/ThalHiEEG/run_LDA.py
#echo $sub | python /Users/kahwang/bin/ThalHiEEG/run_RSA.py
echo $sub | python /Users/kahwang/bin/ThalHiEEG/GC_RSA_ts.py
