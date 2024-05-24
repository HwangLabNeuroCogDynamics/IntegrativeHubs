#####Set Scheduler Configuration Directives#####
#$ -N thalhistrsa
#$ -o /Users/kahwang/sge_logs
#$ -e /Users/kahwang/sge_logs
#$ -q SEASHORE
#$ -l mem_free=20G
#$ -t 1-59
#$ -tc 10
#$ -pe smp 10
##### 1-4740
#!/bin/bash
source activate py3.8
subjects=(10001 10002 10003 10004 10005 10008 10010 10012 10013 10014 10017 10018 10019 10020 10022 10023 10024 10025 10027 10028 10031 10032 10033 10034 10035 10036 10037 10038 10039 10040 10041 10042 10043 10044 10054 10057 10058 10059 10060 10063 10064 10066 10068 10069 10071 10072 10073 10074 10076 10077 10080 10162 10169 10170 10173 10174 10175 10176 10179)

# select subject
#echo ${SGE_TASK_ID}
sub=${subjects[${SGE_TASK_ID}-1]}
echo ${sub}

#### run script
echo $sub | python /Users/kahwang/bin/IntegrativeHubs/trial_rsa.py
echo $sub | python /Users/kahwang/bin/IntegrativeHubs/trial_regression.py