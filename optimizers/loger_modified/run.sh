# # conda 
# conda activate loger


# # JOB RS
# # prewarm
# cd /home/postgres/data/workloads
# bash run.sh JOB rs 5432

# cd /home/postgres/data/optimizer/LOGER_extend/LOGER-main
# # python train.py -d 'dataset/job_rs_train' 'dataset/job_rs_test' -e 200 -F 1 -D imdb -U postgres --port 5432

# # nohup python train.py -d 'dataset/job_rs_train' 'dataset/job_rs_test' -e 200 -F 1 -D imdb -U postgres --port 5432 >> my.log 2>&1 &
# # nohup python train.py -d 'dataset/job_rs_train' 'dataset/job_rs_test' -e 200 -F 1 -D imdb -U postgres --port 5432 >> my2.log 2>&1 &
nohup python train.py -d 'dataset/job_rs_train' 'dataset/job_rs_test' -e 200 -F 1 -D imdb -U postgres --port 5432 >> job_rs.log 2>&1 &
nohup python train.py -d 'dataset/job_rs_train' 'dataset/job_rs_test' -e 200 -F 1_t2 -D imdb -U postgres --port 5432 >> job_rs_t2.log 2>&1 &


# JOB RS top10
# prewarm
cd /home/postgres/data/workloads
bash run.sh JOB rs

# cd /home/postgres/data/optimizer/LOGER_extend/LOGER-main
# python val_job_rs_top10.py -d 'dataset/job_rs_train' 'dataset/job_rs_test' -e 200 -F 3 -D imdb -U postgres --port 5432
nohup python val_job_rs_top10.py -d 'dataset/job_rs_train' 'dataset/job_rs_test' -e 200 -F 3 -D imdb -U postgres --port 5432 >> job_rs_top10_2.log 2>&1 &

nohup python val_job_rs_top10.py -d 'dataset/job_slow_train' 'dataset/job_slow_test' -e 200 -F 4 -D imdb -U postgres --port 5432 >> job_slow_top10_1.log 2>&1 &


# # JOB Slow
# cd /home/postgres/data/workloads
# bash run.sh JOB slow 5432

# cd /home/postgres/data/optimizer/LOGER_extend/LOGER-main
# nohup python train.py -d 'dataset/job_slow_train' 'dataset/job_slow_test' -e 200 -F 2 -D imdb -U postgres --port 5432 >> job_slow_1.log 2>&1 &


# JOB Slow top10
