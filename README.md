# Delta

Delta is a unified framework on top of DRL-based Query Optimizers.

This is the source code of "Delta: A Unified Reinforcement Learning-Based Query Optimization Framework [Scalable Data Science]"

## Install

To quickly get started, run the following on one machine which will run both the leanrned quey optimizers and query execution.

### Requirement

Clone and install Delta

```bash
git clone https://github.com/PengJiazhen408/Delta.git
cd Delta
```

Install the requirements of Balsa

```bash
conda create -n balsa python=3.7 -y
conda activate balsa

cd optimizers/balsa_modified
pip install -r requirements.txt
pip install -e .
pip install -e pg_executor
```

Install the requirements of LOGER

```bash
conda create -n LOGER python=3.7 -y
conda activate LOGER

cd optimizers/loger_modified
pip install -r requirements.txt
```

### PostgreSQL

Install the PostgreSQL 12.5 from source by running following commands.
Remeber replace $YOUR_PG_LOCATION and $PSQL_DATA_DIRECTORY by your configures.

```bash
cd ~/
wget https://ftp.postgresql.org/pub/source/v12.5/postgresql-12.5.tar.gz
tar xzvf postgresql-12.5.tar.gz
cd postgresql-12.5
./configure --prefix=$YOUR_PG_LOCATION --enable-depend --enable-cassert --enable-debug CFLAGS="-ggdb -O0"
make 
make install

echo 'export PATH=$YOUR_PG_LOCATION/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$YOUR_PG_LOCATION/lib/:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
initdb -D $PSQL_DATA_DIRECTORY

# start server
pg_ctl start -D $PSQL_DATA_DIRECTORY
```

Install the pg_hint_plan extension v1.3.7

```bash
cd ~/
git clone https://github.com/ossc-db/pg_hint_plan.git -b REL12_1_3_7
cd pg_hint_plan
make
make install
```

Modify the PostgeSQL configuration
Add the following configures to the tail of  $PSQL_DATA_DIRECTORY/postgres.conf

```bash
shared_buffers = 96GB # the reasonable initial value for shared_buffers is 25% of the memory on your system(376G).
effective_cache_size = 192GB # the reasonable initial value for effective_cache_size is 50% of the memory on your system(376G).
enable_bitmapscan = off
enable_tidscan = off
max_worker_processes = 8		# (change requires restart)
max_parallel_maintenance_workers = 8	# taken from max_parallel_workers
max_parallel_workers_per_gather = 8	# taken from max_parallel_workers
max_parallel_workers = 8
geqo = off
shared_preload_libraries = 'pg_hint_plan'
```

Restart the PostgreSQL

```bash
pg_ctl restart -D $PSQL_DATA_DIRECTORY
```

### Dataset

1. Load the Join Order Benchmark (JOB) tables: (following Balsa)

```bash
mkdir -p datasets/job && pushd datasets/job
wget -c http://homepages.cwi.nl/~boncz/job/imdb.tgz && tar -xvzf imdb.tgz && popd
# Prepend headers to CSV files
conda activate balsa
cd  ~/optimizers/balsa_modified
python3 scripts/prepend_imdb_headers.py
bash load-postgres/load_job_postgres.sh ../../datasets/job
```

2. Load the STACK:

```bash
mkdir -p datasets/stack && pushd datasets/stack
wget https://rmarcus.info/stack.html
pg_restore -U postgres -d stack -v so_pg13
popd
```

3. Load the TPC-DS:

```bash
mkdir -p datasets/tpcds && pushd datasets/tpcds
mkdir -p tpcds_data
wget https://www.tpc.org/TPC_Documents_Current_Versions/download_programs/tools-download-request5.asp?bm_type=TPC-DS&bm_vers=3.2.0&mode=CURRENT-ONLY
unzip TPC-DS-Tool.zip

pushd TPC-DS-Tool/tools

cp Makefile.suite Makefile

make

./dsdgen -scale 10GB -dir ../../tpcds_data  -TERMINATE N

psql -c "create database tpcds"

psql -d tpcds -f tpcds.sql

psql -d tpcds -f ../../../scripts/tpcds/load_data.sql

# if found: psql:../../../scripts/tpcds/load_data.sql:5: ERROR:  invalid byte sequence for encoding "UTF8": 0xd4 0x54
#           CONTEXT:  COPY customer, line 28
# then: 
#    psql -d tpcds
#    copy customer from '/home/postgres/data/datasets/tpcds_data/customer.dat' with delimiter as '|' NULL '';
#    \q

psql -d tpcds -f tpcds_ri.sql

psql -d tpcds -f ../../../scripts/tpcds/add_fkindex.sql

popd

popd
```

### Extension

1. Install extension pg_prewarm

```bash
cd ~/postgresql-12.5/contrib/pg_prewarm
make
make install

psql -d imdb -c create extension pg_prewarm
psql -d stack -c create extension pg_prewarm
psql -d tpcds -c create extension pg_prewarm
```

2. Install extension pgfincore

```bash
cd ~/
git clone git://git.postgresql.org/git/pgfincore.git 
cd pgfincore
make
make install

psql -d imdb -c create extension pgfincore
psql -d stack -c create extension pgfincore
psql -d tpcds -c create extension pgfincore
```

## Prepare

### Prewarm the buffer

```bash
cd ./workloads
conda activate balsa

# If run the JOB rs workload:
bash run.sh JOB rs 

# If run the JOB slow workload:
bash run.sh JOB slow

# If run the tpc-ds workload:
bash run.sh tpcds

# If run the stack workload:
bash run.sh stack
```

### Run baseline and store PG plans for comparation

```bash
cd ./workloads
conda activate balsa

# If run the JOB rs workload:
python run_workload_store_plans.py -w JOB -s rs

# If run the JOB slow workload:
python run_workload_store_plans.py -w JOB -s slow

# If run the tpc-ds workload:
python run_workload_store_plans.py -w tpcds

# If run the stack workload:
python run_workload_store_plans.py -w stack
```

The key output files:

- The executed plans generated by PostgreSQL for test queries are saved in `./workloads/Plans`

Copy  `pg_exec` and `pg_e2e` dict from the output of `run_workload_store_plans.py`  to

1. `./optimizers/balsa_modified/test_runtime.py`
2. `./Delta/test.py`
   for computing metrics like `GMRL` and `WRL`.

## Stage one

If you run Balsa-Delta, only need to run Balsa in the stage one.

If you run LOGER-Delta, only need to run LOGER in the stage one.

### Balsa

We made the following modifications to the source code of [Balsa](https://github.com/balsa-project/balsa) to support Delta:

1. We saved the executed plans for training according to their respective iteration round \(i\) in: `./optimizers/balsa_modified/train_history/${run_id}/${i}`.
2. We saved the model parameters of the agent trained in each iteration round to: `./optimizers/balsa_modified/qo_models/${run_id}`.
3. We modified parts of the code to support the TPC-DS and STACK datasets. Special thanks to Tianyi Chen, the author of [LOGER](https://www.vldb.org/pvldb/vol16/p1777-gao.pdf), for providing us with the modified Balsa code.
4. We implement the code of generating and executing the top plans for test queries based on a trained Balsa agent in `test_runtime.py`, and the executed plans for testing are stored in `./optimizers/balsa_modified/test_history/${run_id}/${i}`.

#### Train

Train the agent of Balsa.

```bash

cd ./optimizers/balsa_modified

# If run the JOB rs workload:
python run.py --run JOBRandSplit_PostgresSim --local

# If run the JOB slow workload:
python run.py --run JOBSlowSplit_PostgresSim --local

# If run the tpc-ds workload:
python run.py --run TPCDS_PostgresSim --local
```

The key output files:

1. The executed plans to train the agent: `./optimizers/balsa_modified/train_history/${run_id}/${i}`
2. The trained agents: `./optimizers/balsa_modified/qo_models/${run_id}`

#### Test

Generate and execute the top 10 plans for test queries based on each trained Balsa agent in `qo_models/${run_id}`,

```bash

cd ./optimizers/balsa_modified

# If run the JOB rs workload:
python test_runtime.py --run JOBRandSplit_PostgresSim --exp_id ${exp_id}

# If run the JOB slow workload:
python test_runtime.py --run JOBSlowSplit_PostgresSim --exp_id ${exp_id}

# If run the tpc-ds workload:
python test_runtime.py --run TPCDS_PostgresSim --exp_id ${exp_id}
```

The `exp_id` is the random set by `wandb` in the training phase.

The key output files:

- The executed top-10 plans generated by Balsa for test queries : `./optimizers/balsa_modified/test_history/${run_id}/${i}`.

### LOGER

#### Train

```bash
cd ./optimizers/loger_modified

# If run the JOB rs workload:
python train.py -d 'dataset/job_rs_train' 'dataset/job_rs_test' -e 200 -F 1 -D imdb -U postgres --port 5432 --mode job_rs

# If run the JOB slow workload:
python train.py -d 'dataset/job_slow_train' 'dataset/job_slow_test' -e 200 -F 1 -D imdb -U postgres --port 5432 --mode job_slow

# If run the tpc-ds workload:
python train.py -d 'dataset/tpcds_train' 'dataset/tpcds_test' -e 200 -F 1 -D tpcds -U postgres --port 5432 --mode tpcds

# If run the stack workload:
python train.py -d 'dataset/stack_train' 'dataset/stack_test' -e 200 -F 1 -D stack -U postgres --port 5432 --mode stack
```

#### Test

```bash
cd ./optimizers/loger_modified

# If run the JOB rs workload:
python test.py -d 'dataset/job_rs_train' 'dataset/job_rs_test' -e 200 -F 1 -D imdb -U postgres --port 5432 --mode job_rs

# If run the JOB slow workload:
python test.py -d 'dataset/job_slow_train' 'dataset/job_slow_test' -e 200 -F 1 -D imdb -U postgres --port 5432 --mode job_slow

# If run the tpc-ds workload:
python test.py -d 'dataset/tpcds_train' 'dataset/tpcds_test' -e 200 -F 1 -D tpcds -U postgres --port 5432 --mode tpcds

# If run the stack workload:
python test.py -d 'dataset/stack_train' 'dataset/stack_test' -e 200 -F 1 -D stack -U postgres --port 5432 --mode stack
```

## Stage two

### Delta

#### Copy the intermediate results generated in the stage one

```bash

# 1. copy plans generated by PostgreSQL for test queries
cp -r ./workloads/Plans ./Delta/PG_Plans

# 2. copy training plans from stage one 
# if balsa:
cp -r ./optimizers/balsa_modified/train_history ./Delta/train_history

# if LOGER:
cp -r ./optimizers/loger_modified/train_history ./Delta/train_history

# 3. copy test plans from stage one
# if balsa:
cp -r ./optimizers/balsa_modified/test_history ./Delta/test_history

# if LOGER:
cp -r ./optimizers/loger_modified/test_history ./Delta/test_history

```

#### Train

```bash
python train.py --qo_name $qo_name

```

`$qo_name` is Balsa_$dataset/$exp_id or LOGER_$dataset/$exp_id which consistent with the name in train_history, such as, Balsa_JOB_slow/d8awdme0

The key output files:

- The trained models in stage two is saved in `./Delta/topk_models`.

#### Test

```bash
python test.py --qo_name $qo_name

```

`$qo_name` is Balsa_$dataset/$exp_id or LOGER_$dataset/$exp_id which consistent with the name in test_history, such as, Balsa_JOB_slow/d8awdme0

The key output files:

- The trained models in stage two is saved in `./Delta/results`.
