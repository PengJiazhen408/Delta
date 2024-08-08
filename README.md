# Delta

Delta is a unified framework on top of DRL-based Query Optimizers.

This is the source code of "Delta: A Unified Reinforcement Learning-Based Query Optimization Framework [Scalable Data Science]"

## install

To quickly get started, run the following on one machine which will run both the leanrned quey optimizers and query execution.

### requirement

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

### dataset

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

```

3. Load the TPC-DS:
```bash

```

### extension

Install 


## prepare

### Prewarm the buffer

### run baseline and store PG plans for comparation

## Stage one

### balsa

#### train

#### test

### LOGER

#### train

#### test

## Stage two

### Delta

#### train

#### test

