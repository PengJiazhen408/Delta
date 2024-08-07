import psycopg2
import numpy as np
import os
import argparse
import sys
parser = argparse.ArgumentParser(description='run workload')
parser.add_argument('-w', '--workload', type=str, default='JOB', help='workload to run')
parser.add_argument('-p', '--port', type=int, default='5432', help='workload to run')
sys.path.append(os.path.dirname(__file__))
from utils import *

def pre_load(rel_or_index):
    sql = f'select * from pgfadvise_willneed(\'{rel_or_index}\');'  # pre load os cache
    print(sql)
    execute_sql(cur, sql)
    sql = f'select pg_prewarm(\'{rel_or_index}\', \'buffer\', \'main\');' # pre load shared buffer
    print(sql)
    execute_sql(cur, sql)


if __name__ == '__main__':
    # load queries
    os.system("free -h")
    args = parser.parse_args()
    workload = args.workload
    port = args.port

    conn = connect_db(wld2db[workload], port=port)
    cur = conn.cursor()

    relations, indexes = get_rels_and_idx(cur)
    print(f"Read {len(relations)} relations and {len(indexes)} indexes")
    
    for i in indexes:
        pre_load(i)
    
    for r in relations:
        pre_load(r)
    
    close_db_connection(conn)
    os.system("free -h")