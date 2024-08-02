import psycopg2
import numpy as np
import os
import argparse
import sys
import time
parser = argparse.ArgumentParser(description='run workload')
parser.add_argument('-w', '--workload', type=str, default='JOB', help='workload to run')
parser.add_argument('-p', '--port', type=int, default='5432', help='workload to run')
sys.path.append(os.path.dirname(__file__))
from utils import *

def get_state(rel_or_index):
    sql = f'select * from pgfincore(\'{rel_or_index}\');'
    # print(sql)
    result = execute_sql(cur, sql)
    # print(result)
    print(f'{rel_or_index}: {len(result)} Segments')
    rel_os_pages = 0
    pages_mem = 0
    for i, p in enumerate(result):
        print(f"[Seg {i}] {p[3]}, {p[4]}, {p[4]/p[3] * 100: .2f}%")
        # print(p[3], p[4], p[4]/p[3] * 100)
        rel_os_pages += p[3]
        pages_mem += p[4]
    print(f"[ ALL ] {rel_os_pages}, {pages_mem}, {pages_mem/rel_os_pages * 100: .2f}%")
    # print(rel_os_pages, pages_mem, pages_mem/rel_os_pages * 100)



if __name__ == '__main__':
    # load queries
    args = parser.parse_args()
    workload = args.workload
    port = args.port

    conn = connect_db(wld2db[workload], port=port)
    cur = conn.cursor()

    relations, indexes = get_rels_and_idx(cur)
    print(f"Read {len(relations)} relations and {len(indexes)} indexes")
    
    start = time.time()
    for i in indexes:
        get_state(i)
        print("\n")
    
    for r in relations:
        get_state(r)
        print("\n")
    end = time.time()

    print(f'{(end-start) * 1000:.2f} ms')
    
    close_db_connection(conn)
    os.system("free -h")