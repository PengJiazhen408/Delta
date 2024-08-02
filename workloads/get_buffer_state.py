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

def get_os_cache_state(cur, rel_or_index):
    sql = f'select * from pgfincore(\'{rel_or_index}\');'
    # print(sql)
    result = execute_sql(cur, sql)
    # print(result)
    # print(f'{rel_or_index}: {len(result)} Segments')
    rel_os_pages = 0
    pages_mem = 0
    for i, p in enumerate(result):
        # print(f"[Seg {i}] {p[3]}, {p[4]}, {p[4]/p[3] * 100: .2f}%")
        # print(p[3], p[4], p[4]/p[3] * 100)
        rel_os_pages += p[3]
        pages_mem += p[4]
    # print(f"[ ALL ] {rel_os_pages}, {pages_mem}, {pages_mem/rel_os_pages * 100: .2f}%")
    # print(rel_os_pages, pages_mem, pages_mem/rel_os_pages * 100)
    return rel_os_pages, pages_mem

def get_pg_table_size(cur, rel_or_idx):
    sql = f'select pg_table_size(\'{rel_or_idx}\')/8192;' 
    pages_all = execute_sql(cur, sql)[0][0]
    return pages_all

def get_pg_buffer_state(cur):
    sql = 'select * from get_pg_buffer_state();' 
    result = execute_sql(cur, sql)
    
    pg_buffer_state={}
    for i, t in enumerate(result):
        if i == 0:
            print(f"PG Buffer Time: {t[2]}ms")
            continue
        rel_or_idx = t[1]
        pages_mem = t[2]
        
        pages_all = get_pg_table_size(cur, rel_or_idx)
        pg_buffer_state[rel_or_idx] = (pages_all, pages_mem)
        # print(f"{rel_or_idx}: {pages_all}, {pages_mem}, {pages_mem/pages_all * 100: .2f}%")
    return pg_buffer_state

def get_state(cur, p=False):
    relations, indexes = get_rels_and_idx(cur)
    print(f"Read {len(relations)} relations and {len(indexes)} indexes")
    
    pg_buffer_state = get_pg_buffer_state(cur)

    idx_mem = {}
    rel_mem = {}
    for i in indexes:
        res = pg_buffer_state.get(i, None)
        if res is None:
            pages_all = get_pg_table_size(cur, i)
            pages_mem = 0
        else:
            pages_all, pages_mem = res
        if p:
            print(i)
            print(f"[PG]: {pages_all}, {pages_mem}, {pages_mem/pages_all * 100: .2f}%")
        idx_mem[i] = [round(pages_mem/pages_all, 2)]
        rel_os_pages, pages_mem = get_os_cache_state(cur, i)
        if p:
            print(f"[OS]: {rel_os_pages}, {pages_mem}, {pages_mem/rel_os_pages * 100: .2f}%")
        idx_mem[i].append(round(pages_mem/rel_os_pages, 2))

    for r in relations:
        res = pg_buffer_state.get(r, None)
        if res is None:
            pages_all = get_pg_table_size(cur, r)
            pages_mem = 0
        else:
            pages_all, pages_mem = res
        if p:
            print(r)
            print(f"[PG]: {pages_all}, {pages_mem}, {pages_mem/pages_all * 100: .2f}%")
        rel_mem[r] =  [round(pages_mem/pages_all, 2)]
        rel_os_pages, pages_mem = get_os_cache_state(cur, r)
        if p:
            print(f"[OS]: {rel_os_pages}, {pages_mem}, {pages_mem/rel_os_pages * 100: .2f}%")
        rel_mem[r].append(round(pages_mem/rel_os_pages, 2))
    return idx_mem, rel_mem

if __name__ == '__main__':
    # load queries
    args = parser.parse_args()
    workload = args.workload
    port = args.port

    os.system("free -h")

    conn = connect_db(wld2db[workload], port=port)
    cur = conn.cursor()

    start = time.time()
    idx_mem, rel_mem = get_state(cur, p=True)
    end = time.time()

    print(f'{(end-start):.2f} s')
    
    close_db_connection(conn)
    