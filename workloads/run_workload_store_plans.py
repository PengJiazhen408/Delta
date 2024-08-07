import psycopg2
import numpy as np
import os
import argparse
import sys
sys.path.append(os.path.dirname(__file__))
from utils import *
import hashlib
import json
parser = argparse.ArgumentParser(description='run workload')
parser.add_argument('-w', '--workload', type=str, default='JOB', help='workload to run, JOB, STACK, TPCH')
parser.add_argument('-m', '--mode', type=str, default='test', help='train or test queries to run')
parser.add_argument('-s', '--split', type=str, default='rs', help='rs or slow')



wld2path = {
    'JOB-rs-test': 'JOB/job_rs_test',
    'JOB-rs-train': 'JOB/job_rs_train',
    'JOB-slow-test': 'JOB/job_slow_test',
    'JOB-slow-train': 'JOB/job_slow_train',
    'STACK-train': 'STACK/stack_train',
    'STACK-test': 'STACK/stack_test',
    'TPCH-test': 'TPC-H/tpch_test.txt',
    'TPCH-train': 'TPC-H/tpch_train.txt',
    'TPCDS-test': 'TPC-DS/tpcds_test',
    'TPCDS-train': 'TPC-DS/tpcds_train',
}

SCAN_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", 'Bitmap Heap Scan']
JOIN_TYPES = ["Nested Loop", "Hash Join", "Merge Join"]

def get_tree_signature(plan_tree):
    """get_tree_signature.

        Transform plan tree to tree signature(string type)

        Args:
          plan_tree: the plan tree. Type: json str or dict. 
        Output: 
          The tree signature of the plan tree.
    """
    if (isinstance(plan_tree, str)):
        try:
            plan_tree = json.loads(plan_tree)
        except:
            print("The plan is illegal.")
    if "Plan" in plan_tree:
        plan_tree = plan_tree['Plan']
    signature = {}
    if "Plans" in plan_tree:
        children = plan_tree['Plans']
        if len(children) == 1:
            signature['L'] = get_tree_signature(children[0])
            # pass
        else:
            assert len(children) == 2
            signature['L'] = get_tree_signature(children[0])
            signature['R'] = get_tree_signature(children[1])

    node_type = plan_tree['Node Type']
    if node_type in SCAN_TYPES:
        signature["T"] = plan_tree['Relation Name']
    elif node_type in JOIN_TYPES:
        signature["J"] = node_type[0]
    return signature

def hash_md5_str(s):
    """hash_md5_str.
        Encode string to md5 code

        Args:
          s: the string to encode
        Output: 
          The md5 code of the input string.
    """
    md5 = hashlib.md5()
    md5.update(s.encode('utf-8'))
    return md5.hexdigest()

def save_history(plan, query_name, history_dir, buffer_states=None):
    """save_history.
        Save the plan. 
        Args:
          plan: the plan to save.
          query_name: the corresponding query name for the plan. 
          history_dir: the directory for storing historical plans.
          is_test: true if in test phase else false.
          buffer_states(optional): the execution buffer states of query to save.

        IF in training phase:
            The location where the file is kept is: history_dir/iter_value/queryName_planSignatrueHash/plan
        IF in test phase:
            The location where the file is kept is: history_dir/queryName_planSignatrueHash/plan
    """
    plan_signature = str(get_tree_signature(plan))
    # print(plan_signature)
    plan_hash = hash_md5_str(plan_signature)

    if not os.path.exists(history_dir):
        os.makedirs(history_dir)
    history_plan_path = os.path.join(history_dir, query_name+'_'+plan_hash)
    if os.path.exists(history_plan_path):
        print(f"the plan has been saved: {history_plan_path}")
        return
    else:
        os.makedirs(history_plan_path)
    with open(os.path.join(history_plan_path, "plan_signature"), "w") as f:
        f.write(plan_signature)
    with open(os.path.join(history_plan_path, "plan"), "w") as f:
        f.write(str(plan))
    if buffer_states is not None:
        with open(os.path.join(history_plan_path, "buffer"), "w") as f:
            f.write(str(buffer_states))
    print(f"save history: {history_plan_path}")

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

def explain_sql(cur, q):
    sql = 'explain (analyze, format json) ' + q.rstrip('\n')
    result = execute_sql(cur, sql)
    plan_time = result[0][0][0]['Planning Time']
    exec_time = result[0][0][0]['Execution Time']
    return plan_time, exec_time, result[0][0][0]

def print_result(planning_times, exec_times):
    print("plan_time(avg): {} ms".format(np.mean(np.array(planning_times))))
    print("exec_time(avg): {} s".format(np.mean(np.array(exec_times))/1000))
    times = np.sum(np.array(exec_times))
    print("exec_time(all): {} s".format(times/1000))
    print("exec_time(all): {} min".format(times/1000/60))


def load_sql(queries_path):
    queries = []
    query_names = []
    for path, dir_list, file_list in os.walk(queries_path): 
        for f in file_list:
            query_name = f.split('.')[0]
            query_names.append(query_name)
            query_path = os.path.join(path, f)
            with open(query_path, 'r') as f:
                query = f.read().strip()
                queries.append(query)
    print("Read", len(queries),  "queries from", queries_path)
    return query_names, queries

def load_sql2(queries_path):
    queries = []
    query_names = []
    with open(queries_path, 'r') as f:
        for line in f.readlines():
            arr = line.strip().split("#####")
            query_names.append(arr[0])
            queries.append(arr[1])
    print("Read", len(queries),  "queries from", queries_path)
    return query_names, queries

if __name__ == '__main__':
    # load queries
    args = parser.parse_args()
    workload = args.workload
    mode = args.mode
    split = args.split
    if workload == 'JOB':
        workload_mode = '-'.join((workload, split, mode))
    else:
        workload_mode = '-'.join((workload, mode))
    queries_path = wld2path[workload_mode]
    if workload == 'TPCH':
        query_names, queries = load_sql2(queries_path)
    else:
        query_names, queries = load_sql(queries_path)

    
    conn = connect_db(wld2db[workload])
    cur = conn.cursor()

    planning_times = []
    exec_times = []
    baseline_plan = {}
    baseline_exec = {}
    baseline_e2e = {}
    plan_dir = os.path.join("Plans", workload)
    if workload == 'JOB':
        plan_dir = os.path.join("Plans", workload, split)
    os.makedirs(plan_dir, exist_ok=True)
    for qname, q in zip(query_names, queries):
        bufferstate = get_state(cur)
        plan_time, exec_time, plan = explain_sql(cur, q)
        print("Excuting sql: ", qname, f", plan time: {plan_time:.2f} ms, \
              exec time: {exec_time/1000:.2f} s, \
                end-to-end time: {(plan_time+exec_time)/1000:.2f} s")
        baseline_plan[qname] = plan_time
        baseline_exec[qname] = exec_time
        baseline_e2e[qname] = plan_time + exec_time
        planning_times.append(plan_time)
        exec_times.append(exec_time)
        save_history(plan, qname, plan_dir, bufferstate)
    print(baseline_plan)
    print(baseline_exec)
    print(baseline_e2e)
    print_result(planning_times, exec_times)
    close_db_connection(conn)
    

    