import psycopg2
import numpy as np
import os
import argparse
import sys
sys.path.append(os.path.dirname(__file__))
from utils import *
parser = argparse.ArgumentParser(description='run workload')
parser.add_argument('-w', '--workload', type=str, default='JOB', help='workload to run, JOB, STACK, TPCH')
parser.add_argument('-m', '--mode', type=str, default='test', help='train or test queries to run')
parser.add_argument('-s', '--split', type=str, default='rs', help='rs or slow')
parser.add_argument('-p', '--port', type=int, default='5432', help='workload to run')

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

def explain_sql(cur, q):
    sql = 'explain (analyze, format json) ' + q.rstrip('\n')
    result = execute_sql(cur, sql)
    plan_time = result[0][0][0]['Planning Time']
    exec_time = result[0][0][0]['Execution Time']
    return plan_time, exec_time

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
    port = args.port
    if workload == 'JOB':
        workload_mode = '-'.join((workload, split, mode))
    else:
        workload_mode = '-'.join((workload, mode))
    queries_path = wld2path[workload_mode]
    if workload == 'TPCH':
        query_names, queries = load_sql2(queries_path)
    else:
        query_names, queries = load_sql(queries_path)

    print(len(query_names))
    conn = connect_db(wld2db[workload], port=port)
    cur = conn.cursor()

    planning_times = []
    exec_times = []
    baseline_plan = {}
    baseline_exec = {}
    baseline_e2e = {}
    for qname, q in zip(query_names, queries):
        plan_time, exec_time = explain_sql(cur, q)
        print("Excuting sql: ", qname, f", plan time: {plan_time:.2f} ms, \
              exec time: {exec_time/1000:.2f} s, \
                end-to-end time: {(plan_time+exec_time)/1000:.2f} s")
        baseline_plan[qname] = plan_time
        baseline_exec[qname] = exec_time
        baseline_e2e[qname] = plan_time + exec_time
        planning_times.append(plan_time)
        exec_times.append(exec_time)
    print(baseline_plan)
    print(baseline_exec)
    print(baseline_e2e)
    print_result(planning_times, exec_times)
    close_db_connection(conn)
    

    