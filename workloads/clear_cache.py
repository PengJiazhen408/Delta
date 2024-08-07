import os
import sys
sys.path.append(os.path.dirname(__file__))
from utils import wld2db
import argparse
parser = argparse.ArgumentParser(description='run workload')
parser.add_argument('-w', '--workload', type=str, default='JOB', help='workload to run')
parser.add_argument('-p', '--port', type=int, default='5432', help='workload to run')

if __name__ == '__main__':
    # load queries
    args = parser.parse_args()
    workload = args.workload
    port = args.port
    db = wld2db[workload]
    os.system(f"psql -p {port} {db} -c 'select pg_dropcache();'")
    os.system("free -h")
    print("drop cache")
    os.system("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")
    os.system("free -h")