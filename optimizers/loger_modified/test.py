mode = 'stack'# default Dataset: stack
FILE_ID = 't1' # job rs top10 test
Train_ID = f'ckpts_{mode}/'+str(FILE_ID)

test_history_path = f"test_history_{mode}_top10_{FILE_ID}"

# # JOB RS
# pg_plan = {'15d': 34.257, '2a': 3.392, '8a': 5.14, '12b': 16.311, '15b': 19.743, '15a': 18.997, '4c': 2.975, '26b': 100.903, '17e': 5.044, '20a': 21.75, '24b': 142.789, '22b': 51.424, '4b': 1.59, '16a': 9.67, '22c': 51.741, '10a': 3.844, '10b': 9.965, '17c': 9.097, '30c': 135.519}
# # latency
# pg_exec = {'15d': 733.713, '2a': 1015.485, '8a': 942.421, '12b': 142.699, '15b': 36.847, '15a': 381.905, '4c': 162.877, '26b': 163.89, '17e': 18425.863, '20a': 2878.609, '24b': 647.899, '22b': 312.216, '4b': 110.244, '16a': 259.463, '22c': 3551.677, '10a': 388.766, '10b': 288.986, '17c': 7390.888, '30c': 3807.331}
# # end to end time
# pg_e2e = {'15d': 767.9699999999999, '2a': 1018.8770000000001, '8a': 947.561, '12b': 159.01000000000002, '15b': 56.59, '15a': 400.902, '4c': 165.852, '26b': 264.793, '17e': 18430.907000000003, '20a': 2900.359, '24b': 790.688, '22b': 363.64, '4b': 111.834, '16a': 269.13300000000004, '22c': 3603.418, '10a': 392.61, '10b': 298.95099999999996, '17c': 7399.985, '30c': 3942.8500000000004}

# JOB Slow
pg_plan = {'18c': 84.594, '25c': 26.819, '19c': 37.589, '9d': 32.504, '26c': 105.001, '19d': 33.729, '6d': 1.669, '16b': 9.641, '17e': 5.051, '20a': 21.73, '17a': 5.059, '30a': 144.067, '10c': 3.984, '17d': 5.2, '17b': 5.072, '6f': 1.671, '17c': 5.044, '8c': 4.766, '17f': 5.329}
pg_exec = {'18c': 5399.613, '25c': 9350.164, '19c': 449.683, '9d': 4882.497, '26c': 8095.416, '19d': 8070.566, '6d': 5742.657, '16b': 33309.496, '17e': 18146.3, '20a': 2926.603, '17a': 18160.169, '30a': 3711.606, '10c': 6911.177, '17d': 8468.091, '17b': 8661.482, '6f': 6492.148, '17c': 8229.624, '8c': 5935.364, '17f': 14664.4}
pg_e2e = {'18c': 5484.207, '25c': 9376.983, '19c': 487.272, '9d': 4915.001, '26c': 8200.417, '19d': 8104.295, '6d': 5744.326, '16b': 33319.137, '17e': 18151.351, '20a': 2948.333, '17a': 18165.228000000003, '30a': 3855.6730000000002, '10c': 6915.161, '17d': 8473.291000000001, '17b': 8666.554, '6f': 6493.819, '17c': 8234.668, '8c': 5940.129999999999, '17f': 14669.729}

from save_history import save_history, hash_md5_str

import typing
import os
from lib.randomize import seed, get_random_state, set_random_state

seed(0)

import torch
import random
from collections.abc import Iterable
from tqdm import tqdm
import pandas as pd
import math
import numpy as np
import pickle
from lib.log import Logger
from lib.timer import timer
from lib.cache import HashCache

from core import database, Sql, Plan, load
from model.dqn import DeepQNet
from model import explorer

from core.oracle import oracle_database
USE_ORACLE = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

USE_LATENCY = True
SEED = 0

cache_manager = HashCache()
CACHE_FILE = 'latency_cache.pkl'

def dataset_generate(path, verbose=False):
    sqls = load(database.config, path, device=device, verbose=verbose)
    return sqls

def batched(gen, batch_size=64):
    res = []
    iterable = False
    init = False
    for v in gen:
        if not init:
            init = True
            if isinstance(v, Iterable):
                iterable = True
        res.append(v)
        if len(res) == batch_size:
            if iterable:
                yield list(zip(*res))
            else:
                yield res
            res = []
    if res:
        if iterable:
            yield list(zip(*res))
        else:
            yield res

def _cost(plan, latency=USE_LATENCY, cache=True):
    if isinstance(plan, Plan):
        if USE_ORACLE:
            _sql = plan.oracle()
        else:
            _sql = str(plan)
        origin = str(plan.sql)
    else:
        if USE_ORACLE and isinstance(plan, Sql):
            _sql = plan.oracle()
        else:
            _sql = str(plan)
        origin = None
    if USE_ORACLE:
        return oracle_database.latency(_sql, origin=origin, cache=cache)
    
    if latency:
        return database.latency(_sql, origin=origin, cache=cache)
    else:
        database.cost(_sql, cache=cache)

def cost(plan, latency=USE_LATENCY, cache=True):
    if not cache:
        return _cost(plan, latency=True, cache=False)

    tmp = cache_latency(plan)
    if len(tmp) == 3:
        _, raw_value, plan = tmp
        return raw_value, plan
    elif len(tmp) == 2:
        _, raw_value = tmp
        return raw_value

class BaselineCache:
    def __init__(self, sqls=None):
        self.data = {}
        self.timeout = None
        self.max_time = None
        if sqls:
            self.init(sqls)

    def state_dict(self):
        return {
            'data': self.data,
            'timeout': self.timeout,
            'max_time': self.max_time
        }

    def load_state_dict(self, state_dict):
        res = state_dict.get('data', None)
        if res is not None:
            self.data = res
        res = state_dict.get('timeout', NotImplemented)
        if res is not NotImplemented:
            self.timeout = res
        res = state_dict.get('max_time', NotImplemented)
        if res is not NotImplemented:
            self.max_time = res

    def init(self, sqls, verbose=False):
        if verbose:
            sqls = tqdm(sqls)
        costs = []
        for sql in sqls:
            sql : Sql
            _, _cost = cache_latency(sql)
            costs.append(_cost)
            _baseline = sql.baseline.join_order
            baseline = []
            valid = True
            leftdeep = True
            if not _baseline or len(sql.baseline.aliases) != len(sql.aliases):
                # might include subqueries
                if _baseline:
                    print(sql.baseline.aliases, sql.aliases)
                log(f'Warning: Baseline of SQL {sql.filename} is not valid')
                valid = False
            else:
                for index, (left, right) in enumerate(_baseline):
                    if database.config.bushy:
                        if index > 0:
                            if isinstance(right, int):
                                if isinstance(left, int):
                                    leftdeep = False
                                left, right = right, left
                            elif not isinstance(left, int):
                                leftdeep = False
                        baseline.append(((left, right), 0))
                    else:
                        if index > 0:
                            if isinstance(right, int):
                                if isinstance(left, int):
                                    log(f'Warning: Baseline of SQL {sql.filename} is not left-deep')
                                    valid = False
                                    break
                                left, right = right, left
                            elif not isinstance(left, int):
                                log(f'Warning: Baseline of SQL {sql.filename} is not left-deep')
                                valid = False
                                break
                        baseline.append(((left, right), 0))
            if not valid:
                continue
            plan = Plan(sql)
            for left, right in _baseline:
                plan.join(left, right)
            _, plan_cost = cache_latency(plan)
            value = plan_cost / _cost
            self.data[str(sql)] = (value, tuple(baseline), leftdeep)

        self.max_time = max(costs)
        self.timeout = int(database.config.sql_timeout_limit * self.max_time)
        if USE_LATENCY:
            self.set_timeout()

    def set_timeout(self):
        if self.max_time:
            self.timeout = int(database.config.sql_timeout_limit * self.max_time)
        database.statement_timeout = self.timeout
        if USE_ORACLE:
            oracle_database.statement_timeout = self.timeout
        log(f'Set timeout limit to {database.statement_timeout}')

    def update(self, sql, baseline, value):
        s = str(sql)
        prev = self.data.get(s, None)
        if prev is None or value < prev[0]:
            leftdeep = True
            baseline = tuple(baseline)
            for index, ((left, right), join) in enumerate(baseline):
                if index > 0:
                    if isinstance(right, int):
                        if isinstance(left, int):
                            leftdeep = False
                            break
                    elif not isinstance(left, int):
                        leftdeep = False
                        break
            self.data[s] = (value, baseline, leftdeep)

    def get_all(self, sql):
        res = self.data.get(str(sql), None)
        if res is not None:
            return res
        return (None, None, False)

    def get(self, sql):
        res = self.data.get(str(sql), None)
        if res is not None:
            return res[1]
        return None

    def get_cost(self, sql):
        res = self.data.get(str(sql), None)
        if res is not None:
            return res[0]
        return None

CACHE_INVALID_COUNT = 0
CACHE_BACKUP_INTERVAL = 400
_cache_use_count = 0
def cache_latency(sql : typing.Union[Sql, Plan]):
    if isinstance(sql, Plan):
        hash = f'{sql.sql.filename} {sql._hash_str(hint=True)}'
    elif isinstance(sql, Sql):
        hash = f'{sql.filename}$'
    else:
        hash = str(sql)
    cache = cache_manager.get(hash, default=None)
    if cache is not None:
        res, count = cache
        count += 1
        if CACHE_INVALID_COUNT <= 0 or count < CACHE_INVALID_COUNT:
            cache_manager.update(hash, (res, count))
            return res
    else:
        pass
    if USE_ORACLE:
        if isinstance(sql, Plan):
            key = sql.oracle()
        elif isinstance(sql, Sql):
            key = sql.oracle()
        else:
            key = str(sql)
    else:
        key = str(sql)
    _timer = timer()
    with _timer:

        res = cost(key, cache=False)
        if type(res) == int or type(res) == float:
            raw_value = res
            plan = None
        elif len(res) == 2:
            raw_value, plan = res
    value = _timer.time * 1000
    res = (value, raw_value)
    cache_manager.put(key, (res, 0), hash)

    if CACHE_BACKUP_INTERVAL > 0:
        global _cache_use_count
        _cache_use_count += 1
        if _cache_use_count >= CACHE_BACKUP_INTERVAL:
            _cache_use_count = 0
            dic = cache_manager.dump(copy=False)
            with open(CACHE_FILE, 'wb') as f:
                pickle.dump(dic, f)

    if plan is not None:
        return value, raw_value, plan
    else:
        return res

_validate_cache = {}
def validate(test_set, iter, train=False, bushy=False, resample=False):

    results_iter_path = os.path.join(f'results/{mode}_top10')
    if not os.path.exists(results_iter_path):
        os.makedirs(results_iter_path)
    
    model.eval_mode()
    res = []
    rcs = []
    result_res = []
    _timer = timer()
    _top10_timer = timer()

    with torch.no_grad():
        gen = tqdm(test_set)

        sql_df = []

        for sql in gen:
            beam4_search_time = 0
            beam20_search_time = 0
            top10_actual_found = 0

            with _timer:
                plan, use_generated, use_generated_time = model.beam_plan(sql, bushy=bushy, judge=False)

            timer_value = _timer.time * 1000

            beam4_search_time = timer_value
            

            tmp = cost(str(plan), True, False)
            if type(tmp) == int or type(tmp) == float:
                raw_value = res
                result_plan = None
            elif len(tmp) == 2:
                raw_value, result_plan = tmp
            if result_plan is not None:

                save_history(result_plan, sql.filename[:-4], os.path.join(test_history_path, 'top1_beam4'), True, iter)

                raw_value = result_plan['Execution Time']
                _value = beam4_search_time + result_plan['Planning Time'] + raw_value
            

            origin_value, raw_origin_value = pg_e2e[sql.filename[:-4]], pg_exec[sql.filename[:-4]]

            if use_generated:
                value = timer_value
            else:
                value = use_generated_time * 1000 + origin_value

            # wrl, gmrl raw_cost raw_origin
            raw_rc = raw_value / raw_origin_value
            # wre_x, gmre_x cost origin
            rc = value / origin_value

            timer_value = _value
            use_generated_rc = _value / origin_value
            
            postfix = {
                'rc': raw_rc,
            }
            gen.set_postfix(postfix)

            plan_str = plan._hash_str() + str(plan) # plan.hash_str()
                        # 'filename', 'raw_cost', 'cost', 'timer', 'raw_origin', 'origin', 'raw_relative',
                        # 'relative', 'use_generated', 'use_generated_rc', 'plan', 'baseline',
            res.append((sql.filename, raw_value, value, timer_value, raw_origin_value, origin_value, \
                        raw_rc, rc, use_generated, use_generated_rc, plan_str, str(sql.baseline.result_order)))
            rcs.append((use_generated_rc, raw_rc, rc))


            pg_e2e_time, pg_exec_time = origin_value, raw_origin_value
            top1beam4_e2e_time = _value 
            top1beam4_exec_time = raw_value
            planning_time = top1beam4_e2e_time - top1beam4_exec_time
            top1beam4_relative_latency = top1beam4_exec_time / pg_exec_time
            top1beam4_relative_e2e = top1beam4_e2e_time / pg_e2e_time


            with _top10_timer:
                top10_plans, top10_use_generated, top_10_use_generated_time = model.beam_plan(sql, bushy=bushy, judge=False, find_top10=True)

            top10_timer_value = _top10_timer.time * 1000
            beam20_search_time = top10_timer_value

            top10_plans_size = len(top10_plans)

            top10_execs = []
            top10_e2es = []
            for idx in range(0, top10_plans_size):
                top10_plan = top10_plans[idx]

                tmp = cost(str(top10_plan), True, False)
                if type(tmp) == int or type(tmp) == float:
                    exec_time = res
                    top10_result_plan = None
                elif len(tmp) == 2:
                    exec_time, top10_result_plan = tmp
                if top10_result_plan is not None:

                    top10_execs.append(exec_time)
                    save_history(top10_result_plan, sql.filename[:-4], os.path.join(test_history_path, 'top10_beam20'), True, iter, is_top1=f'top{idx+1}')
                    _value = beam20_search_time + top10_result_plan['Planning Time'] + exec_time
                    top10_e2es.append(_value)

            top1beam20_exec_time = top10_execs[0]
            top1beam20_e2e_time = top10_e2es[0]
            top1beam20_relative_latency = top1beam20_exec_time / pg_exec_time
            # top1beam20_relative_e2e = top1beam20_e2e_time / pg_e2e_time
            best_exec_time = np.min(np.array(top10_execs))
            best_idx = np.argmin(np.array(top10_execs))
            best_e2e_time = beam20_search_time + top10_e2es[best_idx]
            bestbeam20_relative_latency = best_exec_time / pg_exec_time
            bestbeam20_relative_e2e = best_e2e_time / pg_e2e_time

            # 'filename', 'pg_exec_time', 'pg_e2e_time', 'top1_beam4_exec','top1_beam4_e2e', 'top1_beam20_exec','top1_beam20_e2e', 'best_top10_beam20_exec','best_top10_beam20_e2e', 
            result_res.append((sql.filename[:-4], pg_exec_time, pg_e2e_time, top1beam4_exec_time, top1beam4_e2e_time, top1beam20_exec_time, top1beam20_e2e_time, best_exec_time, best_e2e_time))

            sql_name = sql.filename[:-4]
            # beam_search找到的计划会有重复，但是没关系，可以用之前写的process_result.py找到真正不同的计划一共有多少，但是没必要，因为这里计算的是top10的beam_search开销，无论找到的计划是否重复，那都是他的开销
            beam4_search_time = beam4_search_time
            beam20_search_time = beam20_search_time
            top10_actual_found = top10_plans_size
            sql_df.append((sql_name, beam4_search_time, beam20_search_time, pg_exec_time, top1beam4_relative_latency, top1beam20_relative_latency, bestbeam20_relative_latency, best_idx+1))
            _sql_df = pd.DataFrame({k : v for k, v in zip(('sql_name', 'beam4_search_time', 'beam20_search_time', 'pg_exec_time', 'top1beam4_rl', 'top1beam20_rl', 'bestbeam20_rl', 'best_idx'), zip(*sql_df))})
            _sql_df.to_csv(f'{results_iter_path}/{FILE_ID}_{iter}_sql.csv', index=False)

    rcs, raw_rcs, gen_rcs = zip(*rcs)
    return rcs, res, gen_rcs, raw_rcs, result_res

def database_warmup(train_set, k=400):
    if k <= len(train_set):
        data = random.sample(train_set, k=k)
    else:
        data = random.choices(train_set, k=k - len(train_set))
        data.extend(train_set)
        random.shuffle(data)
    gen = tqdm(data, desc='Warm up')
    for sql in gen:
        gen.set_postfix({'file': sql.filename})
        database.latency(str(sql), cache=False)

def getGeometricMean(metric_qo, metric_pg):
    total = 1
    for i, j in zip(metric_qo, metric_pg):
       total *= i / j
    return pow(total, 1 / len(metric_qo))


def getMetrics(latency_pg,latency_pred):
    
    WRL = (sum(latency_pred) / sum(latency_pg))
    GMRL = getGeometricMean(latency_pred, latency_pg)

    return WRL, GMRL

def validate_top10(beam_width=1, epochs=400):
    use_beam = beam_width >= 1
    start_epoch = 0
    latency_res = []
    e2e_res = []

    seed(SEED)

    for epoch in range(start_epoch, epochs):

        checkpoint_file_per_epoch = f'temps/{Train_ID}.{epoch}.checkpoint.pkl'
        assert os.path.exists(checkpoint_file_per_epoch), checkpoint_file_per_epoch + " does not exist, can't load ckpts."
        # load model
        dic = torch.load(checkpoint_file_per_epoch, map_location=device)
        if 'use_gen' in dic:
            del dic['use_gen']

        print(f"Load ckpt from {checkpoint_file_per_epoch}...")

        model.model_recover(dic['model'])

        model.eval_mode()


        if epoch < 4:
            bushy = False
        else:
            bushy = database.config.bushy

        if epoch >= database.config.validate_start \
            and (epoch + 1) % database.config.test_interval == 0:
            
            log('--------------------------------')

            log('Validating')

            rcs, res, gen_rcs, raw_rcs, result_rcs = validate(test_set, epoch, bushy=bushy)


            df = pd.DataFrame({k : v for k, v in zip((
                'sqlname', 'pg_exec_time', 'pg_e2e_time', 'top1_beam4_exec', 'top1_beam4_e2e', 'top1_beam20_exec', 'top1_beam20_e2e',
                'best_top10_beam20_exec', 'best_top10_beam20_e2e'
            ), zip(*result_rcs))})

            top1beam4_wrl, top1beam4_gmrl= getMetrics(df['pg_exec_time'], df['top1_beam4_exec'])
            top1beam4_wre, top1beam4_gmre = getMetrics(df['pg_e2e_time'], df['top1_beam4_e2e'])
            top1beam20_wrl, top1beam20_gmrl= getMetrics(df['pg_exec_time'], df['top1_beam20_exec'])
            top1beam20_wre, top1beam20_gmre = getMetrics(df['pg_e2e_time'], df['top1_beam20_e2e'])
            bestbeam20_wrl, bestbeam20_gmrl= getMetrics(df['pg_exec_time'], df['best_top10_beam20_exec'])
            bestbeam20_wre, bestbeam20_gmre = getMetrics(df['pg_e2e_time'], df['best_top10_beam20_e2e'])
            latency_res.append((epoch, top1beam4_wrl, top1beam20_wrl, bestbeam20_wrl, top1beam4_gmrl, top1beam20_gmrl, bestbeam20_gmrl))
            e2e_res.append((epoch, top1beam4_wre, top1beam20_wre, bestbeam20_wre, top1beam4_gmre, top1beam20_gmre, bestbeam20_gmre))
            latency_df = pd.DataFrame({k : v for k, v in zip(('epoch', 'top1beam4_wrl', 'top1beam20_wrl', 'bestbeam20_wrl', 'top1beam4_gmrl', 'top1beam20_gmrl', 'best1beam20_gmrl'), zip(*latency_res))})
            e2e_df = pd.DataFrame({k : v for k, v in zip(('epoch', 'top1beam4_wre', 'top1beam20_wre', 'bestbeam20_wre', 'top1beam4_gmre', 'top1beam20_gmre', 'best1beam20_gmre'), zip(*e2e_res))})
            latency_df.to_csv(f'results/{mode}_top10/latency_{FILE_ID}.csv', index=False)
            e2e_df.to_csv(f'results/{mode}_top10/e2e_{FILE_ID}.csv', index=False)
                    
        model.schedule()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-d', '--dataset', nargs=2, type=str, default=['dataset/job_rs_train', 'dataset/job_rs_test'],
                        help='Training and testing dataset.')
    parser.add_argument('-e', '--epochs', type=int, default=200,
                        help='Total epochs.')
    parser.add_argument('-F', '--id', type=str, default=100,
                        help='File ID.')
    parser.add_argument('-b', '--beam', type=int, default=4,
                        help='Beam width. A beam width less than 1 indicates simple epsilon-greedy policy.')
    parser.add_argument('-s', '--switches', type=int, default=4,
                        help='Branch amount of join methods.')
    parser.add_argument('-l', '--layers', type=int, default=1,
                        help='Number of graph transformer layer in the model.')
    parser.add_argument('-w', '--weight', type=float, default=0.1,
                        help='The weight of reward weighting.')
    parser.add_argument('-N', '--no-restricted-operator', action='store_true',
                        help='Not to use restricted operators.')
    parser.add_argument('--oracle', type=str, default=None, # database/password@localhost:1521
                        help='To use oracle with given connection settings.')
    parser.add_argument('--cache-name', type=str, default=None,
                        help='Cache file name.')
    parser.add_argument('--bushy', action='store_true',
                        help='To use bushy search space.')
    parser.add_argument('--log-cap', type=float, default=1.0,
                        help='Cap of log transformation.')
    parser.add_argument('--warm-up', type=int, default=None,
                        help='To warm up the database with specific iterations.')
    parser.add_argument('--no-exploration', action='store_true',
                        help='To use the original beam search.')
    parser.add_argument('--no-expert-initialization', action='store_true',
                        help='To discard initializing the replay memory with expert knowledge.')
    parser.add_argument('-p', '--pretrain', type=str, default=None,
                        help='Pretrained checkpoint.')
    parser.add_argument('-S', '--seed', type=int, default=3407,
                        help='Random seed.')
    parser.add_argument('-D', '--database', type=str, default='imdb',
                        help='PostgreSQL database.')
    parser.add_argument('-U', '--user', type=str, default='postgres',
                        help='PostgreSQL user.')
    parser.add_argument('-P', '--password', type=str, default=None,
                        help='PostgreSQL user password.')
    parser.add_argument('--port', type=int, default=5432,
                        help='PostgreSQL port.')
    parser.add_argument('--mode', type=str, default='stack',
                        help='Dataset name')

    args = parser.parse_args()

    args_dict = vars(args)

    # FILE_ID = args.id

    log = Logger(f'log/{FILE_ID}.log', buffering=1, stderr=True)

    #torch.use_deterministic_algorithms(True)
    seed(args.seed)
    SEED = args.seed
    mode = args.mode
    if mode is None:
        mode = 'stack'
    
    if mode == 'job_rs':
        FILE_ID = 'jobrs'
        pg_plan = {'15d': 34.257, '2a': 3.392, '8a': 5.14, '12b': 16.311, '15b': 19.743, '15a': 18.997, '4c': 2.975, '26b': 100.903, '17e': 5.044, '20a': 21.75, '24b': 142.789, '22b': 51.424, '4b': 1.59, '16a': 9.67, '22c': 51.741, '10a': 3.844, '10b': 9.965, '17c': 9.097, '30c': 135.519}
        # # latency
        pg_exec = {'15d': 733.713, '2a': 1015.485, '8a': 942.421, '12b': 142.699, '15b': 36.847, '15a': 381.905, '4c': 162.877, '26b': 163.89, '17e': 18425.863, '20a': 2878.609, '24b': 647.899, '22b': 312.216, '4b': 110.244, '16a': 259.463, '22c': 3551.677, '10a': 388.766, '10b': 288.986, '17c': 7390.888, '30c': 3807.331}
        # # end to end time
        pg_e2e = {'15d': 767.9699999999999, '2a': 1018.8770000000001, '8a': 947.561, '12b': 159.01000000000002, '15b': 56.59, '15a': 400.902, '4c': 165.852, '26b': 264.793, '17e': 18430.907000000003, '20a': 2900.359, '24b': 790.688, '22b': 363.64, '4b': 111.834, '16a': 269.13300000000004, '22c': 3603.418, '10a': 392.61, '10b': 298.95099999999996, '17c': 7399.985, '30c': 3942.8500000000004}
    elif mode == 'job_slow':
        FILE_ID = 'jobslow'
        pg_plan = {'18c': 84.594, '25c': 26.819, '19c': 37.589, '9d': 32.504, '26c': 105.001, '19d': 33.729, '6d': 1.669, '16b': 9.641, '17e': 5.051, '20a': 21.73, '17a': 5.059, '30a': 144.067, '10c': 3.984, '17d': 5.2, '17b': 5.072, '6f': 1.671, '17c': 5.044, '8c': 4.766, '17f': 5.329}
        # latency
        pg_exec = {'18c': 5399.613, '25c': 9350.164, '19c': 449.683, '9d': 4882.497, '26c': 8095.416, '19d': 8070.566, '6d': 5742.657, '16b': 33309.496, '17e': 18146.3, '20a': 2926.603, '17a': 18160.169, '30a': 3711.606, '10c': 6911.177, '17d': 8468.091, '17b': 8661.482, '6f': 6492.148, '17c': 8229.624, '8c': 5935.364, '17f': 14664.4}
        # end to end time
        pg_e2e = {'18c': 5484.207, '25c': 9376.983, '19c': 487.272, '9d': 4915.001, '26c': 8200.417, '19d': 8104.295, '6d': 5744.326, '16b': 33319.137, '17e': 18151.351, '20a': 2948.333, '17a': 18165.228000000003, '30a': 3855.6730000000002, '10c': 6915.161, '17d': 8473.291000000001, '17b': 8666.554, '6f': 6493.819, '17c': 8234.668, '8c': 5940.129999999999, '17f': 14669.729}
    elif mode == 'tpcds':
        FILE_ID = 'tpcds'
        pg_plan = {'query98_3': 68.617, 'query82_2': 3.277, 'query18_3': 20.959, 'query98_1': 0.897, 'query27_2': 19.237, 'query27_3': 1.854, 'query52_1': 5.476, 'query98_2': 1.668, 'query52_2': 0.895, 'query82_3': 5.426, 'query27_1': 1.848, 'query18_1': 5.216, 'query18_2': 5.242, 'query82_1': 1.836, 'query52_3': 0.888}
        # latency
        pg_exec = {'query98_3': 3376.013, 'query82_2': 7418.183, 'query18_3': 2524.767, 'query98_1': 3346.911, 'query27_2': 5753.706, 'query27_3': 5737.1, 'query52_1': 3547.815, 'query98_2': 3342.593, 'query52_2': 3484.015, 'query82_3': 447.585, 'query27_1': 5681.117, 'query18_1': 2793.231, 'query18_2': 3539.971, 'query82_1': 473.897, 'query52_3': 3418.466}
        # end to end time
        pg_e2e = {'query98_3': 3444.63, 'query82_2': 7421.46, 'query18_3': 2545.7259999999997, 'query98_1': 3347.808, 'query27_2': 5772.943, 'query27_3': 5738.954000000001, 'query52_1': 3553.291, 'query98_2': 3344.261, 'query52_2': 3484.91, 'query82_3': 453.01099999999997, 'query27_1': 5682.965, 'query18_1': 2798.447, 'query18_2': 3545.213, 'query82_1': 475.733, 'query52_3': 3419.354}
    elif mode == 'stack':
        FILE_ID = 'stack'
        pg_plan =  {'q14_2_60e10e94f8356f019e90caff1adae7b3c7e7df82': 215.216, 'q5_2_q5-027': 34.394, 'q3_0_q3-033': 480.959, 'q3_1_q3-092': 445.619, 'q5_0_q5-018': 19.99, 'q1_2_q1-072': 2.34, 'q3_2_q3-066': 449.05, 'q5_4_q5-091': 20.036, 'q5_1_q5-068': 19.603, 'q14_1_242c393daaec760e4c1597c1bfa8c8f21dc8eb78': 71.227, 'q16_2_ab899cf3a1d5aad39faec3a8a48389e86cd0ba9d': 32.596, 'q16_3_e25cd79d91cb036a4b162831ca91ef70de7e3740': 20.157, 'q12_4_75dd50f0d2debbbbc571ab068a7b97e2efeeb3fa': 19.692, 'q16_0_c5dbc1eb440ba3eef8f8c6ad6659d1c20071dfc7': 20.864, 'q3_4_q3-001': 537.505, 'q12_1_1349d223cd64d6506968a91e84881bb7069ef83a': 17.59, 'q3_3_q3-045': 438.64, 'q14_0_63c0776f1727638316b966fe748df7cc585a335b': 70.41, 'q14_3_6fa85fc0fe36ff6f4f7ce7ce340177ffd4f8ace0': 68.16, 'q16_1_469b915d6ca7078b4c66ea6256b8eb3d77305f9f': 20.626, 'q12_2_062bfd2b89537ed5fe4a1a6b488860f4587e54ec': 16.788, 'q16_4_60adfa44cd3f671e1f74b162796cd659cb9630ac': 20.786, 'q14_4_03aae0b5a60091b040d0cb97d8bbd78d203b6a44': 70.078, 'q12_3_f919c1ec2117227e9821f0ad96a153017c381b56': 17.38, 'q1_0_q1-041': 2.346, 'q1_1_q1-033': 2.285, 'q5_3_q5-040': 19.102, 'q1_4_q1-009': 2.304, 'q1_3_q1-047': 2.28, 'q12_0_ae8c54ce8fa00e0497293c791b8ce5c85932eb36': 16.832}
        # latency
        pg_exec =  {'q14_2_60e10e94f8356f019e90caff1adae7b3c7e7df82': 898.607, 'q5_2_q5-027': 58.125, 'q3_0_q3-033': 749.478, 'q3_1_q3-092': 655.017, 'q5_0_q5-018': 56.335, 'q1_2_q1-072': 91.833, 'q3_2_q3-066': 466.012, 'q5_4_q5-091': 37.193, 'q5_1_q5-068': 53.323, 'q14_1_242c393daaec760e4c1597c1bfa8c8f21dc8eb78': 214.762, 'q16_2_ab899cf3a1d5aad39faec3a8a48389e86cd0ba9d': 2762.318, 'q16_3_e25cd79d91cb036a4b162831ca91ef70de7e3740': 9535.739, 'q12_4_75dd50f0d2debbbbc571ab068a7b97e2efeeb3fa': 600.716, 'q16_0_c5dbc1eb440ba3eef8f8c6ad6659d1c20071dfc7': 726.584, 'q3_4_q3-001': 532.135, 'q12_1_1349d223cd64d6506968a91e84881bb7069ef83a': 85.462, 'q3_3_q3-045': 652.427, 'q14_0_63c0776f1727638316b966fe748df7cc585a335b': 88.636, 'q14_3_6fa85fc0fe36ff6f4f7ce7ce340177ffd4f8ace0': 453.33, 'q16_1_469b915d6ca7078b4c66ea6256b8eb3d77305f9f': 421.675, 'q12_2_062bfd2b89537ed5fe4a1a6b488860f4587e54ec': 6357.103, 'q16_4_60adfa44cd3f671e1f74b162796cd659cb9630ac': 21462.251, 'q14_4_03aae0b5a60091b040d0cb97d8bbd78d203b6a44': 658.994, 'q12_3_f919c1ec2117227e9821f0ad96a153017c381b56': 125.228, 'q1_0_q1-041': 34.156, 'q1_1_q1-033': 32.956, 'q5_3_q5-040': 49.973, 'q1_4_q1-009': 32.53, 'q1_3_q1-047': 33.837, 'q12_0_ae8c54ce8fa00e0497293c791b8ce5c85932eb36': 17957.907}
        # end to end time
        pg_e2e =  {'q14_2_60e10e94f8356f019e90caff1adae7b3c7e7df82': 1113.8229999999999, 'q5_2_q5-027': 92.519, 'q3_0_q3-033': 1230.437, 'q3_1_q3-092': 1100.636, 'q5_0_q5-018': 76.325, 'q1_2_q1-072': 94.173, 'q3_2_q3-066': 915.062, 'q5_4_q5-091': 57.229, 'q5_1_q5-068': 72.926, 'q14_1_242c393daaec760e4c1597c1bfa8c8f21dc8eb78': 285.98900000000003, 'q16_2_ab899cf3a1d5aad39faec3a8a48389e86cd0ba9d': 2794.914, 'q16_3_e25cd79d91cb036a4b162831ca91ef70de7e3740': 9555.895999999999, 'q12_4_75dd50f0d2debbbbc571ab068a7b97e2efeeb3fa': 620.408, 'q16_0_c5dbc1eb440ba3eef8f8c6ad6659d1c20071dfc7': 747.448, 'q3_4_q3-001': 1069.6399999999999, 'q12_1_1349d223cd64d6506968a91e84881bb7069ef83a': 103.052, 'q3_3_q3-045': 1091.067, 'q14_0_63c0776f1727638316b966fe748df7cc585a335b': 159.046, 'q14_3_6fa85fc0fe36ff6f4f7ce7ce340177ffd4f8ace0': 521.49, 'q16_1_469b915d6ca7078b4c66ea6256b8eb3d77305f9f': 442.301, 'q12_2_062bfd2b89537ed5fe4a1a6b488860f4587e54ec': 6373.891, 'q16_4_60adfa44cd3f671e1f74b162796cd659cb9630ac': 21483.037, 'q14_4_03aae0b5a60091b040d0cb97d8bbd78d203b6a44': 729.072, 'q12_3_f919c1ec2117227e9821f0ad96a153017c381b56': 142.608, 'q1_0_q1-041': 36.501999999999995, 'q1_1_q1-033': 35.241, 'q5_3_q5-040': 69.075, 'q1_4_q1-009': 34.834, 'q1_3_q1-047': 36.117000000000004, 'q12_0_ae8c54ce8fa00e0497293c791b8ce5c85932eb36': 17974.738999999998}

    Train_ID = f'ckpts_{mode}/'+str(FILE_ID)

    test_history_path = f"test_history_{mode}_top10_{FILE_ID}"
    
    cache_name = args.cache_name
    if cache_name is None:
        cache_name = FILE_ID

    CACHE_FILE = f'{args.database}.{cache_name}.pkl'
    if os.path.isfile(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            dic = pickle.load(f)
            cache_manager.load(dic)

    if args.oracle is not None:
        USE_ORACLE = True
        oracle_database.setup(args.oracle, dbname=args.database, cache=False)
    try:
        database.setup(dbname=args.database, cache=False)
    except:
        try:
            database_args = {'dbname': args.database}
            if args.user is not None:
                database_args['user'] = args.user
            if args.password is not None:
                database_args['password'] = args.password
            if args.port is not None:
                database_args['port'] = args.port
            database.setup(**database_args, cache=False)
        except:
            database.assistant_setup(dbname=args.database, cache=False)

    database.config.bushy = args.bushy

    dataset_file = f'temps/{FILE_ID}.dataset.pkl'
    if os.path.isfile(dataset_file):
        print("load ", dataset_file)
        dataset = torch.load(dataset_file, map_location=device)
        train_set, test_set = dataset
        for _set in (train_set, test_set):
            for sql in _set:
                sql.to(device)
    else:
        train_path, test_path = args.dataset

        log('Generating train set')
        train_set = dataset_generate(train_path, verbose=True)
        log('Generating test set')
        test_set = dataset_generate(test_path, verbose=True)

        torch.save([train_set, test_set], dataset_file, _use_new_zipfile_serialization=False)

    if args.warm_up is not None:
        database_warmup(train_set, k=args.warm_up)
        seed(args.seed)
        SEED = args.seed

    restricted_operator = not args.no_restricted_operator
    reward_weighting = args.weight

    model = DeepQNet(device=device, half=200, out_dim=args.switches, num_table_layers=args.layers,
                     use_value_predict=False, restricted_operator=restricted_operator,
                     reward_weighting=reward_weighting, log_cap=args.log_cap)

    pretrain_file = args.pretrain
    if pretrain_file is not None and os.path.isfile(pretrain_file):
        dic = torch.load(pretrain_file, map_location=device)
        if 'use_gen' in dic:
            del dic['use_gen']
        model.model_recover(dic)

    database.config.beam_width = args.beam

    validate_top10(beam_width=args.beam, epochs=args.epochs)
