mode = 'stack' # job_rs, job_slow, tpcds, stack

if mode == 'job_rs':
    FILE_ID = 'jobrs'
elif mode == 'job_slow':
    FILE_ID = 'jobslow'
elif mode == 'tpcds':
    FILE_ID = 'tpcds'
elif mode == 'stack':
    FILE_ID = 'stack'
# FILE_ID = '1' # job rs
# FILE_ID = '2' # job slow
# FILE_ID = '1000' # job rs test

# train_history_path = "train_history_job_slow"
# test_history_path = "test_history_job_slow"

train_history_path = "train_history_" + mode
test_history_path = "test_history_" + mode

# # # JOB RS
# pg_plan = {'15d': 34.257, '2a': 3.392, '8a': 5.14, '12b': 16.311, '15b': 19.743, '15a': 18.997, '4c': 2.975, '26b': 100.903, '17e': 5.044, '20a': 21.75, '24b': 142.789, '22b': 51.424, '4b': 1.59, '16a': 9.67, '22c': 51.741, '10a': 3.844, '10b': 9.965, '17c': 9.097, '30c': 135.519}
# # # latency
# pg_exec = {'15d': 733.713, '2a': 1015.485, '8a': 942.421, '12b': 142.699, '15b': 36.847, '15a': 381.905, '4c': 162.877, '26b': 163.89, '17e': 18425.863, '20a': 2878.609, '24b': 647.899, '22b': 312.216, '4b': 110.244, '16a': 259.463, '22c': 3551.677, '10a': 388.766, '10b': 288.986, '17c': 7390.888, '30c': 3807.331}
# # # end to end time
# pg_e2e = {'15d': 767.9699999999999, '2a': 1018.8770000000001, '8a': 947.561, '12b': 159.01000000000002, '15b': 56.59, '15a': 400.902, '4c': 165.852, '26b': 264.793, '17e': 18430.907000000003, '20a': 2900.359, '24b': 790.688, '22b': 363.64, '4b': 111.834, '16a': 269.13300000000004, '22c': 3603.418, '10a': 392.61, '10b': 298.95099999999996, '17c': 7399.985, '30c': 3942.8500000000004}

# # JOB Slow
# pg_plan = {'18c': 84.594, '25c': 26.819, '19c': 37.589, '9d': 32.504, '26c': 105.001, '19d': 33.729, '6d': 1.669, '16b': 9.641, '17e': 5.051, '20a': 21.73, '17a': 5.059, '30a': 144.067, '10c': 3.984, '17d': 5.2, '17b': 5.072, '6f': 1.671, '17c': 5.044, '8c': 4.766, '17f': 5.329}
# # latency
# pg_exec = {'18c': 5399.613, '25c': 9350.164, '19c': 449.683, '9d': 4882.497, '26c': 8095.416, '19d': 8070.566, '6d': 5742.657, '16b': 33309.496, '17e': 18146.3, '20a': 2926.603, '17a': 18160.169, '30a': 3711.606, '10c': 6911.177, '17d': 8468.091, '17b': 8661.482, '6f': 6492.148, '17c': 8229.624, '8c': 5935.364, '17f': 14664.4}
# # end to end time
# pg_e2e = {'18c': 5484.207, '25c': 9376.983, '19c': 487.272, '9d': 4915.001, '26c': 8200.417, '19d': 8104.295, '6d': 5744.326, '16b': 33319.137, '17e': 18151.351, '20a': 2948.333, '17a': 18165.228000000003, '30a': 3855.6730000000002, '10c': 6915.161, '17d': 8473.291000000001, '17b': 8666.554, '6f': 6493.819, '17c': 8234.668, '8c': 5940.129999999999, '17f': 14669.729}


# # TPC-DS
# pg_plan = {'query98_3': 68.617, 'query82_2': 3.277, 'query18_3': 20.959, 'query98_1': 0.897, 'query27_2': 19.237, 'query27_3': 1.854, 'query52_1': 5.476, 'query98_2': 1.668, 'query52_2': 0.895, 'query82_3': 5.426, 'query27_1': 1.848, 'query18_1': 5.216, 'query18_2': 5.242, 'query82_1': 1.836, 'query52_3': 0.888}
# # latency
# pg_exec = {'query98_3': 3376.013, 'query82_2': 7418.183, 'query18_3': 2524.767, 'query98_1': 3346.911, 'query27_2': 5753.706, 'query27_3': 5737.1, 'query52_1': 3547.815, 'query98_2': 3342.593, 'query52_2': 3484.015, 'query82_3': 447.585, 'query27_1': 5681.117, 'query18_1': 2793.231, 'query18_2': 3539.971, 'query82_1': 473.897, 'query52_3': 3418.466}
# # end to end time
# pg_e2e = {'query98_3': 3444.63, 'query82_2': 7421.46, 'query18_3': 2545.7259999999997, 'query98_1': 3347.808, 'query27_2': 5772.943, 'query27_3': 5738.954000000001, 'query52_1': 3553.291, 'query98_2': 3344.261, 'query52_2': 3484.91, 'query82_3': 453.01099999999997, 'query27_1': 5682.965, 'query18_1': 2798.447, 'query18_2': 3545.213, 'query82_1': 475.733, 'query52_3': 3419.354}

# Stack
pg_plan =  {'q14_2_60e10e94f8356f019e90caff1adae7b3c7e7df82': 215.216, 'q5_2_q5-027': 34.394, 'q3_0_q3-033': 480.959, 'q3_1_q3-092': 445.619, 'q5_0_q5-018': 19.99, 'q1_2_q1-072': 2.34, 'q3_2_q3-066': 449.05, 'q5_4_q5-091': 20.036, 'q5_1_q5-068': 19.603, 'q14_1_242c393daaec760e4c1597c1bfa8c8f21dc8eb78': 71.227, 'q16_2_ab899cf3a1d5aad39faec3a8a48389e86cd0ba9d': 32.596, 'q16_3_e25cd79d91cb036a4b162831ca91ef70de7e3740': 20.157, 'q12_4_75dd50f0d2debbbbc571ab068a7b97e2efeeb3fa': 19.692, 'q16_0_c5dbc1eb440ba3eef8f8c6ad6659d1c20071dfc7': 20.864, 'q3_4_q3-001': 537.505, 'q12_1_1349d223cd64d6506968a91e84881bb7069ef83a': 17.59, 'q3_3_q3-045': 438.64, 'q14_0_63c0776f1727638316b966fe748df7cc585a335b': 70.41, 'q14_3_6fa85fc0fe36ff6f4f7ce7ce340177ffd4f8ace0': 68.16, 'q16_1_469b915d6ca7078b4c66ea6256b8eb3d77305f9f': 20.626, 'q12_2_062bfd2b89537ed5fe4a1a6b488860f4587e54ec': 16.788, 'q16_4_60adfa44cd3f671e1f74b162796cd659cb9630ac': 20.786, 'q14_4_03aae0b5a60091b040d0cb97d8bbd78d203b6a44': 70.078, 'q12_3_f919c1ec2117227e9821f0ad96a153017c381b56': 17.38, 'q1_0_q1-041': 2.346, 'q1_1_q1-033': 2.285, 'q5_3_q5-040': 19.102, 'q1_4_q1-009': 2.304, 'q1_3_q1-047': 2.28, 'q12_0_ae8c54ce8fa00e0497293c791b8ce5c85932eb36': 16.832}
# latency
pg_exec =  {'q14_2_60e10e94f8356f019e90caff1adae7b3c7e7df82': 898.607, 'q5_2_q5-027': 58.125, 'q3_0_q3-033': 749.478, 'q3_1_q3-092': 655.017, 'q5_0_q5-018': 56.335, 'q1_2_q1-072': 91.833, 'q3_2_q3-066': 466.012, 'q5_4_q5-091': 37.193, 'q5_1_q5-068': 53.323, 'q14_1_242c393daaec760e4c1597c1bfa8c8f21dc8eb78': 214.762, 'q16_2_ab899cf3a1d5aad39faec3a8a48389e86cd0ba9d': 2762.318, 'q16_3_e25cd79d91cb036a4b162831ca91ef70de7e3740': 9535.739, 'q12_4_75dd50f0d2debbbbc571ab068a7b97e2efeeb3fa': 600.716, 'q16_0_c5dbc1eb440ba3eef8f8c6ad6659d1c20071dfc7': 726.584, 'q3_4_q3-001': 532.135, 'q12_1_1349d223cd64d6506968a91e84881bb7069ef83a': 85.462, 'q3_3_q3-045': 652.427, 'q14_0_63c0776f1727638316b966fe748df7cc585a335b': 88.636, 'q14_3_6fa85fc0fe36ff6f4f7ce7ce340177ffd4f8ace0': 453.33, 'q16_1_469b915d6ca7078b4c66ea6256b8eb3d77305f9f': 421.675, 'q12_2_062bfd2b89537ed5fe4a1a6b488860f4587e54ec': 6357.103, 'q16_4_60adfa44cd3f671e1f74b162796cd659cb9630ac': 21462.251, 'q14_4_03aae0b5a60091b040d0cb97d8bbd78d203b6a44': 658.994, 'q12_3_f919c1ec2117227e9821f0ad96a153017c381b56': 125.228, 'q1_0_q1-041': 34.156, 'q1_1_q1-033': 32.956, 'q5_3_q5-040': 49.973, 'q1_4_q1-009': 32.53, 'q1_3_q1-047': 33.837, 'q12_0_ae8c54ce8fa00e0497293c791b8ce5c85932eb36': 17957.907}
# end to end time
pg_e2e =  {'q14_2_60e10e94f8356f019e90caff1adae7b3c7e7df82': 1113.8229999999999, 'q5_2_q5-027': 92.519, 'q3_0_q3-033': 1230.437, 'q3_1_q3-092': 1100.636, 'q5_0_q5-018': 76.325, 'q1_2_q1-072': 94.173, 'q3_2_q3-066': 915.062, 'q5_4_q5-091': 57.229, 'q5_1_q5-068': 72.926, 'q14_1_242c393daaec760e4c1597c1bfa8c8f21dc8eb78': 285.98900000000003, 'q16_2_ab899cf3a1d5aad39faec3a8a48389e86cd0ba9d': 2794.914, 'q16_3_e25cd79d91cb036a4b162831ca91ef70de7e3740': 9555.895999999999, 'q12_4_75dd50f0d2debbbbc571ab068a7b97e2efeeb3fa': 620.408, 'q16_0_c5dbc1eb440ba3eef8f8c6ad6659d1c20071dfc7': 747.448, 'q3_4_q3-001': 1069.6399999999999, 'q12_1_1349d223cd64d6506968a91e84881bb7069ef83a': 103.052, 'q3_3_q3-045': 1091.067, 'q14_0_63c0776f1727638316b966fe748df7cc585a335b': 159.046, 'q14_3_6fa85fc0fe36ff6f4f7ce7ce340177ffd4f8ace0': 521.49, 'q16_1_469b915d6ca7078b4c66ea6256b8eb3d77305f9f': 442.301, 'q12_2_062bfd2b89537ed5fe4a1a6b488860f4587e54ec': 6373.891, 'q16_4_60adfa44cd3f671e1f74b162796cd659cb9630ac': 21483.037, 'q14_4_03aae0b5a60091b040d0cb97d8bbd78d203b6a44': 729.072, 'q12_3_f919c1ec2117227e9821f0ad96a153017c381b56': 142.608, 'q1_0_q1-041': 36.501999999999995, 'q1_1_q1-033': 35.241, 'q5_3_q5-040': 69.075, 'q1_4_q1-009': 34.834, 'q1_3_q1-047': 36.117000000000004, 'q12_0_ae8c54ce8fa00e0497293c791b8ce5c85932eb36': 17974.738999999998}
from save_history import save_history

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
import time
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
            tmp = cache_latency(sql)
            if len(tmp) == 3:
                _, _cost, pgPlan = tmp
            elif len(tmp) == 2:
                _, _cost = tmp
            costs.append(_cost)
            if pgPlan is not None:
                save_history(pgPlan, sql.filename[:-4], train_history_path, False, -1)
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

            tmp = cache_latency(plan)
            if len(tmp) == 3:
                _, plan_cost, pgPlan = tmp
            elif len(tmp) == 2:
                 _, plan_cost = tmp
                 pgPlan = None
            if pgPlan is not None:
                save_history(pgPlan, sql.filename[:-4], train_history_path, False, -1)
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
def validate(test_set, iter, train=False, bushy=False):
    model.eval_mode()
    res = []
    rcs = []
    pjz_res = []
    train_data = []
    _timer = timer()
    with torch.no_grad():
        gen = tqdm(test_set)
        for sql in gen:
            with _timer:
                plan, use_generated, use_generated_time = model.beam_plan(sql, bushy=bushy, judge=False)
            timer_value = _timer.time * 1000
            
            tmp = cache_latency(plan)
            if len(tmp) == 3:
                _value, raw_value, _ = tmp
            elif len(tmp) == 2:
                _value, raw_value = tmp
            if _value < raw_value:
                log(f'Warning: value ({_value}) < raw_value ({raw_value})')
            timer_value += _value

            tmp = cache_latency(sql)
            if len(tmp) == 3:
                origin_value, raw_origin_value, _ = tmp
            elif len(tmp) == 2:
                origin_value, raw_origin_value = tmp

            if use_generated:
                value = timer_value
            else:
                value = use_generated_time * 1000 + origin_value
            raw_rc = raw_value / raw_origin_value
            rc = value / origin_value
            use_generated_rc = timer_value / origin_value

            if not train:
                pg_e2e_time, pg_exec_time = pg_e2e[sql.filename[:-4]], pg_exec[sql.filename[:-4]]
            else:
                pg_e2e_time, pg_exec_time = origin_value, raw_origin_value
            e2e_time = _value 
            exec_time = raw_value
            planning_time = e2e_time - exec_time
            relative_latency = exec_time / pg_exec_time
            relative_e2e = e2e_time / pg_e2e_time
            # 'filename', 'planning_time', 'exec_time', 'e2e_time', 'pg_exec_time', 'pg_e2e_time', 'relative_latency','relative_e2e', 
            pjz_res.append((sql.filename[:-4], planning_time, exec_time, e2e_time, pg_exec_time, pg_e2e_time, relative_latency, relative_e2e))
            
            postfix = {
                'rc': raw_rc,
            }
            gen.set_postfix(postfix)
            # if train:
            #     train_data.append((plan, timer_value / origin_value))
            plan_str = plan._hash_str() + str(plan) # plan.hash_str()
                        # 'filename', 'raw_cost', 'cost', 'timer', 'raw_origin', 'origin', 'raw_relative',
                        # 'relative', 'use_generated', 'use_generated_rc', 'plan', 'baseline',
            res.append((sql.filename, raw_value, value, timer_value, raw_origin_value, origin_value, \
                        raw_rc, rc, use_generated, use_generated_rc, plan_str, str(sql.baseline.result_order)))
            rcs.append((use_generated_rc, raw_rc, rc))
    rcs, raw_rcs, gen_rcs = zip(*rcs)
    return rcs, res, gen_rcs, raw_rcs, pjz_res

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

def train(beam_width=1, epochs=400):
    checkpoint_file = f'ckpts/{FILE_ID}.checkpoint.pkl'
    baseline_explorer = explorer.HalfTimeExplorer(0.5, 0.2, 80)
    if os.path.isfile(checkpoint_file):
        dic = torch.load(checkpoint_file, map_location=device)
        model.model_recover(dic['model'])
        start_epoch = dic['epoch'] + 1
        model.schedule(start_epoch)
        baseline_manager = BaselineCache()
        baseline_manager.load_state_dict(dic['baseline'])
        sample_weights = dic['sample_weights']
        train_rcs = dic['train_rcs']
        if 'timeout_limit' in dic:
            database.statement_timeout = dic['timeout_limit']
        baseline_explorer.count = dic['baseline_count']
        global_df = dic['global_df']
        result_df = dic['result_df']
        timer_df = dic['timer_df']
        total_training_time = dic['total_time']
        random_state = dic.get('random_state', None)

        if random_state:
            set_random_state(random_state)
            log('Loaded random state')
    else:
        total_training_time = 0.0

        baseline_pkl = f'{FILE_ID}.baseline.pkl'
        start_epoch = 0
        log('Preprocessing baselines')
        baseline_manager = BaselineCache()
        if os.path.isfile(baseline_pkl):
            with open(baseline_pkl, 'rb') as f:
                baseline_manager.load_state_dict(pickle.load(f))
                if USE_LATENCY:
                    baseline_manager.set_timeout()
        else:
            baseline_manager.init(train_set, verbose=True)
            if args.no_expert_initialization:
                baseline_manager.data.clear()
            with open(baseline_pkl, 'wb') as f:
                pickle.dump(baseline_manager.state_dict(), f)
        sample_weights = None
        train_rcs = None

        global_df = []
        result_df = []
        timer_df = []
        if not args.no_expert_initialization:
            log('Adding baselines to memory')
            for sql in train_set:
                baseline_order = baseline_manager.get(sql)
                if baseline_order is not None:
                    baseline_value = baseline_manager.get_cost(sql)
                    with torch.no_grad():
                        plan = model.init(sql)
                        while not plan.completed:
                            prev_plan = plan.clone()
                            action, join = baseline_order[plan.total_branch_nodes]
                            model.step(plan, action, join=join)
                            model.add_memory(prev_plan, baseline_value, info=(action, 0,), is_baseline=True)
        if beam_width <= 0:
            postfix = dict(
                loss='/',
            )
            gen = tqdm(range(50), total=50, desc='Pretraining', postfix=postfix)
            for i in gen:
                loss, _ = model.train(False)
                postfix.update(
                    loss=loss.item()
                )
                gen.set_postfix(postfix)

        seed(SEED)

    _total_train_timer = timer()
    use_beam = beam_width >= 1

    for epoch in range(start_epoch, epochs):
        
        with _total_train_timer:
            if False and epoch >= 100 and epoch % 25 == 0:
                model.explorer.reset()
                baseline_explorer.reset()

            if epoch < database.config.eval_mode:
                model.train_mode()
            else:
                model.eval_mode()

            if train_rcs is None:
                _train_set = list(map(lambda x: (x, 0), train_set))
            else:
                _train_set = list(zip(train_set, train_rcs))
            if epoch >= 10:
                if database.config.resample_mode == 'replace':
                    if len(_train_set) > database.config.resample_amount:
                        _new_train_set = random.sample(_train_set, k=len(_train_set) - database.config.resample_amount)
                    else:
                        _new_train_set = []
                    _sample_weights = None if sample_weights is None else list(map(lambda x: min(x, 2.0), sample_weights))
                    _new_train_set.extend(random.choices(_train_set, weights=_sample_weights, k=database.config.resample_amount))
                    _train_set = _new_train_set
                else:
                    # augment
                    _sample_weights = None if sample_weights is None else list(map(lambda x: min(x, 2.0), sample_weights))
                    _train_set.extend(random.choices(_train_set, weights=_sample_weights, k=database.config.resample_amount))
            random.shuffle(_train_set)

            postfix = dict(
                loss='/',
            )
            gen = tqdm(_train_set, desc=f'Epoch {epoch:03d}', postfix=postfix)

            _Time_database = 0.0
            _model_timer = timer()
            _database_timer = timer()
            _plan_timer = timer()
            _Time_plan = 0.0
            _Time_train = 0.0
            _Time_base = 0.0
            _Time_search = 0.0
            _search_timer = timer()
            _Time_train_detail = [0.0, 0.0, 0.0, 0.0]

            _save_timer = timer()
            _Time_save = 0.0

            plan_size = 0

            if epoch < 4:
                bushy = False
            else:
                bushy = database.config.bushy

            result_data_id = 0
            with _model_timer:
                for sql, sample_weight in gen:
                    result_data_id += 1

                    _bushy = bushy
                    is_new_baseline = False

                    with _plan_timer:
                        _, baseline_order, is_leftdeep = baseline_manager.get_all(sql)
                        prob_use_baseline = baseline_explorer.prob
                        use_baseline = baseline_order is not None and (is_leftdeep or _bushy) and random.random() < prob_use_baseline
                        if use_baseline:
                            baseline_explorer.step()
                        with torch.no_grad():
                            plan = model.init(sql)

                            search_plans = [[plan, None, (None, None), None, None, False]]
                            prev_plans = []
                            is_exploration = False
                            step = 0
                            records = []
                            while not search_plans[0][0].completed:
                                prev_plans.append(search_plans)

                                if use_baseline:
                                    with _search_timer:
                                        try:
                                            action, join = baseline_order[step]
                                        except:
                                            log(baseline_order)
                                            raise
                                        prev_plan = search_plans[0][0]
                                        # No need to do deep copy, since there's no grad and no inplace operations
                                        plan = prev_plan.clone(deep=False)
                                        model.step(plan, action)
                                        search_plans = [[plan, 0, (action, join), None, prev_plan, False]]
                                        is_exploration = False
                                        if random.random() < (1 - 0.5 ** (1/6)):
                                            use_baseline = False
                                        records.append('b')
                                    _Time_base += _search_timer.time
                                    prob = model.explorer.prob
                                else:
                                    with _search_timer:
                                        if use_beam:
                                            if epoch < 10:
                                                _beam_width = 1 + random.randrange(beam_width)
                                            else:
                                                _beam_width = beam_width
                                        else:
                                            _beam_width = 1
                                            sample_weight = 0
                                        selected, _explore, selected_values, (min_, max_, mean_), prob = model.topk_search(
                                            map(lambda x: (x[0], x[5]), search_plans), _beam_width,
                                            exploration=not args.no_exploration,
                                            detail=True, exp_bias=sample_weight, use_beam=use_beam, bushy=_bushy)
                                        _is_exploration = _explore != 0
                                        is_exploration |= _is_exploration
                                        if _is_exploration:
                                            records.append(f'{_explore}')
                                        else:
                                            records.append('p')

                                        search_plans = []
                                        for _this_index, (((_state_index, _state, _action), join), value) in enumerate(zip(selected, selected_values)):
                                            # No need to do deep copy, since there's no grad and no inplace operations
                                            this_state = _state.clone(deep=False)
                                            model.step(this_state, _action, join)
                                            search_plans.append([this_state, _state_index, (_action, join), None, _state, _this_index >= _beam_width - _explore])
                                    _Time_search += _search_timer.time
                                step += 1

                            res = cost(sql)
                            if type(res) == int or type(res) == float:
                                origin_value = res
                                pgPlan = None
                            elif len(res) == 2:
                                origin_value, pgPlan = res
                            if pgPlan is not None:
                                plan_size += 1
                                save_history(pgPlan, sql.filename[:-4], train_history_path, False, epoch)
                                print(str(epoch) + " " + str(result_data_id) + " "  + sql.filename[:-4] + " " + str(plan_size) + " " + str(pgPlan is None) + " " +  "\n")

                            _relative_costs = []
                            for index, (plan, parent_index, _action, _, prev_plan, is_explore) in enumerate(search_plans):
                                with _database_timer:
                                  
                                    res = cost(plan)
                                    if type(res) == int or type(res) == float:
                                        value = res
                                        pgPlan = None
                                    elif len(res) == 2:
                                        value, pgPlan = res
                                _Time_database += _database_timer.time

                                if pgPlan is not None:
                                    plan_size += 1
                                    save_history(pgPlan, sql.filename[:-4], train_history_path, False, epoch)
                                    print(str(epoch) + " " + str(result_data_id) + " "  + sql.filename[:-4] + " " + str(plan_size) + " " + str(pgPlan is None) + " " +  "\n")

                                i = 0
                                search_plans[index][3] = value
                                relative_cost = value / origin_value

                                _relative_costs.append(relative_cost)
                                actions = [_action]
                                while parent_index is not None:
                                    parent_list = prev_plans[-1 - i]
                                    parent = parent_list[parent_index]
                                    if parent[3] is None:
                                        parent[3] = value
                                    else:
                                        parent[3] = min(parent[3], value)
                                    parent_index = parent[1]
                                    if parent_index is not None:
                                        actions.append(parent[2])
                                    i += 1
                                if relative_cost < 1:
                                    is_new_baseline = True
                                    model.clear_baseline_dict(sql)
                                    baseline_manager.update(sql, reversed(actions), relative_cost)

                            prev_plans.append(search_plans)
                            for step_plans in prev_plans:
                                for _plan, _, (_action, join), value, prev_plan, is_explore in step_plans:
                                    if value is None or join is None:
                                        continue
                                    memory_value = value / origin_value
                                    # The representation will be updated, so there's no need to do deep copy
                                    model.add_memory(prev_plan.clone(deep=False), value / origin_value, memory_value, info=(_action, join, ), is_baseline=is_new_baseline)
                    _Time_plan += _plan_timer.time

                    with _plan_timer:
                        loss, use_gen_loss, _Times = model.train(detail=True)
                        for i in range(len(_Times)):
                            _Time_train_detail[i] += _Times[i]
                        postfix.update(
                            loss=loss.item(),
                            lug=use_gen_loss.item(),
                            rc=sum(_relative_costs) / len(_relative_costs),
                            prob=prob,
                            pbase=baseline_explorer.prob,
                            type=''.join(records),
                            lr=model.optim.param_groups[0]['lr'],
                        )
                        gen.set_postfix(postfix)
                    _Time_train += _plan_timer.time

            _Time_plan -= _Time_database
            _Time_model = _model_timer.time - _Time_database
            log(f'Model time: {_Time_model} = (Plan time) {_Time_plan} + (Train time) {_Time_train}, database time: {_Time_database}')

            base_gmrl = np.exp(np.mean(np.log(list(map(lambda x: x[0], baseline_manager.data.values())))))
            log(f'Baseline: {base_gmrl} / {len(baseline_manager.data)}')
            
            results_dir = 'results/' + mode
            os.makedirs(results_dir, exist_ok=True)
            
            if epoch >= database.config.validate_start \
                and (epoch + 1) % database.config.validate_interval == 0:
                with _model_timer:
                    log('--------------------------------')
                    # train metrics
                    collect_time = _Time_database # latency
                    planning_time = _Time_plan # train
                    train_time = _Time_train # training time 
                    
                    log('Resampling')
                    resample_start_time = time.time()
                    rcs, res, gen_rcs, raw_rcs, pjz_res = validate(train_set, epoch, bushy=bushy, train=True)

                    train_gmrc = math.exp(sum(map(math.log, rcs)) / len(rcs))
                    train_gmrc_ori = math.exp(sum(map(math.log, gen_rcs)) / len(gen_rcs))
                    train_gmrc_raw = math.exp(sum(map(math.log, raw_rcs)) / len(raw_rcs))
                    
                    train_median = np.median(rcs)
                    df = pd.DataFrame({k : v for k, v in zip((
                        'filename', 'raw_cost', 'cost', 'timer', 'raw_origin', 'origin', 'raw_relative',
                        'relative', 'use_generated', 'use_generated_rc', 'plan', 'baseline',
                    ), zip(*res))})
                    _raw_tops = df['raw_cost'].nlargest(max(round(len(df) * 0.125), 1)).index.astype(np.int32).tolist()
                    raw_to_total_portion = [0 for i in range(len(df))]
                    for index in _raw_tops:
                        raw_to_total_portion[index] += 1 / len(_raw_tops)
                    
                    # df.to_csv(results_dir + f'/{FILE_ID}.{epoch:03d}.train.csv', index=False)
                    train_total_accelerate = df['raw_origin'].sum() / df['raw_cost'].sum()
                    log(f'Train set GMRC: {train_gmrc} / {train_gmrc_ori} / {train_gmrc_raw}, accelerate: {train_total_accelerate}, train set median: {train_median}')
                    log('--------------------------------')

                    # global_df.append((epoch, gmrc, gmrc_ori, gmrc_raw, total_accelerate, train_gmrc, train_gmrc_ori, train_gmrc_raw, train_total_accelerate))
                    # _global_df = pd.DataFrame({k : v for k, v in zip(('epoch', 'test_timer', 'test', 'test_raw', 'test_accelerate', 'train_timer', 'train', 'train_raw', 'train_accelerate'), zip(*global_df))})
                    # _global_df.to_csv(results_dir + f'/{FILE_ID}.csv', index=False)

                    train_rcs = list(map(lambda x: min(max(x - train_gmrc_raw, 0), 200.0), raw_rcs))

                    if epoch >= 10:
                        _relative_weights = list(map(lambda x: min(max(x - train_gmrc_raw, 0), 200.0), raw_rcs))
                        _total_weight = sum(_relative_weights)
                        alpha_absolute = 0.4
                        sample_weights = [(1 - alpha_absolute) * (rel / _total_weight) + alpha_absolute * por for rel, por in zip(_relative_weights, raw_to_total_portion)]
                    else:
                        sample_weights = None
                    
                    resample_stop_time = time.time()
                    resample_time = resample_start_time - resample_stop_time
                    result_df.append((epoch, collect_time, planning_time, train_time, resample_time, plan_size))
                    _result_df = pd.DataFrame({k : v for k, v in zip((
                        'epoch', 'collect_time', 'plan_time', 'model_time', 'resample_time', 'plan_size'
                    ), zip(*result_df))})
                    _result_df.to_csv(results_dir + f'/result_{FILE_ID}.csv', index=False)
                    
                _Time_validate = _model_timer.time
                log(f'Validation time: {_Time_validate}')
            else:
                _Time_validate = None

            timer_df.append((base_gmrl, len(baseline_manager.data), _Time_model, _Time_database, _Time_validate, _Time_plan, _Time_train, _Time_search, _Time_base, *_Time_train_detail))
            df = pd.DataFrame({k: v for k, v in zip((
                'base_gmrl', 'base_len', 'time_model', 'time_database', 'time_validate', 'time_plan', 'time_train', 'time_search', 'time_base', 'time_train_batch_update', 'time_train_predict', 'time_train_train', 'time_train_use_generated',
            ), zip(*timer_df))})
            # df.to_csv(f'results/time.{FILE_ID}.csv', index=False)

            model.schedule()

        _Time_train_total = _total_train_timer.time
        total_training_time += _Time_train_total

        random_state = get_random_state()
        torch.save({
            'model': model.model_export(),
            'epoch': epoch,
            'baseline': baseline_manager.state_dict(),
            'sample_weights': sample_weights,
            'train_rcs': train_rcs,
            'baseline_count': baseline_explorer.count,
            'timeout_limit': database.statement_timeout,
            'global_df': global_df,
            'result_df': result_df,
            'timer_df': timer_df,
            'total_time': total_training_time,
            'random_state': random_state,
            'args': args_dict,
        }, checkpoint_file, _use_new_zipfile_serialization=False)


        model_dir = 'ckpts/ckpts_' + mode
        os.makedirs(model_dir, exist_ok=True)

        result_checkpoint_file = model_dir + f'/{FILE_ID}.{epoch}.checkpoint.pkl'
        torch.save({
            'model': model.model_export(),
            'epoch': epoch,
            'baseline': baseline_manager.state_dict(),
            'sample_weights': sample_weights,
            'train_rcs': train_rcs,
            'baseline_count': baseline_explorer.count,
            'timeout_limit': database.statement_timeout,
            'global_df': global_df,
            'result_df': result_df,
            'timer_df': timer_df,
            'total_time': total_training_time,
            'random_state': random_state,
            'args': args_dict,
        }, result_checkpoint_file, _use_new_zipfile_serialization=False)

    with open(f'baseline.pkl', 'wb') as f:
        pickle.dump(baseline_manager.data, f)

    random_state = get_random_state()
    torch.save({
        'model': model.model_export(),
        'baseline': baseline_manager.state_dict(),
        'sample_weights': sample_weights,
        'train_rcs': train_rcs,
        'baseline_count': baseline_explorer.count,
        'timeout_limit': database.statement_timeout,
        'global_df': global_df,
        'result_df': result_df,
        'timer_df': timer_df,
        'total_time': total_training_time,
        'random_state': random_state,
        'args': args_dict,
    }, f'pretrained/step{FILE_ID}.pkl', _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-d', '--dataset', nargs=2, type=str, default=['dataset/train', 'dataset/test'],
                        help='Training and testing dataset.')
    parser.add_argument('-e', '--epochs', type=int, default=200,
                        help='Total epochs.')
    parser.add_argument('-F', '--id', type=str, default=FILE_ID,
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
    parser.add_argument('-D', '--database', type=str, default='stack',
                        help='PostgreSQL database.')
    parser.add_argument('-U', '--user', type=str, default='postgres',
                        help='PostgreSQL user.')
    parser.add_argument('-P', '--password', type=str, default=None,
                        help='PostgreSQL user password.')
    parser.add_argument('--port', type=int, default=5432,
                        help='PostgreSQL port.')

    args = parser.parse_args()

    args_dict = vars(args)

    FILE_ID = args.id

    log = Logger(f'log/{FILE_ID}.log', buffering=1, stderr=True)

    #torch.use_deterministic_algorithms(True)
    seed(args.seed)
    SEED = args.seed

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

    dataset_file = f'ckpts/{FILE_ID}.dataset.pkl'
    if os.path.isfile(dataset_file):
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

    train(beam_width=args.beam, epochs=args.epochs)
