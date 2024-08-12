import collections
import copy
import logging
import os
import pickle
import pprint
import signal
import time
import glob

from absl import app
from absl import flags
import numpy as np
import pandas as pd
import psycopg2
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import ray
import ray.util
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import wandb

import balsa
from balsa import costing
from balsa import envs
from balsa import execution
from balsa import plan_analysis
from balsa.experience import Experience
from balsa.models.transformer import ReportModel
import balsa.optimizer as optim
from balsa.util import dataset as ds
from balsa.util import plans_lib
from balsa.util import postgres

import pg_executor
from pg_executor import dbmsx_executor
import run

FLAGS = flags.FLAGS
flags.DEFINE_string('run_test', 'JOBSlowSplit_PostgresSim', 'Experiment config to run.')
flags.DEFINE_string('exp_id', 'd8awdme0', 'Experiment config to run.')

flags.DEFINE_integer('topK', 10, 'top-k plans')
flags.DEFINE_integer('start_iter', 0, 'start_iter')
flags.DEFINE_integer('end_iter', 500, 'end_iter, [s, e)')
topKList=[1,3,5,10]

# JOB rs
# pg_exec = {'15d': 733.713, '2a': 1015.485, '8a': 942.421, '12b': 142.699, '15b': 36.847, '15a': 381.905, '4c': 162.877, '26b': 163.89, '17e': 18425.863, '20a': 2878.609, '24b': 647.899, '22b': 312.216, '4b': 110.244, '16a': 259.463, '22c': 3551.677, '10a': 388.766, '10b': 288.986, '17c': 7390.888, '30c': 3807.331}
# pg_e2e = {'15d': 767.9699999999999, '2a': 1018.8770000000001, '8a': 947.561, '12b': 159.01000000000002, '15b': 56.59, '15a': 400.902, '4c': 165.852, '26b': 264.793, '17e': 18430.907000000003, '20a': 2900.359, '24b': 790.688, '22b': 363.64, '4b': 111.834, '16a': 269.13300000000004, '22c': 3603.418, '10a': 392.61, '10b': 298.95099999999996, '17c': 7399.985, '30c': 3942.8500000000004}
        
# JOB slow
pg_exec = {'18c': 5399.613, '25c': 9350.164, '19c': 449.683, '9d': 4882.497, '26c': 8095.416, '19d': 8070.566, '6d': 5742.657, '16b': 33309.496, '17e': 18146.3, '20a': 2926.603, '17a': 18160.169, '30a': 3711.606, '10c': 6911.177, '17d': 8468.091, '17b': 8661.482, '6f': 6492.148, '17c': 8229.624, '8c': 5935.364, '17f': 14664.4}
pg_e2e = {'18c': 5484.207, '25c': 9376.983, '19c': 487.272, '9d': 4915.001, '26c': 8200.417, '19d': 8104.295, '6d': 5744.326, '16b': 33319.137, '17e': 18151.351, '20a': 2948.333, '17a': 18165.228000000003, '30a': 3855.6730000000002, '10c': 6915.161, '17d': 8473.291000000001, '17b': 8666.554, '6f': 6493.819, '17c': 8234.668, '8c': 5940.129999999999, '17f': 14669.729}

# TPC-DS
# pg_exec = {'query98_3': 3376.013, 'query82_2': 7418.183, 'query18_3': 2524.767, 'query98_1': 3346.911, 'query27_2': 5753.706, 'query27_3': 5737.1, 'query52_1': 3547.815, 'query98_2': 3342.593, 'query52_2': 3484.015, 'query82_3': 447.585, 'query27_1': 5681.117, 'query18_1': 2793.231, 'query18_2': 3539.971, 'query82_1': 473.897, 'query52_3': 3418.466}
# pg_e2e = {'query98_3': 3444.63, 'query82_2': 7421.46, 'query18_3': 2545.7259999999997, 'query98_1': 3347.808, 'query27_2': 5772.943, 'query27_3': 5738.954000000001, 'query52_1': 3553.291, 'query98_2': 3344.261, 'query52_2': 3484.91, 'query82_3': 453.01099999999997, 'query27_1': 5682.965, 'query18_1': 2798.447, 'query18_2': 3545.213, 'query82_1': 475.733, 'query52_3': 3419.354}
        

SCAN_TYPES = ["Seq Scan", "Index Scan", "Index Only Scan", 'Bitmap Heap Scan']
JOIN_TYPES = ["Nested Loop", "Hash Join", "Merge Join"]

def get_history(query_name, plan_hash, plan_signature, history_dir):
    history_path = os.path.join(history_dir, query_name, plan_hash)
    # print(plan_hash)
    if not os.path.exists(history_path):
        return None
    print(f"visit histroy path: {history_path}")
    with open(os.path.join(history_path, "plan_signature"), "r") as f:
        history_plan_str = f.read().strip()
        if plan_signature != history_plan_str:
            print(f"there is a hash conflict between two plans: {history_path}")
            print(f"given: { plan_signature}")
            print(f"wanted: {history_plan_str}")
            return None
    print(f"get the history file: {history_path}")
    with open(os.path.join(history_path, "plan"), "r") as f:
        plan_json = f.read().strip()
        plan_json = eval(plan_json)
        return plan_json

# @ray.remote
def ExecuteSql(query_name,
               sql_str,
               hint_str,
               hinted_plan,
               query_node,
               predicted_latency,
               curr_timeout_ms=None,
               found_plans=None,
               predicted_costs=None,
               silent=False,
               is_test=False,
               use_local_execution=True,
               plan_physical=True,
               repeat=1,
               engine='postgres',
               iter_num=None):
    """Executes a query.

    Returns:
      If use_local_execution:
        A (pg_executor, dbmsx_executor).Result.
      Else:
        A ray.ObjectRef of the above.
    """
    # Unused args.
    del query_node, predicted_latency, found_plans,\
        predicted_costs, silent, is_test, plan_physical

    assert engine in ('postgres', 'dbmsx'), engine
    if engine == 'postgres':
        _, json_dict = postgres.SqlToPlanNode(sql_str,comment=hint_str,verbose=False)
        plan_signature = str(run.get_tree_signature(json_dict))
        plan_hash = run.hash_md5_str(plan_signature)
        plan_json = get_history(query_name, plan_hash, plan_signature, "./test_history") 
        if plan_json is None:
            return postgres.ExplainAnalyzeSql(sql_str,
                                          comment=hint_str,
                                          verbose=False,
                                          geqo_off=True,
                                          timeout_ms=curr_timeout_ms,
                                          remote=not use_local_execution)
        else:
            result = [[[plan_json]]]
            has_timeout = False
            ip = "127.0.0.1"
            return pg_executor.Result(result, has_timeout, ip)
    else:
        return DbmsxExecuteSql(sql_str,
                               comment=hint_str,
                               timeout_ms=curr_timeout_ms,
                               remote=not use_local_execution,
                               repeat=repeat)

class BalsaEvaAgent(run.BalsaAgent):
    def __init__(self, params, run_name, exp_id):
        self.run_name = run_name
        self.exp_id = exp_id
        super().__init__(params)
    
    def _InitLogging(self):
        p = self.params
        # experiment = str(p.cls).split('.')[-1][:-2]
        self.loggers = [
            pl_loggers.TensorBoardLogger(save_dir=os.getcwd(),
                                         version=None,
                                         name='tensorboard_logs'),
            pl_loggers.WandbLogger(save_dir=os.getcwd(), project='balsa_test', name=f"Test-Actual-{self.run_name}-{self.exp_id}"),
        ]
        self.summary_writer = SummaryWriter()
        self.wandb_logger = self.loggers[-1]
        p_dict = balsa.utils.SanitizeToText(dict(p))
        for logger in self.loggers:
            logger.log_hyperparams(p_dict)
        with open(os.path.join(self.wandb_logger.experiment.dir, 'params.txt'),
                  'w') as f:
            # Files saved to wandb's rundir are auto-uploaded.
            f.write(p.ToText())
        if not p.run_baseline:
            self.LogExpertExperience(self.train_nodes, self.test_nodes)
        
    def LogTestExperience(self, all_latency, balsa_planning_times, with_pg=False):
        assert len(self.test_nodes) == len(all_latency)
        iter_total_latency = {}
        iter_end_to_end_time = {}
        relative_lentcy = {}
        relative_end_to_end_time = {}
        for k in topKList:
            iter_total_latency[k] = 0
            relative_lentcy[k] = []
            iter_end_to_end_time[k] = 0
            relative_end_to_end_time[k] = []
        workload_total_latency = 0
        workload_end_to_end_time = 0
        rows = []
        rows_e2e = []
        data = []
        has_timeouts = False
        for node in self.test_nodes:
            query_name = node.info['query_name']
            # pg_end_to_end_time = node.cost + node.info['pg_planning_time']
            pg_end_to_end_time = pg_e2e[query_name.split(".")[0]]
            pg_latency = pg_exec[query_name.split(".")[0]]
            latencies = all_latency[query_name]
            balsa_planning_time = balsa_planning_times[query_name]
            min_pos = np.argmin(latencies)+1
            if with_pg and pg_latency < np.min(latencies):
                min_pos = 0
            topK_latency = [np.min(latencies[:k]) for k in topKList]
            for k, latency in zip(topKList, topK_latency):
                if with_pg:
                    latency = min(latency, pg_latency)
                iter_total_latency[k] += latency
                balsa_end_to_end_time = latency+balsa_planning_time
                iter_end_to_end_time[k] += balsa_end_to_end_time
                relative_lentcy[k].append(latency/pg_latency)
                relative_end_to_end_time[k].append(balsa_end_to_end_time/pg_end_to_end_time)
                if with_pg:
                    data.append(('top_{}_latency_with_pg/q{}'.format(k, query_name),
                             latency / 1e3, self.curr_value_iter))
                    data.append(('top_{}_relative_latency_with_pg/q{}'.format(k, query_name),
                             latency / pg_latency, self.curr_value_iter))
                    data.append(('top_{}_e2e_with_pg/q{}'.format(k, query_name),
                             balsa_end_to_end_time / 1e3, self.curr_value_iter))
                    data.append(('top_{}_relative_e2e_with_pg/q{}'.format(k, query_name),
                             balsa_end_to_end_time / pg_end_to_end_time, self.curr_value_iter))
                else:
                    data.append(('top_{}_latency/q{}'.format(k, query_name),
                             latency / 1e3, self.curr_value_iter))
                    data.append(('top_{}_relative_latency/q{}'.format(k, query_name),
                             latency / pg_latency, self.curr_value_iter))
                    data.append(('top_{}_e2e/q{}'.format(k, query_name),
                             balsa_end_to_end_time / 1e3, self.curr_value_iter))
                    data.append(('top_{}_relative_e2e/q{}'.format(k, query_name),
                             balsa_end_to_end_time / pg_end_to_end_time, self.curr_value_iter))
            if with_pg:
                data.append(('min_pos_with_pg/q{}'.format(query_name),
                         min_pos, self.curr_value_iter))
            else:
                data.append(('min_pos/q{}'.format(query_name),
                         min_pos, self.curr_value_iter))
            data.append(('num_total_timeouts', self.num_total_timeouts, self.curr_value_iter))
            workload_total_latency += pg_latency
            workload_end_to_end_time += pg_end_to_end_time
            row = [self.curr_value_iter, query_name]
            row.extend([l/pg_latency for l in topK_latency])
            row.append(min_pos)
            row = tuple(row)
            rows.append(row)
            row = [self.curr_value_iter, query_name]
            row.extend([(l+balsa_planning_time)/pg_end_to_end_time for l in topK_latency])
            row.append(min_pos)
            row = tuple(row)
            rows_e2e.append(row)
        
        # Log a table of latencies, sorted by descending latency.
        rows = list(sorted(rows, key=lambda r: r[1], reverse=True))
        rows_e2e = list(sorted(rows_e2e, key=lambda r: r[1], reverse=True))
        columns = ['curr_value_iter', 'query_name']
        columns.extend([f'top_{k}_relative_latency' for k in topKList])
        columns.append('min_pos')
        table = wandb.Table(columns=columns, rows=rows)
        if with_pg:
            self.wandb_logger.experiment.log({'topK_relative_lentency_with_pg_table': table})
        else:
            self.wandb_logger.experiment.log({'topK_relative_lentency_table': table})
        columns = ['curr_value_iter', 'query_name']
        columns.extend([f'top_{k}_relative_e2e_time' for k in topKList])
        columns.append('min_pos')
        table = wandb.Table(columns=columns, rows=rows_e2e)
        if with_pg:
            self.wandb_logger.experiment.log({'topK_relative_e2e_time_with_pg_table': table})
        else:
            self.wandb_logger.experiment.log({'topK_relative_e2e_time_table': table})
        for k, v in iter_total_latency.items():
            gmrl = np.exp(np.mean(np.log(relative_lentcy[k])))
            if with_pg:
                data.extend([
                    (f'top_{k}_latency_with_pg/workload', v / 1e3, self.curr_value_iter),
                    (f'top_{k}_relative_latency_with_pg/workload', v / workload_total_latency, self.curr_value_iter),
                    (f'top_{k}_relative_latency_with_pg/gmrl', gmrl, self.curr_value_iter),
                    ('curr_value_iter', self.curr_value_iter, self.curr_value_iter),
                ])
            else:
                data.extend([
                    (f'top_{k}_latency/workload', v / 1e3, self.curr_value_iter),
                    (f'top_{k}_relative_latency/workload', v / workload_total_latency, self.curr_value_iter),
                    (f'top_{k}_relative_latency/gmrl', gmrl, self.curr_value_iter),
                    ('curr_value_iter', self.curr_value_iter, self.curr_value_iter),
                ])
        for k, v in iter_end_to_end_time.items():
            gmre = np.exp(np.mean(np.log(relative_end_to_end_time[k])))
            if with_pg:
                data.extend([
                    (f'top_{k}_e2e_with_pg/workload', v / 1e3, self.curr_value_iter),
                    (f'top_{k}_relative_e2e_with_pg/workload', v / workload_end_to_end_time, self.curr_value_iter),
                    (f'top_{k}_relative_e2e_with_pg/gmre', gmre, self.curr_value_iter),
                    ('curr_value_iter', self.curr_value_iter, self.curr_value_iter),
                ])
            else:
                data.extend([
                    (f'top_{k}_e2e/workload', v / 1e3, self.curr_value_iter),
                    (f'top_{k}_relative_e2e/workload', v / workload_end_to_end_time, self.curr_value_iter),
                    (f'top_{k}_relative_e2e/gmre', gmre, self.curr_value_iter),
                    ('curr_value_iter', self.curr_value_iter, self.curr_value_iter),
                ])
        self.LogScalars(data)
    
    def _MakeModel(self, dataset):
        p = self.params
        model = run.MakeModel(p, self.exp, dataset)
        model.reset_weights()
        # Wrap it to get pytorch_lightning niceness.
        model = run.BalsaModel(
            p,
            model,
            loss_type=p.loss_type,
            torch_invert_cost=dataset.TorchInvertCost,
            query_featurizer=self.exp.query_featurizer,
            perturb_query_features=p.perturb_query_features,
            l2_lambda=p.l2_lambda,
            learning_rate=self.lr_schedule.Get()
            if self.adaptive_lr_schedule is None else
            self.adaptive_lr_schedule.Get(),
            optimizer_state_dict=self.prev_optimizer_state_dict,
            reduce_lr_within_val_iter=p.reduce_lr_within_val_iter)
        print('iter', self.curr_value_iter, 'lr', model.learning_rate)
        return model

    def PlanAndExecute(self, model, planner, is_test=True, max_retries=3):
        p = self.params
        model.eval()
        nodes = self.test_nodes 

        # Plan the workload.
 
        planner_config = None
        if p.planner_config is not None:
            planner_config = optim.PlannerConfig.Get(p.planner_config)

        all_execs = {}
        all_latency = {}
        balsa_planning_time = {}
        all_found_plans = {}
        for i, node in enumerate(nodes):
            print('---------------------------------------')
            query_name = node.info['query_name']
            tup = planner.plan(
                node,
                p.search_method,
                bushy=p.bushy,
                return_all_found=True,
                verbose=False,
                planner_config=planner_config,
                epsilon_greedy=0,
                # prevents Ext-JOB test query hints from failing.
                avoid_eq_filters=is_test and p.avoid_eq_filters,
            )
            planning_time, found_plan, predicted_latency, found_plans = tup
            found_plans = sorted(found_plans, key=lambda x: x[0])
            all_found_plans[query_name] = found_plans

            # Calculate monitoring info.
            predicted_costs = None
            # Model-predicted latency of the expert plan.  Allows us to track
            # what exactly the model thinks of the expert plan.
            node.info['curr_predicted_latency'] = planner.infer(node, [node])[0]
            

            # Launch tasks.
            # Roughly 5 mins.  Good enough to cover disk filled error.
            curr_timeout = 300000

            kwargs = []
            task_lambdas = []
            exec_results = []
            to_execute = []
            tasks = []
            for i, (predicted_latency, found_plan) in enumerate(found_plans):
                hint_str = found_plan.hint_str(with_physical_hints=False)
                hinted_plan = found_plan
                print('q{},[{}],(predicted {:.1f}),{}'.format(node.info['query_name'], 
                                                     i+1, predicted_latency,
                                                     hint_str))
                hint_str = run.HintStr(found_plan,
                               with_physical_hints=p.plan_physical,
                               engine=p.engine)
                to_execute.append((node.info['sql_str'], hint_str, planning_time,
                               found_plan, predicted_latency, curr_timeout))
                if p.use_cache:
                    exec_result = self.query_execution_cache.Get(
                        key=(node.info['query_name'], hint_str))
                else:
                    exec_result = None
                exec_results.append(exec_result)
                
                kwarg = {
                    'query_name': node.info['query_name'],
                    'sql_str': node.info['sql_str'],
                    'hint_str': hint_str,
                    'hinted_plan': hinted_plan,
                    'query_node': node,
                    'predicted_latency': predicted_latency,
                    'curr_timeout_ms': curr_timeout,
                    'found_plans': None,
                    'predicted_costs': None,
                    'silent': True, 
                    'is_test': is_test,
                    'use_local_execution': p.use_local_execution,
                    'plan_physical': p.plan_physical,
                    'engine': p.engine,
                    'iter_num': self.curr_value_iter,
                }
            execution_results = []
            print('{}Waiting on executing plans...value_iter={}'.format(
                '[Test set] ' if is_test else '', self.curr_value_iter))
            for i, (kwarg, exec_result) in enumerate(zip(kwargs, exec_results)):
                if exec_result is None:
                    try:
                        buffer_states = postgres.GetBufferState()
                        result_tup = ExecuteSql(**kwarg)
                    except psycopg2.errors.DiskFull:
                        # Catch double DiskFull errors and treat as a timeout.
                        # TODO: what if a test query triggered this?
                        # assert is_disk_full, 'DiskFull should happen twice.'
                        print('DiskFull happens; treating as a timeout.'
                                '  *NOTE* The agent will train on the timeout '
                                'label regardless of whether use_timeout is set.')
                        result_tup = pg_executor.Result(result=[],
                                                            has_timeout=True,
                                                            server_ip=None)
                    is_cached_plan = False
                else:
                    is_cached_plan = True
                    buffer_states = None
                    result_tup = exec_result[0][0]
                is_balsa_top1 = True if i==0 else False
                result_tups = run.ParseExecutionResult(result_tup, buffer_states, is_cached_plan, is_balsa_top1, **kwargs[i])
                assert len(result_tups) == 5
                print(result_tups[-1])  # Messages.
                if result_tups[1] < 0:   # real_latency
                    self.num_total_timeouts += 1
                    execution_results.append(np.inf)
                else:
                    execution_results.append(result_tups[1]) 
            all_latency[query_name] = execution_results
            all_execs[query_name] = [] 
            balsa_planning_time[query_name] = planning_time
            print('q{},[B], predicted {:.1f}, real {:.1f} ms'.format(node.info['query_name'], 
                                                    node.info['curr_predicted_latency'], node.cost)) 
            for i, (execute_info, real_cost) in enumerate(zip(to_execute, execution_results)):
                sql_str, hint_str, planning_time, found_plan, predicted_latency, curr_timeout = execute_info
                all_execs[query_name].append([found_plan, predicted_latency, curr_timeout, real_cost])
                print('q{},[{}], predicted {:.1f}, real {:.1f} ms, rl {:.1f}'.format(node.info['query_name'], 
                                                     i+1, predicted_latency, real_cost, real_cost/node.cost))
        return all_execs, all_latency, balsa_planning_time, all_found_plans

    def Run(self):
        # self.curr_value_iter = 0
        p = self.params
        # self.num_query_execs = 0
        self.num_total_timeouts = 0
        # self.overall_best_train_latency = np.inf
        # self.overall_best_test_latency = np.inf
        # self.overall_best_test_swa_latency = np.inf
        # self.overall_best_test_ema_latency = np.inf
        self.exp, self.exp_val = None, None
        self.exp, self.exp_val = self._MakeExperienceBuffer()
        train_ds, train_loader, _, val_loader = self._MakeDatasetAndLoader(
            log=False)
        # Fields accessed: 'costs' (for p.cross_entropy; unused);
        # 'TorchInvertCost', 'InvertCost'.  We don't access the actual data.
        # Thus, it doesn't matter if we use a Dataset referring to the entire
        # data or just the train data.  (Subset.dataset returns the entire
        # original data is where the subset is sampled.)
        #
        # The else branch is for when self.exp_val is not None
        # (p.prev_replay_buffers_glob_val).
        plans_dataset = train_ds.dataset if isinstance(
            train_ds, torch.utils.data.Subset) else train_ds
        model = self._MakeModel(plans_dataset)
        model.load_state_dict(torch.load(p.agent_checkpoint))
        planner = self._MakePlanner(model, plans_dataset)
        planner.search_until_n_complete_plans = FLAGS.topK
        all_execs, all_latency, balsa_planning_time, _ = self.PlanAndExecute(model, planner)
        self.LogTestExperience(all_latency, balsa_planning_time, with_pg=True)
        self.LogTestExperience(all_latency, balsa_planning_time, with_pg=False)


def Main(argv):
    del argv
    start_iter = FLAGS.start_iter
    end_iter = FLAGS.end_iter
    name = FLAGS.run_test
    exp_id = FLAGS.exp_id
    ckpts = np.array(range(start_iter, end_iter))
    buffer_dir = "./data/" + name + "-" + exp_id
    # model_dir = FLAGS.model_dir
    model_dir = "./qo_models/" + name + "-" + exp_id
    print('Test:', name)
    # name = "Baseline"
    p = balsa.params_registry.Get(name)
    p.use_local_execution = True
    p.sim = False
    # p.dedup_training_data = False
    # p.use_last_n_iters = 1
    # p.on_policy = False
    agent = BalsaEvaAgent(p, name, exp_id)
    agent.train_nodes = plans_lib.FilterScansOrJoins(agent.train_nodes)
    agent.test_nodes = plans_lib.FilterScansOrJoins(agent.test_nodes)
    
    for ckpt in ckpts:
        print(f'**************Test [{ckpt}]****************')
        p = agent.params
        p.prev_replay_buffers_glob = glob.glob(buffer_dir+f"/*-{ckpt}iters-*.pkl")[0]
        p.agent_checkpoint = os.path.join(model_dir, f"checkpoint_{ckpt}.pt")
        assert os.path.exists(p.prev_replay_buffers_glob), p.prev_replay_buffers_glob
        assert os.path.exists(p.agent_checkpoint), p.agent_checkpoint
        agent.curr_value_iter = ckpt
        agent.Run()

if __name__ == '__main__':
    run.set_seed(3047)
    app.run(Main)