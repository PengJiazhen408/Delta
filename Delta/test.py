T = 500
topK = 5
basemodel = '_balsa' # _loger, _balsa
pg_plan_dir = "PG_Plans/JOB/rs" # PG_Plans/TPCDS, PG_Plans/JOB/rs, PG_Plans/JOB/slow, PG_Plans/STACK
# job rs
pg_exec_dict = {'15d': 733.713, '2a': 1015.485, '8a': 942.421, '12b': 142.699, '15b': 36.847, '15a': 381.905, '4c': 162.877, '26b': 163.89, '17e': 18425.863, '20a': 2878.609, '24b': 647.899, '22b': 312.216, '4b': 110.244, '16a': 259.463, '22c': 3551.677, '10a': 388.766, '10b': 288.986, '17c': 7390.888, '30c': 3807.331}

# # job slow
# pg_exec_dict = {'18c': 5399.613, '25c': 9350.164, '19c': 449.683, '9d': 4882.497, '26c': 8095.416, '19d': 8070.566, '6d': 5742.657, '16b': 33309.496, '17e': 18146.3, '20a': 2926.603, '17a': 18160.169, '30a': 3711.606, '10c': 6911.177, '17d': 8468.091, '17b': 8661.482, '6f': 6492.148, '17c': 8229.624, '8c': 5935.364, '17f': 14664.4}
    
# TPC-DS
# pg_exec_dict = {'query98_3': 3376.013, 'query82_2': 7418.183, 'query18_3': 2524.767, 'query98_1': 3346.911, 'query27_2': 5753.706, 'query27_3': 5737.1, 'query52_1': 3547.815, 'query98_2': 3342.593, 'query52_2': 3484.015, 'query82_3': 447.585, 'query27_1': 5681.117, 'query18_1': 2793.231, 'query18_2': 3539.971, 'query82_1': 473.897, 'query52_3': 3418.466}
    
# STACK
# pg_exec_dict =  {'q14_2_60e10e94f8356f019e90caff1adae7b3c7e7df82': 898.607, 'q5_2_q5-027': 58.125, 'q3_0_q3-033': 749.478, 'q3_1_q3-092': 655.017, 'q5_0_q5-018': 56.335, 'q1_2_q1-072': 91.833, 'q3_2_q3-066': 466.012, 'q5_4_q5-091': 37.193, 'q5_1_q5-068': 53.323, 'q14_1_242c393daaec760e4c1597c1bfa8c8f21dc8eb78': 214.762, 'q16_2_ab899cf3a1d5aad39faec3a8a48389e86cd0ba9d': 2762.318, 'q16_3_e25cd79d91cb036a4b162831ca91ef70de7e3740': 9535.739, 'q12_4_75dd50f0d2debbbbc571ab068a7b97e2efeeb3fa': 600.716, 'q16_0_c5dbc1eb440ba3eef8f8c6ad6659d1c20071dfc7': 726.584, 'q3_4_q3-001': 532.135, 'q12_1_1349d223cd64d6506968a91e84881bb7069ef83a': 85.462, 'q3_3_q3-045': 652.427, 'q14_0_63c0776f1727638316b966fe748df7cc585a335b': 88.636, 'q14_3_6fa85fc0fe36ff6f4f7ce7ce340177ffd4f8ace0': 453.33, 'q16_1_469b915d6ca7078b4c66ea6256b8eb3d77305f9f': 421.675, 'q12_2_062bfd2b89537ed5fe4a1a6b488860f4587e54ec': 6357.103, 'q16_4_60adfa44cd3f671e1f74b162796cd659cb9630ac': 21462.251, 'q14_4_03aae0b5a60091b040d0cb97d8bbd78d203b6a44': 658.994, 'q12_3_f919c1ec2117227e9821f0ad96a153017c381b56': 125.228, 'q1_0_q1-041': 34.156, 'q1_1_q1-033': 32.956, 'q5_3_q5-040': 49.973, 'q1_4_q1-009': 32.53, 'q1_3_q1-047': 33.837, 'q12_0_ae8c54ce8fa00e0497293c791b8ce5c85932eb36': 17957.907}

from regression.uncertainty_model import Model 
from regression import featurize
import torch
import os
import numpy as np
import random

import run

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import pandas as pd
import time

FLAGS = flags.FLAGS
flags.DEFINE_string('qo_name', 'Balsa_JOB_rs/55u46s73',
                     'qo_name')
flags.DEFINE_string('run_name', 'augment-uncertainty', 'Experiment config to run.')
flags.DEFINE_string('test_dir', 'test_history', 'Test Dir')
flags.DEFINE_integer('iters', 100, 'model iter')
flags.DEFINE_integer('gpu_id', 2, 'gpu id used for training')

def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

def extend_plans(train_plans, train_plans_hash, plan_cache):
    all_plans = []
    before = len(train_plans)
    def recurse(plan, query_name):
        unique = True
        plan_sign = str(run.get_tree_signature(plan))
        plan_hash = run.hash_md5_str(plan_sign)
        plan_hash = query_name + plan_hash
        if plan_hash in plan_cache.keys():
            unique = False
        else:
            plan_cache[plan_hash] = 1
        rel_num = 1
        children = plan.get("Plans", [])
        if len(children) == 0:
            rel_num = 1
        if len(children) == 1:
            rel_num = recurse(children[0], query_name)
        if len(children) == 2:
            left_num = recurse(children[0], query_name)
            right_num = recurse(children[1], query_name)
            rel_num = left_num + right_num
        runtime = plan["Actual Total Time"]
        all_plans.append({"Plan": plan, "Execution Time": runtime})
        return rel_num
    
    for plan, plan_hash in zip(train_plans, train_plans_hash):
        query_name = plan_hash.split("_")[0]
        recurse(plan["Plan"], query_name)
    after = len(all_plans)
    print(f"augment. [size] before: {before}, after: {after}")
    return all_plans

def _load_plans(plans_dir):
    assert os.path.exists(plans_dir), plans_dir + " does not exist, can't load plans."
    plans = []
    plans_hash = []
    for path, dir_list, file_list in os.walk(plans_dir):
        hashs = []
        tops = []
        for d in dir_list:
            # print(d)
            if basemodel=='_balsa':
                if os.path.exists(os.path.join(path, d, "is_balsa_top1")):
                    plans_hash.append(d+"_top1")
                else:
                    plans_hash.append(d)
            elif basemodel=='_loger':
                hashs.append(d)
                f_generator = os.walk(os.path.join(path, d))
                files = next(f_generator)[2]
                tags = [x for x in files if 'top' in x]
                if len(tags) != 0:
                    tag =  tags[0]
                    tops.append(tag)

            plan_path = os.path.join(path, d, "plan")
            with open(plan_path, "r") as f:
                try:
                    plan = f.read().strip()
                    plan = eval(plan)
                    plans.append(plan)
                except:
                    continue
        if basemodel == '_loger':
            if len(tops) != 0:
                min_idx = np.argmin(np.array(tops))
                hashs[min_idx] = hashs[min_idx] + "_" + tops[min_idx]
            plans_hash.extend(hashs)
        
    return plans, plans_hash

def load_pg_plans():
    plans_dir = pg_plan_dir
    pg_plans, pg_plans_hash = _load_plans(plans_dir)
    q2pgPlans = {}
    for h, p in zip(pg_plans_hash, pg_plans):
        qname = h[:h.rfind('_')]
        q2pgPlans[qname]=p
        exec_time = p["Execution Time"]
        # print(f"{qname}: {exec_time} ms")
    return q2pgPlans

def load_plans(test_dir):
    print(test_dir)
    dirs = os.walk(test_dir)
    _, test_queries, _ = next(dirs)
    q2plans = {}
    q2balsaPlans = {}
    for q in test_queries:
        path = os.path.join(test_dir, q)
        plans, plans_hash = _load_plans(path)
        plans.sort(key= lambda item: item["Execution Time"])
        q2plans[q] = plans
        balsa_plan_hash = list(filter(lambda x: 'top' in x, plans_hash))
        if len(balsa_plan_hash) == 1: # len(balsa_plan_hash) == 0  means timeout
            balsa_plan_index = plans_hash.index(balsa_plan_hash[0])
            balsa_plan = plans[balsa_plan_index]
            q2balsaPlans[q] = balsa_plan
    return q2plans, q2balsaPlans

def load_train_plans(train_history, qo_name, iters):
    train_plans = []
    plan_hashs = []
    plan_cache = {}
    def load_one(iter):
        plans_dir = os.path.join(train_history, qo_name, str(iter))
        if not os.path.exists(plans_dir):
                return 
        plans, plans_hash = _load_plans(plans_dir)
        for plan, plan_hash in zip(plans, plans_hash):
            if plan_hash not in plan_cache.keys():
                plan_cache[plan_hash] = 1
                train_plans.append(plan)
                plan_hashs.append(plan_hash)
    load_one(-1)
    for i in range(0, iters+1):
        load_one(i)
    # if qo_name == "loger_tpcds":
    # if pg_plan_dir == "PG_Plans/TPCDS": 
    #     train_plans = extend_plans(train_plans, plan_hashs, plan_cache)
    # train_plans = extend_plans(train_plans, plan_hashs, plan_cache)
    return train_plans


def getGeometricMean(metric_qo, metric_pg):
    total = 1
    for i, j in zip(metric_qo, metric_pg):
       total *= i / j
    return pow(total, 1 / len(metric_qo))


def getMetrics(latency_pg,latency_pred):
    
    WRL = (sum(latency_pred) / sum(latency_pg))
    GMRL = getGeometricMean(latency_pred, latency_pg)

    return WRL, GMRL

def combineMetrics(singleMetrics, allMetrics, interation=0):

    # assert interation <= allMetrics.shape[0], "interation is out of range "
    
    allMetrics = pd.concat([allMetrics, pd.DataFrame(singleMetrics, [interation])])
    return allMetrics

def saveMetrics(Metrics, path = './data/Metrics.csv'):
    """Save Metrics in to a csv-file. 
    
    Args:
      Metrics: DataFrame, a container storing metrics.
      path: str,  the path of csv file.
    Returns:
      None.  
    """
    
    current_dir = os.path.dirname(path)
    
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
        print("Path: "+current_dir+" has been created successfully!\n")

    Metrics.to_csv(path, sep=',',mode='w')
    
    return 

def mahalanobis_distance_to_distribution(x, data, device, regularization=1e-5):
    """
    Calculate the Mahalanobis distance from a sample to a distribution

    Args:
    - x: torch.Tensor, the sample vector to be tested
    - data: torch.Tensor, a tensor containing multiple samples, each row represents a sample

    Returns:
    - mahalanobis_dist: torch.Tensor, the Mahalanobis distance
    """
    # Calculate the mean of the data
    mean = torch.mean(data, dim=0)

    # Center the data by subtracting the mean
    centered_data = data - mean

    # Calculate the covariance matrix
    covariance_matrix = torch.matmul(centered_data.t(), centered_data) / (data.size(0))
    # covariance_matrix += regularization * torch.eye(covariance_matrix.size(0)).to(device)

    # Calculate the difference from the sample to the mean
    diff = x - mean

    # Calculate the inverse of the covariance matrix
    try:
        covariance_inv = torch.inverse(covariance_matrix)
    # except torch._C._LinAlgError as e:
    except:
        # print(f"Error: {e}")
        print("Handling singular matrix using pseudo-inverse...")
        covariance_inv = torch.pinverse(covariance_matrix)

     # Calculate the Mahalanobis distance
    mahalanobis_dist = torch.sqrt(torch.einsum('i,ij,j->', diff, covariance_inv, diff))

    return mahalanobis_dist.item()

def test(model, q2plans, q2pgPlans, q2balsaPlans, train_plans, device):
    min_gths = []
    min_gth_wps = []
    min_preds= []
    min_preds_de= []
    balsa_des = []
    pg_exec = []
    balsa_exec = []
    q2rank, q2mse, q2unc = {}, {}, {}
    TP, FP, FN, TN = 0,0,0,0
    TP_balsa, FP_balsa, FN_balsa, TN_balsa = 0,0,0,0
    pg_mses, pg_uncs, uncs  = [], [], []
    goods = []
    gains = []
    pgPlans = list(q2pgPlans.values())
    n = len(train_plans)
    m = len(q2pgPlans)
    X, Y = model._feature_generator.transform(train_plans+pgPlans)
    flat_trees, _= model._net.build_trees(X, device)
    tmp = flat_trees.reshape(n+m, -1)
    data = tmp[:n]
    xs = tmp[n:]
    x_dicts = {}
    for x, y in zip(q2pgPlans.keys(), xs):
        x_dicts[x] = y
    
    for q, plans in q2plans.items():
        
        pg_time = pg_exec_dict[q]
        balsa_plan = q2balsaPlans.get(q, 300000)
        balsa_time = 300000 if balsa_plan == 300000 else balsa_plan["Execution Time"]

        # pg plan
        pgPlan = q2pgPlans[q]
        X, Y = model._feature_generator.transform([pgPlan])
        pg_pred, pg_mse, pg_uncertainty = model.test(X,Y)
        # pg_mse = abs(pg_pred-pg_time)/pg_time
        pg_mses.append(pg_mse)
        
        x = x_dicts[q]
        
        dist = mahalanobis_distance_to_distribution(x, data, device)

        score = dist
        print("dist:", dist)
        pg_uncs.append(score)

        X, Y = model._feature_generator.transform(plans)
        y_pred, mse, unc = model.test(X, Y)
        uncs.append(unc)
        plan_num = len(plans)
        good_plan = sum([1 if plans[i]["Execution Time"] < pg_time else 0 for i in range(plan_num)])
        goods += [1 if good_plan != 0 else 0]
        best_plan = plans[0]["Execution Time"]
        gains.append((pg_time-best_plan)/pg_time)
        # y_pred = model._feature_generator.normalizer.inverse_norm(y_pred, "Execution Time")
        rank = np.argsort(y_pred)
        # print(rank)
        q2rank[q] = rank
        q2mse[q] = mse
        q2unc[q] = unc
        
        min_gth = plans[0]["Execution Time"]
        min_gths.append(min_gth)
        min_gth_wp = min(min_gth, pg_time)
        min_gth_wps.append(min_gth_wp)

        # according to pg plan uncertainty
        
        min_pred = plans[rank[0]]["Execution Time"]
        min_preds.append(min_pred)
        if score <= T:
            if min_gth < pg_time:
                TP += 1   # learned OK 
            else:
                FP += 1   # avoid, regression
            
            if balsa_time < pg_time:
                TP_balsa += 1   # learned OK 
            else:
                FP_balsa += 1   # avoid, regression
            balsa_de = balsa_time 

        else:
            if min_gth < pg_time:
                FN += 1  # pity
            else:
                TN += 1  # prevent regression
            min_pred = pg_time
            
            if balsa_time < pg_time:
                FN_balsa += 1  # pity
            else:
                TN_balsa += 1  # prevent regression
            balsa_de = pg_time

        balsa_des.append(balsa_de)
        min_preds_de.append(min_pred)
        pg_exec.append(pg_time)
        balsa_exec.append(balsa_time)
        pg_true = pgPlan["Execution Time"]
        print(f'---{q}---')
        print(f"PG plan: baseline: {pg_time:.2f} ms, true time: {pg_true:.2f} ms, pred time: {pg_pred:.2f} ms, mse: {pg_mse:.6f}, unc: {pg_uncertainty:.6f}")
        print(f"Balsa plan: exec time: {balsa_time: .2f} ms; dector: {balsa_de:.2f} ms")
        print(f"top-k + de: rank: {rank}, pred_min: {min_pred:.2f} ms, min: {min_gth} ms, mse: {mse:.2f}, unc: {unc:.6f}")
        print(f"double-q: num: {plan_num}, good: {good_plan}")
    Metrics = {}
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    WRL, GMRL = getMetrics(pg_exec, balsa_exec)
    print(f"[Balsa] WRL: {WRL:.2f}, GMRL: {GMRL:.2f}")
    Metrics['WRL'+basemodel] = WRL
    Metrics['GMRL'+basemodel] = GMRL
    WRL, GMRL = getMetrics(pg_exec, min_gths)
    print(f"[Best] WRL: {WRL:.2f}, GMRL: {GMRL:.2f}")
    WRL, GMRL = getMetrics(pg_exec, min_gth_wps)
    print(f"[BsWp] WRL: {WRL:.2f}, GMRL: {GMRL:.2f}")
    WRL, GMRL = getMetrics(pg_exec, min_preds)
    print(f"[Model] WRL: {WRL:.2f}, GMRL: {GMRL:.2f}")
    Metrics['WRL_mp'] = WRL
    Metrics['GMRL_mp'] = GMRL
    WRL, GMRL = getMetrics(pg_exec, balsa_des)
    print(f"[Detector] WRL: {WRL:.2f}, GMRL: {GMRL:.2f}")
    Metrics['WRL'+basemodel+'_pgdist'] = WRL
    Metrics['GMRL'+basemodel+'_pgdist'] = GMRL
    WRL, GMRL = getMetrics(pg_exec, min_preds_de)
    print(f"[Ours] WRL: {WRL:.2f}, GMRL: {GMRL:.2f}")
    Metrics['WRL_pgdist'] = WRL
    Metrics['GMRL_pgdist'] = GMRL
    return Metrics
    
def main(argv):
    del argv
    name = FLAGS.run_name
    set_seed(3407)
    qo_name = FLAGS.qo_name
    run_name = FLAGS.run_name
    iters = FLAGS.iters
    # if basemodel == '_balsa':
    #     balsa_id = qo_name.split("-")[1]
    test_dir = FLAGS.test_dir
    # device = torch.device("cuda: " + str(FLAGS.gpu_id) if torch.cuda.is_available() else "cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    allMetrics = pd.DataFrame(columns=['WRL'+basemodel, 'WRL_mp', 'WRL'+basemodel+'_pgdist', 'WRL_pgdist', 'GMRL'+basemodel, 'GMRL_mp', 'GMRL'+basemodel+'_pgdist', 'GMRL_pgdist'])
    train_dir = "./train_history"
    for iter in range(0, iters):
        print(f"----{iter}----")
        if basemodel == '_balsa':
            test_dir_iter = os.path.join(test_dir+f"/test_history_k_{topK}", str(iter))
        else:
            test_dir_iter = os.path.join(test_dir+"/"+qo_name+"/top10_beam20", str(iter))
        if not os.path.exists(test_dir_iter):
            continue
        q2plans, q2balsaPlans = load_plans(test_dir_iter)
        q2pgPlan = load_pg_plans()
        train_plans = load_train_plans(train_dir, qo_name, iter)
        model_dir = os.path.join("./topK_models", qo_name)
        model_path = os.path.join(model_dir, run_name, str(iter))
        model = Model(None, device)
        model.load(model_path)
    
        dirs = os.path.join("ranks", qo_name, run_name, f"myfeat_{T}") + f"_{iter}"
        Metrics = test(model, q2plans, q2pgPlan, q2balsaPlans, train_plans, device)
        allMetrics = combineMetrics(Metrics, allMetrics, iter)
        saveMetrics(allMetrics, os.path.join("results", qo_name, run_name)+f"k={topK}_myfeat_pgdist_{T}.csv")


if __name__ == '__main__':
    app.run(main)
