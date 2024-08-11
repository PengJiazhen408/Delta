from regression.uncertainty_model import Model 
from regression import featurize
import torch
from pytorch_lightning import loggers as pl_loggers
import os
import numpy as np
import random
from train_utils import Timer
import run
import balsa

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('run_name', 'TopK_Augment', 'Experiment config to run.')
flags.DEFINE_string('qo_name', 'Balsa_JOB_rs/55u46s73',
                     'qo_name')
flags.DEFINE_integer('iters', 200, 'all iters')
flags.DEFINE_integer('gpu_id', 1, 'gpu id used for training')

suffix = "-uncertainty"
# os.environ['WANDB_MODE'] = 'disabled'
def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
 

def _load_plans(plans_dir):
    assert os.path.exists(plans_dir), plans_dir + " does not exist, can't load plans."
    plans = []
    plans_hash = []
    for path, dir_list, file_list in os.walk(plans_dir):
        for d in dir_list:
            plans_hash.append(d)
            plan_path = os.path.join(path, d, "plan")
            with open(plan_path, "r") as f:
                plan = f.read().strip()
                plan = eval(plan)
                plans.append(plan)
    return plans, plans_hash


class TopKAgent(object):

    def __init__(self, params, device):
        self.params = params.Copy()
        p = self.params
        print('TopKAgent params:\n{}'.format(p))
        self.model_dir = os.path.join("./topK_models", p.qo_name)
        self.model_path = None
        self.model = None
        self.timer = Timer()
        self.all_plans = []
        self.plans_hash = []
        self.iters = p.all_iters
        self.cur_iter = 0
        self.train_plans = []
        self.train_plans_hash = []
        self.test_plans = []
        self.wandb_logger = None
        self._InitLogging()
        self.stages = ['ALL', 'Train']
        if p.data_augment:
            self.stages.append('Augment')
        self.plan_cache = {}
        self.duplication_num = 0
        self.device = device
        self.plan_load_iter = -1
    
    def _InitLogging(self):
        p = self.params
        experiment = str(p.cls).split('.')[-1][:-2]
        self.wandb_logger = pl_loggers.WandbLogger(save_dir=os.getcwd(), project='balsa_topK_demo', name=p.qo_name+"-"+experiment+suffix)
        p_dict = balsa.utils.SanitizeToText(dict(p))
        self.wandb_logger.log_hyperparams(p_dict)
    
    def LogScalars(self, metrics):
        if not isinstance(metrics, list):
            assert len(metrics) == 3, 'Expected (tag, val, global_step)'
            metrics = [metrics]
        d = dict([(tag, val) for tag, val, _ in metrics])
        assert len(set([gs for _, _, gs in metrics])) == 1, metrics
        self.wandb_logger.log_metrics(d)
    
    def LogPlanNum(self):
        plans = []
        for plan, plan_hash in zip(self.train_plans, self.train_plans_hash):
            duplication_num = 0
            if plan_hash in self.plan_cache.keys():
                duplication_num += 1
            else:
                self.plan_cache[plan_hash] = 1
                plans.append(plan)

        # print(len(plans))
        data_to_log = [
            ('Plans_Num/all', len(self.all_plans), self.cur_iter),
            ('Plans_Num/test', len(self.test_plans), self.cur_iter),
            ('Plans_Num/Unique train plans', len(plans), self.cur_iter),
        ]
        # X-axis.
        data_to_log.append(
            ('cur_iter', self.cur_iter, self.cur_iter))
        self.LogScalars(data_to_log)
    
    def LogTimings(self):
        """Logs timing statistics."""
        p = self.params
        num_iters_done = self.cur_iter + 1
        timings = [self.timer.GetLatestTiming(s) for s in self.stages]
        cumulative_timings = [self.timer.GetTotalTiming(s) for s in self.stages]
        data_to_log = []
        for stage, timing, cumulative_timing in zip(self.stages, timings,
                                                    cumulative_timings):
            data_to_log.extend([
                # Time, this iter.
                ('timing/{}'.format(stage), timing, self.cur_iter),
                # Total time since beginning.
                ('timing_cumulative/{}'.format(stage), cumulative_timing,
                 self.cur_iter),
            ])
        data_to_log.append(
            ('cur_iter', self.cur_iter, self.cur_iter))
        self.LogScalars(data_to_log)
    
    def extend_plans(self):
        p = self.params
        plans = self.train_plans
        all_plans = []
        before = len(plans)
        def recurse(plan, query_name):
            unique = True
            plan_sign = str(run.get_tree_signature(plan))
            plan_hash = run.hash_md5_str(plan_sign)
            plan_hash = query_name + plan_hash
            if plan_hash in self.plan_cache.keys():
                self.duplication_num += 1
                unique = False
            else:
                self.plan_cache[plan_hash] = 1

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
        
            if p.data_augment_unique and not unique:
                return rel_num
            if p.data_augment_essence:
                if rel_num > 3 and len(children) == 2:
                    runtime = plan["Actual Total Time"]
                    all_plans.append({"Plan": plan, "Execution Time": runtime})
            else:
                runtime = plan["Actual Total Time"]
                all_plans.append({"Plan": plan, "Execution Time": runtime})
            return rel_num
        
        for plan, plan_hash in zip(plans, self.train_plans_hash):
            query_name = plan_hash.split("_")[0]
            recurse(plan["Plan"], query_name)
        after = len(all_plans)
        print(f"augment. [size] before: {before}, after: {after}")
        return all_plans


    def train(self):
        p = self.params
        feature_generator = featurize.FeatureGenerator()
        feature_generator.fit(self.train_plans)
        data_to_log = []
        if p.data_augment:
            self.timer.Start("Augment")
            plans = self.extend_plans()
            self.timer.Stop("Augment")
            # data_to_log.extend([('Plans_Num/augment', len(plans), self.cur_iter)])
        else:
            plans = []
            self.plan_cache = {}
            self.duplication_num = 0
            for plan, plan_hash in zip(self.train_plans, self.train_plans_hash):
                if plan_hash in self.plan_cache.keys():
                    self.duplication_num += 1
                else:
                    self.plan_cache[plan_hash] = 1
                    plans.append(plan)
        print(f"duplication_num: {self.duplication_num}")
        data_to_log.extend([('Plans_Num/duplication_num', self.duplication_num, self.cur_iter)])
        data_to_log.extend([('Plans_Num/train_origin', len(self.train_plans), self.cur_iter)])
        data_to_log.extend([('Plans_Num/train', len(plans), self.cur_iter)])
        data_to_log.extend([('cur_iter', self.cur_iter, self.cur_iter)])
        X, Y = feature_generator.transform(plans)
        self.model = Model(feature_generator, self.device)
        x, y = self.model._feature_generator.transform(self.test_plans)
        print("Training data set size = " + str(len(X)))
        self.timer.Start("Train")
        loss, stop_epoch = self.model.fit(X, Y, x, y)
        self.timer.Stop("Train")
        data_to_log.extend([('loss/train', loss, self.cur_iter), \
            ('epoches', stop_epoch, self.cur_iter), \
            # ('Plans_Num/train', len(X), self.cur_iter), \
            # ('cur_iter', self.cur_iter, self.cur_iter)
            ])
        print("saving model in " + self.model_path)
        self.model.save(self.model_path)
        self.LogScalars(data_to_log)
    
    def load_one(self, iter):
        p = self.params
        plans_dir = os.path.join(p.train_history, p.qo_name, str(iter))
        print("load plans from ", plans_dir)
        if not os.path.exists(plans_dir):
            return 
        plans, plans_hash = _load_plans(plans_dir)
        self.all_plans.extend(plans)
        self.plans_hash.extend(plans_hash)
        assert len(self.all_plans) == len(self.plans_hash)
        self.train_plans = self.all_plans
        self.train_plans_hash = self.plans_hash

    def load_plans(self):
        if self.cur_iter > self.plan_load_iter:
            for i in range(self.plan_load_iter+1, self.cur_iter+1):
                self.load_one(i)
            self.plan_load_iter = self.cur_iter
        else:
            for i in range(0, self.cur_iter+1):
                self.all_plans = []
                self.plans_hash = []
                self.train_plans = []
                self.test_plans = []
                self.load_one(i)
            self.plan_load_iter = self.cur_iter

    def Run(self, iter_list):
        p = self.params
        self.load_one(-1)
        for iter in iter_list:
            self.cur_iter = iter
            self.plan_cache = {}
            self.duplication_num = 0
            print(f"=========={self.cur_iter}==========")
            self.load_plans()
            self.LogPlanNum()
            model_prefix = "augment" if p.data_augment else "naive"
            model_prefix = "augment-essence" if p.data_augment_essence and p.data_augment else model_prefix
            model_prefix += suffix
            self.model_path = os.path.join(self.model_dir, model_prefix, str(iter))
            self.timer.Start("ALL")
            self.train() 
            self.timer.Stop("ALL")
            for s in self.stages:
                print (f"[{s}]: cur: {self.timer.GetLatestTiming(s):.2f}s, sum: {self.timer.GetTotalTiming(s):.2f}s")
            self.LogTimings()


def main(argv):
    del argv
    name = FLAGS.run_name
    p = balsa.params_registry.Get(name)
    set_seed(p.seed)
    p.qo_name = FLAGS.qo_name
    p.all_iters = FLAGS.iters
    iter_list = range(199, p.all_iters)
    device = torch.device("cuda:" + str(FLAGS.gpu_id) if torch.cuda.is_available() else "cpu")
    agent = TopKAgent(p, device)
    # iter_list = [p.all_iters]
    agent.Run(iter_list)

if __name__ == '__main__':
    app.run(main)
