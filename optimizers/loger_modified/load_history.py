import os
import hashlib
import json

def load_plans(plans_dir):
    """load_plans.

        Load plans from plans_dir
        
        Example: the path is: plans_dir/planSignatrueHash/plan

        Args:
          plans_dir:the directory to load historical plans. 
        Output: 
          The plans and their hash values from the plans_dir.
    """
    assert os.path.exists(plans_dir), plans_dir + " does not exist, can't load plans."
    plans = []
    plans_hash = []
    for path, dir_list, file_list in os.walk(plans_dir):  
        for d in dir_list:
            plans_hash.append(d)
            plan_path = os.path.join(path, d, "plan")
            with open(plan_path, "r") as f:
                plan = f.read().strip()
                try:
                    plan = eval(plan)
                    plans.append(plan)
                except:
                    pass
    return plans, plans_hash