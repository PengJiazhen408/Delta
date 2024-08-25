import os
import hashlib
import json


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

def save_history(plan, query_name, history_dir, is_test, iter_value=0, is_top1=None):
    """save_history.
        Save the plan. 
        Args:
          plan: the plan to save.
          query_name: the corresponding query name for the plan. 
          history_dir: the directory for storing historical plans.
          is_test: true if in test phase else false.

        IF in training phase:
            The location where the file is kept is: history_dir/iter_value/queryName_planSignatrueHash/plan
        IF in test phase:
            The location where the file is kept is: history_dir/queryName/planSignatrueHash/plan
    """
    plan_signature = str(get_tree_signature(plan))
    plan_hash = hash_md5_str(plan_signature)

    # print(plan)
    # print(query_name)

    # print(plan_signature)
    # print(plan_hash)

    if is_test:
        history_iter_path = os.path.join(history_dir, f"{iter_value}")
        if not os.path.exists(history_iter_path):
            os.makedirs(history_iter_path)
        history_q_path = os.path.join(history_iter_path, query_name)
        if not os.path.exists(history_q_path):
            os.makedirs(history_q_path)
        history_plan_path = os.path.join(history_q_path, plan_hash)
    else:
        history_iter_path = os.path.join(history_dir, f"{iter_value}")
        if not os.path.exists(history_iter_path):
            os.makedirs(history_iter_path)
        history_plan_path = os.path.join(history_iter_path, query_name+'_'+plan_hash)
    
    if os.path.exists(history_plan_path):
        print(f"the plan has been saved: {history_plan_path}")
        return
    else:
        os.makedirs(history_plan_path)

    with open(os.path.join(history_plan_path, "plan_signature"), "w") as f:
        f.write(plan_signature)
    with open(os.path.join(history_plan_path, "plan"), "w") as f:
        f.write(str(plan))

    if is_top1 is not None:
        with open(os.path.join(history_plan_path, f"{is_top1}"), "w") as f:
            f.write(str(is_top1))
    print(f"save history: {history_plan_path}")


if __name__ == '__main__':  
    # for example:
    is_test = True  
    query_name = "1a"
    plan = "xxxx"
    train_history_dir = "train_history"
    test_history_dir = "test_history"
    os.makedirs(train_history_dir, exist_ok=True)
    os.makedirs(test_history_dir, exist_ok=True)
    history_dir = test_history_dir if is_test else train_history_dir
    save_history(plan, query_name, history_dir, is_test, iter_value, is_top1)
