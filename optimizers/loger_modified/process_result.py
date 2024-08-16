import os
import pandas as pd

from load_history import load_plans


def process_result(result_csv_path, history_dir, max_iter=200):
    result_df = pd.read_csv(result_csv_path)
    dic = dict()

    for iter_value in range(0, max_iter):
        history_iter_path = os.path.join(history_dir, f"{iter_value}")
        assert os.path.exists(history_iter_path), history_iter_path + " does not exist, can't load plans."
        
        q_name_list = next(os.walk(history_iter_path))[1]
        for q_name in q_name_list:
            if q_name not in dic.keys():
                dic[q_name] = set()

            q_path = os.path.join(history_iter_path, q_name)

            plans, plans_hash = load_plans(q_path)
            for plan_hash in plans_hash:
                dic[q_name].add(plan_hash)

        cnt = 0
        for key, val in dic.items():
            print(f'{key} : {len(val)} \n')
            cnt += len(val)
        print(f'iter : {iter_value}, cnt : {cnt}')
        result_df['plan_size'].loc[iter_value] = cnt
    
    result_df.to_csv(f'{result_csv_path[:-4]}_processed.csv', index=False)


if __name__ == '__main__':
    job_rs_train_history_path = "train_history_job_slow"
    job_rs_max_iter = 200
    job_rs_reslut_csv_path = "./job_slow_1.csv"
    process_result(job_rs_reslut_csv_path, job_rs_train_history_path, job_rs_max_iter)
