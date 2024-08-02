import psycopg2
import numpy as np
import os


wld2db = {
    'JOB': 'imdb',
    'STACK': 'stack',
    'TPCH': 'tpch',
    'TPCDS': 'tpcds',
}

def connect_db(database, port=5432):
    try:
        conn = psycopg2.connect(database=database, user='postgres',
                                password='', host='127.0.0.1', port=port)
    except Exception as e:
        print(e)
    else:
        print("Opened database successfully")
        return conn
    return None

def close_db_connection(conn):
    conn.commit()
    conn.close()


def execute_sql(cur, sql):
    # print(sql)
    cur.execute(sql)
    result = cur.fetchall()
    # print(result[-1][0])
    return result

def get_rels_and_idx(cur):
    sql = '''
    SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
AND table_name not like 'pg%'
ORDER BY table_name;
    '''
    
    results = execute_sql(cur, sql)
    relations = []
    indexes = []
    for r in results:
        r = r[0]
        relations.append(r)
        sql = f'select indexname from pg_indexes where tablename = \'{r}\';'
        results = execute_sql(cur, sql)
        for idx in results:
            idx = idx[0]
            indexes.append(idx)
    return relations, indexes
