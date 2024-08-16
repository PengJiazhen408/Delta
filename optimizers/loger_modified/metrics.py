"""Functions of metric """

#from util import postgres
import pandas as pd
import os

def getGeometricMean(metric_qo, metric_pg):  
    """Get Geometric Mean.

    Args:
      metric_qo: list, the data get from DBM using 
           learned query optimizer (including end-to-end time and latency).
      metric_pg: list, the data get from DBM using 
           learned pg only (including end-to-end time and latency) .
    Returns:
      float, Geometric Mean.
    """
    
    total = 1
    for i, j in zip(metric_qo, metric_pg):
       total *= i / j  
    return pow(total, 1 / len(metric_qo))

def createAllMetrics():
    """Get a empty dataframe to store Metrics. 
    
    Args:
      None.
    Returns:
      DataFrame, a empty dataframe.
    """
    
    AllMetrics = pd.DataFrame(columns=['WRL', 'WRE', 'GMRL', 'GMRE', 'collectTime', 'trainTime', 'planSize'])
    return AllMetrics

def getMetrics(end2end_qo,end2end_pg,latency_qo,latency_pg):
    """Function to get metrics.

    Metrics including
    1.Workload Relative Latency(WRL), 
    2.Workload Relative End-to-End Time(WRE), 
    3.Geometric Mean Relevant Latency(GMRL),
    4.Geometric Mean Relative End-to-End Time(GMRE).

    Args:
      end2end_qo: list, the end-to-end time of DBM processing queries 
          using learned query optimizer (= Execution Time + Planning Time).
      end2end_pg: list, the end-to-end time of DBM processing queries 
          using pg only (= Execution Time + Planning Time).
      latency_qo: the execution time of queries using 
          learned query optimizer.
      latency_pg: the execution time of queries using pg only.
    Returns:
      dictory, a container storing metrics got from one iteration.
    """
   
    assert len(end2end_qo) == len(end2end_pg) == len(latency_qo) == len(latency_pg), "The length of the input four lists is not the same."
     
    Metrics = {}
    
    WRL = (sum(end2end_qo) / sum(end2end_pg))
    WRE = (sum(latency_qo) / sum(latency_pg))
    GMRL = getGeometricMean(latency_qo,latency_pg)
    GMRE = getGeometricMean(end2end_qo,end2end_pg)
    
    Metrics["WRL"] = WRL
    Metrics["WRE"] = WRE
    Metrics["GMRL"] = GMRL
    Metrics["GMRE"] = GMRE
    
    return Metrics


def combineMetrics(singleMetrics, allMetrics, collectTime, trainTime, planSize, iteration=0):
    """Combines a single Metrics with a bigger Metrics ->allMetrics.

    Args:
      singleMetrics: dictory, stroing metrics gotten from the function getMetric().
      allMetrics: DataFrame, a container to store all the metrics.
      collectTime: float, time required to collect training samples.
      trainTime: float, time required to train the qo.
      planSize: int, number of new plan executed in this iteration.
      iteration: int, which iteration this is.
    Returns:
      Dataframe, a bigger container storing Metrics.
    """
    assert iteration <= allMetrics.shape[0], "iteration is out of range "
    singleMetrics["collectTime"] = collectTime
    singleMetrics["trainTime"] = trainTime
    singleMetrics["planSize"] = planSize
    
    allMetrics = pd.concat([allMetrics, pd.DataFrame(singleMetrics, [iteration])])
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
    

if __name__ == "__main__":
    allMetrics = createAllMetrics()
    temp_list = list(range(1, 102, 10))
    #print(len(all))emp_l
    tempMetrics2 = getMetrics(temp_list,temp_list,temp_list,temp_list)
    print(tempMetrics2)
    #print(pd.DataFrame(tempMetrics))
    for i in range(5):  
        allMetrics = combineMetrics(tempMetrics2,allMetrics,10,20,30,i)
    
    allMetrics = combineMetrics(tempMetrics2,allMetrics,10,20,50,5)
    print(allMetrics)
    saveMetrics(allMetrics)
    