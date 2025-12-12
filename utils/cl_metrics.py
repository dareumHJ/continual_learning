# utils/cl_metrics.py

from typing import Dict, List
import numpy as np

def acc_from_matrix(acc_matrix: Dict[str, Dict[str, float]]) -> float:
    """
    최종 step에서의 평균 정확도 (ACC).
    acc_matrix: step(str) -> task_name -> acc
    """
    # 마지막 step 키 선택
    last_step = max(acc_matrix.keys(), key=lambda k: int(k))
    last_accs = list(acc_matrix[last_step].values())
    return float(np.mean(last_accs))

def bwt_from_matrix(
    acc_matrix: Dict[str, Dict[str, float]],
    task_order: List[str],
) -> float:
    """
    Backward Transfer (BWT) 계산.
    simple version... -> for each task i, calculate average of
      acc_after_last(i) - acc_after_training_i(i)
    
    acc_matrix: step(str) -> task_name -> acc
    task_order: 학습 순서에 사용된 task 이름 list
         e.g.: ["mnist", "fashionmnist", "kmnist"]
    """
    steps = sorted(acc_matrix.keys(), key=lambda k: int(k))
    last_step = steps[-1]
    
    diffs = []
    for i, task in enumerate(task_order):
        step_i = str(i)
        acc_after_i = acc_matrix[step_i][task]
        acc_after_last = acc_matrix[last_step][task]
        diffs.append(acc_after_last - acc_after_i)
        
    return float(np.mean(diffs))