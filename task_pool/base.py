# task_pool/base.py

class BaseTaskPool:
    def __init__(self, cfg):
        self.cfg = cfg
        
    def evaluate(self, model):
        """Return dict of metrics; 여기선 dummy"""
        return {"dummy_task":0.0}