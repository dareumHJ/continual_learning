# models/model_pool.py

class BaseModelPool:
    def __init__(self, cfg):
        self.cfg = cfg
        self.models = []
    
    def __len__(self):
        return len(self.models)