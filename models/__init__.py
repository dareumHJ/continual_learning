# models/__init__.py

from .model_pool import BaseModelPool

def load_model_pool(cfg):
    # 지금은 타입 하나만 지원하는 더미
    pool = BaseModelPool(cfg)
    # 나중에 여기서 실제 CLIP 모델들을 로드해서 pool.models에 추가!
    pool.models = [object(), object()] # 더미 2개
    return pool