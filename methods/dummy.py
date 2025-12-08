# methods/dummy.py

from .base import BaseMethod
from . import register_method

@register_method("dummy")
class DummyMethod(BaseMethod):
    def run(self, model_pool, task_pool):
        print(f"[DummyMethod] running with {len(model_pool)} models")
        # dummy로 model_pool의 첫 모델을 썼다고 가정
        dummy_model = None
        report = task_pool.evaluate(dummy_model)
        print(f"[DummyMethod] report: {report}")
        return report