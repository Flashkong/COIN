from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from .gdino_collector import GDINO_COLLECTOR

__all__ = ["GLIP_COLLECTOR"]

@META_ARCH_REGISTRY.register()
class GLIP_COLLECTOR(GDINO_COLLECTOR):
    
    def load_state_dict(self, state_dict,
                        strict: bool = True):
        pass