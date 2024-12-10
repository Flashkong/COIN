from tqdm import tqdm
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.config import configurable
import torch
import copy
from torch import nn
from detectron2.utils import comm
__all__ = ["CLIP_COLLECTOR"]

@META_ARCH_REGISTRY.register()
class CLIP_COLLECTOR(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        model: nn.Module = None,
    ):
        super().__init__()
        self.model = model
        self.target_device = model.device
        self._results = {}

    @classmethod
    def from_config(cls, cfg):
        meta_arch = cfg.MODEL.TEACHER_OFFLINE.META_ARCHITECTURE
        model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
        model.to(torch.device(cfg.MODEL.DEVICE))
        return {
            "model": model,
        }

    def get_results(self):
        return self._results

    def set_results(self,results):
        del self._results
        self._results = results

    def set(self, name: str, value) -> None:
        setattr(self, name, value)

    @property
    def device(self):
        return self.target_device
    
    def collect(self, pre_results):
        for dataset_name, loader in self.dataloader.items():
            pre_result = pre_results[dataset_name]
            self._results[dataset_name] = {}
            for inputs in tqdm(loader,desc="Collecting CLIP's results for model pre-train."):
                assert len(inputs)==1 # bs must be 1 for test
                if not inputs[0]['file_name'] in self._results[dataset_name].keys():
                    assert inputs[0]['file_name'] in pre_result.keys()
                    output = self.model(inputs, pre_result[inputs[0]['file_name']])
                    for key in output.keys():
                        if type(output[key]) == dict:
                            if 'instances' in output[key]:
                                output[key]['instances'] = output[key]['instances'].to('cpu')
                    self._results[dataset_name][output['file_name']] = output
            comm.synchronize()
            all_rank_results = comm.all_gather(self._results) 
            for per_rank_results in all_rank_results:
                self._results[dataset_name].update(per_rank_results[dataset_name])
    
    def delete_model(self):
        del self.model
        torch.cuda.empty_cache()

    def forward(self, file_name):
        if type(file_name)==list:
            return self.test(file_name)
        for data_name,results in self._results.items():
            if file_name in results:
                return copy.deepcopy(results[file_name])

    def test(self, batched_input):
        for dataset_name,results in self._results.items():
            batched_input = batched_input[0]['file_name']
            if batched_input in results:
                return [copy.deepcopy(results[batched_input]['RCNN'])]
    