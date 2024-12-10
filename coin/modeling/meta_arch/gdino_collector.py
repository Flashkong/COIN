from tqdm import tqdm
import copy
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.config import configurable
import torch
from torch import nn
from detectron2.utils import comm

__all__ = ["GDINO_COLLECTOR"]

@META_ARCH_REGISTRY.register()
class GDINO_COLLECTOR(nn.Module):
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
        meta_arch = cfg.MODEL.TEACHER_CLOUD.PROCESSOR_ARCHITECTURE
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

    def load_state_dict(self, state_dict,
                        strict: bool = True):
        self.model.load_state_dict(state_dict,strict=strict)

    def set(self, name: str, value) -> None:
        setattr(self, name, value)

    @property
    def device(self):
        return self.target_device
    
    def collect(self):
        for dataset_name, loader in self.dataloader.items():
            self._results[dataset_name] = {}
            for inputs in tqdm(loader,desc="Collecting GINO's results"):
                assert len(inputs)==1 # bs must be 1 for test
                if not inputs[0]['file_name'] in self._results[dataset_name].keys():
                    output = self.model(inputs)
                    if type(output)==list:
                        assert len(output) ==1
                        output_dict = {'file_name': inputs[0]['file_name'],'image_id': inputs[0]['image_id'],'height': inputs[0]['height'],'width': inputs[0]['width']}
                        output_dict['RCNN'] = output[0]
                        output_dict['RCNN']['instances'] = output_dict['RCNN']['instances'].to('cpu')
                        output = output_dict
                    elif type(output)==dict:
                        for key in output.keys():
                            if type(output[key]) == dict:
                                if 'instances' in output[key]:
                                    output[key]['instances'] = output[key]['instances'].to('cpu')
                    else:
                        raise NotImplementedError
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
        for dataset_name,results in self._results.items():
            if file_name in results:
                return copy.deepcopy(results[file_name])
            
    def test(self, batched_input):
        for dataset_name,results in self._results.items():
            batched_input = batched_input[0]['file_name']
            if batched_input in results:
                return [copy.deepcopy(results[batched_input]['RCNN'])]
    
    def update(self, file_name, new_result):
        for dataset_name, results in self._results.items():
            if file_name in results:
                new_result['RCNN']['instances'] = new_result['RCNN']['instances'].to('cpu')
                new_result['RPN']['instances'] = new_result['RPN']['instances'].to('cpu')
                if 'RPN_AUG' in new_result:
                    new_result['RPN_AUG']['instances'] = new_result['RPN_AUG']['instances'].to('cpu')
                self._results[dataset_name][file_name] = new_result
                return