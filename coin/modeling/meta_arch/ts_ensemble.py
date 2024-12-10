# Copyright (c) Shuaifeng Li at UESTC. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.nn as nn
import torch
from detectron2.utils import comm
from collections import OrderedDict
from detectron2.utils.logger import setup_logger

class EnsembleTSModel(nn.Module):
    def __init__(self, offline_teacher, online_teacher, model_student, merge_model, log_dir):
        super(EnsembleTSModel, self).__init__()

        if isinstance(offline_teacher, (DistributedDataParallel, DataParallel)):
            offline_teacher = offline_teacher.module
        if isinstance(model_student, (DistributedDataParallel, DataParallel)):
            model_student = model_student.module

        self.offline_teacher = offline_teacher
        self.model_student = model_student
        self.online_teacher = online_teacher
        self.merge_model = merge_model
        self.logger = setup_logger(log_dir, name = __name__, distributed_rank = comm.get_rank())

    @torch.no_grad()
    def update_params(self, keep_rate, name):
        
        if comm.get_world_size() > 1:
            # When using multiple GPUs for parallel training, PyTorch will automatically distribute the model parameters on each GPU to different processes, 
            # rather than saving a complete parameter dictionary in the main process. 
            # Therefore, if you want to get the complete parameter dictionary, 
            # you need to use the items() method to iterate over the parameters in each process and merge them into a list. 
            # Only the student model uses DDP here, so only the student model needs to be operated in this way.
            student_model_dict = {
                key: value for key, value in self.model_student.state_dict().items()
            }
        else:
            student_model_dict = self.model_student.state_dict()
        
        if name == 'offline':
            new_teacher_dict = OrderedDict()
            for key, value in self.offline_teacher.state_dict().items():
                if key in student_model_dict.keys():
                    new_teacher_dict[key] = (
                            student_model_dict[key] * (1 - keep_rate) + value * keep_rate
                    )
                else:
                    raise Exception("{} is not found in student model".format(key))

            self.offline_teacher.load_state_dict(new_teacher_dict)
        else:
            raise NotImplementedError

        del new_teacher_dict
        del student_model_dict
        torch.cuda.empty_cache()