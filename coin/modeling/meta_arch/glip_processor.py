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
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from .gdino_processor import GDINO_PROCESSOR
import os
__all__ = ["GLIP_PROCESSOR"]

@META_ARCH_REGISTRY.register()
class GLIP_PROCESSOR(GDINO_PROCESSOR):

    def pre_draw(self):
        self.save_path = 'glip_labels'
        os.makedirs(self.save_path,exist_ok=True)
        self.draw_result = False
    
    def check_support(self):
        if self.COLLECT_AUG=="": return True
        else: return False

    def load_state_dict(self, state_dict,
                        strict: bool = True):
        pass