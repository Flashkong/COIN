# Overall environment
- Ubuntu 20.04
- Nvidia GTX 3090
- Python 3.7.16
- [CUDA 11.1](https://developer.nvidia.com/cuda-11.1.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=runfilelocal)
- cuDNN 8.0.5
- Torch 1.9.1
- detectron2 0.5
- gcc 9.4
- g++ 9.4

# Reminders
Due to the differing environment requirements of [CLIP](https://github.com/openai/CLIP), [Detectron2](https://github.com/facebookresearch/detectron2/releases), and [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO), we ultimately chose the setup mentioned above. However, this decision introduces certain challenges, which can be addressed as follows:  

1. **For Ubuntu 24.04 and Ubuntu 22.04**:  
CUDA 11.1 cannot be installed directly. To resolve this, first install `gcc 9.4` and `g++ 9.4`, then proceed with the [CUDA installation](https://developer.nvidia.com/cuda-11.1.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=runfilelocal). 
You can use the `update-alternatives` command to switch between different versions of gcc and g++.

2. **For Python 3.7.16**:  
The latest Python extension for VS Code no longer supports debugging for this version. Downgrading the extension resolves this issue. I use version `v2024.6.0`, which worked seamlessly.

# Install CUDA
First, install [CUDA 11.1](https://developer.nvidia.com/cuda-11.1.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=2004&target_type=runfilelocal) and [cuDNN 8.0.5](https://developer.nvidia.com/rdp/cudnn-archive). After the installation is complete, run the following commands to verify the setup:

```bash
nvcc -V # check cuda

cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2 # check cuDNN
```

# Install Pytorch and others
Follow the steps below to install the environment.
```bash
# create a directory to store the Grounding DINO and GLIP codebases:
cd ~
mkdir python-package

conda create -n coin python=3.7.16
conda activate coin

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
# check torch
python
>>> import torch
>>> torch.cuda.is_available()  # True
>>> torch.version.cuda  # '11.1'
>>> torch.backends.cudnn.version()  # 8005
>>> quit()

# install detectron2
python -m pip install detectron2==0.5 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

# install Grounding DINO
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
modify file: requirements.txt
    "line9:supervision>=0.22.0" -> "supervision==0.6.0"
pip install -e .

# install CLIP's requirements
pip install ftfy regex tqdm
# install pandas
pip install pandas
# modify version
pip install setuptools==59.5.0

# install GLIP
pip install einops shapely timm yacs tensorboardX prettytable pymongo transformers
git clone https://github.com/microsoft/GLIP.git
cd GLIP
python setup.py build develop --user
pip install nltk inflect scipy
```

# Final Check
Run `pip list` to verify the installed packages and their versions.

You should see an output similar to the one below. Note that exact matches are not required; similar versions should suffice as long as the installation steps above have been correctly followed.
```bash
Package                 Version            Editable project location
----------------------- ------------------ ------------------------------------
absl-py                 2.1.0
addict                  2.4.0
antlr4-python3-runtime  4.9.3
appdirs                 1.4.4
black                   21.4b2
cachetools              5.5.0
certifi                 2022.12.7
charset-normalizer      3.4.0
click                   8.1.7
cloudpickle             2.2.1
cycler                  0.11.0
detectron2              0.5+cu111
dnspython               2.3.0
einops                  0.6.1
filelock                3.12.2
fonttools               4.38.0
fsspec                  2023.1.0
ftfy                    6.1.1
future                  1.0.0
fvcore                  0.1.5.post20221221
google-auth             2.36.0
google-auth-oauthlib    0.4.6
groundingdino           0.1.0              /home/lishuaifeng/python-package/GroundingDINO
grpcio                  1.62.3
huggingface-hub         0.16.4
hydra-core              1.3.2
idna                    3.10
importlib-metadata      6.7.0
importlib-resources     5.12.0
inflect                 6.0.5
iopath                  0.1.8
joblib                  1.3.2
kiwisolver              1.4.5
Markdown                3.4.4
MarkupSafe              2.1.5
maskrcnn-benchmark      0.0.0              /home/lishuaifeng/python-package/GLIP
matplotlib              3.5.3
mypy-extensions         1.0.0
nltk                    3.8.1
numpy                   1.21.6
oauthlib                3.2.2
omegaconf               2.3.0
opencv-python           4.10.0.84
packaging               24.0
pandas                  1.3.5
pathspec                0.11.2
Pillow                  9.5.0
pip                     22.3.1
platformdirs            4.0.0
portalocker             2.7.0
prettytable             3.7.0
protobuf                3.20.3
pyasn1                  0.5.1
pyasn1-modules          0.3.0
pycocotools             2.0.7
pydantic                1.10.19
pydot                   2.0.0
pymongo                 4.7.3
pyparsing               3.1.4
python-dateutil         2.9.0.post0
pytz                    2024.2
PyYAML                  6.0.1
regex                   2024.4.16
requests                2.31.0
requests-oauthlib       2.0.0
rsa                     4.9
safetensors             0.4.5
scipy                   1.7.3
setuptools              59.5.0
shapely                 2.0.6
six                     1.16.0
supervision             0.6.0
tabulate                0.9.0
tensorboard             2.11.2
tensorboard-data-server 0.6.1
tensorboard-plugin-wit  1.8.1
tensorboardX            2.6.2.2
termcolor               2.3.0
timm                    0.9.12
tokenizers              0.13.3
toml                    0.10.2
tomli                   2.0.1
torch                   1.9.1+cu111
torchaudio              0.9.1
torchvision             0.10.1+cu111
tqdm                    4.67.1
transformers            4.30.2
typed-ast               1.5.5
typing_extensions       4.7.1
urllib3                 2.0.7
wcwidth                 0.2.13
Werkzeug                2.2.3
wheel                   0.38.4
yacs                    0.1.8
yapf                    0.43.0
zipp                    3.15.0
```

# Install with python3.9
You need to install CUDA 11.1 first. Please see [Overall](#overall-environment) and [Install CUDA](#install-cuda)

If you want to use a higher Python version, you can use Python 3.9 to set up the environment mentioned above. Additionally, [Grounding DINO1.5](https://github.com/IDEA-Research/Grounding-DINO-1.5-API) requires Python >= 3.8, so you can try setting up the environment using Python 3.9.

Below are the installation commands. Please note that our model has not been trained under Python 3.9, and we are unsure whether using Python 3.9 will achieve the expected performance.

```bash
# create a directory to store the Grounding DINO and GLIP codebases:
cd ~
mkdir python-package

conda create -n coin3.9 python=3.9
conda activate coin3.9

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy==1.23.5
# check torch
python
>>> import torch
>>> torch.cuda.is_available()  # True
>>> torch.version.cuda  # '11.1'
>>> torch.backends.cudnn.version()  # 8005
>>> quit()

# install detectron2
python -m pip install detectron2==0.5 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

# install Grounding DINO
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
pip install -e .

# install CLIP's requirements
pip install ftfy regex tqdm
# install pandas
pip install pandas
# modify version
pip install setuptools==59.5.0

# install GLIP
pip install einops shapely timm yacs tensorboardX prettytable pymongo transformers
git clone https://github.com/microsoft/GLIP.git
cd GLIP
python setup.py build develop --user
pip install nltk inflect scipy

# fix
pip install pillow==9.5.0
pip install transformers==4.30.2

# install Grounding DINO 1.5 API. 
# note that GLIP requires numpy < 1.24, but Grounding DINO 1.5 API requires numpy > 1.24
# so if you need to use Grounding DINO 1.5 API, please update numpy to 1.24.4. After install numpy 1.24.4, you can NOT run GLIP anymore.
# And detectron2 requires pillow 9.5.0, but Grounding DINO 1.5 API requires dds-cloudapi-sdk 0.2.1, which requires pillow 10.2.0
# so if you need to use Grounding DINO 1.5 API, please update numpy to 10.2.0. And fix detectron2 to be compatible with pillow 10.2.0
conda create -n coin3.9api --clone coin3.9
conda activate coin3.9api

pip install supervision==0.22.0
pip install numpy==1.24.4
pip install pillow==10.2.0 
# modify one file in detectron2. For details, please see: https://github.com/facebookresearch/detectron2/issues/5010
# BUT DO NOT RUN python3 -m pip install -U 'git+https://github.com/facebookresearch/detectron2.git@ff53992b1985b63bd3262b5a36167098e3dada02'!!!
# It will update detectron2 to v0.6, causing our code to not run properly.
modify file: '~/anaconda3/envs/coin3.9api/lib/python3.9/site-packages/detectron2/data/transforms/transform.py'
  line46: "def __init__(self, src_rect, output_size, interp=Image.LINEAR, fill=0):" 
  -> 
  "def __init__(self, src_rect, output_size, interp=Image.BILINEAR, fill=0):"
# install Grounding DINO 1.5 API. 
git clone https://github.com/IDEA-Research/Grounding-DINO-1.5-API.git
cd Grounding-DINO-1.5-API
pip install -e .
# reinstall setuptools 59.5.0. 
# This is weird, but works, when the error "AttributeError: module 'distutils' has no attribute 'version'" occurs.
pip uninstall setuptools
pip install setuptools==59.5.0
```