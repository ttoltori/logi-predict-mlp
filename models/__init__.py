"""
모델 패키지 초기화 모듈.

본 파일은 models/ 디렉토리 내의 모든 모델 모듈을 임포트하여
train.py에서 models.__dict__[model_file].__dict__[model_name] 형태로
동적 모델 선택이 가능하도록 한다.

지원 모델:
  - sk2res2net_dynamic_mlp: SK2Res2Net + Dynamic MLP (주력)
  - sk2res2net: SK2Res2Net only (image-only 비교용)
  - resnet_dynamic_mlp: ResNet + Dynamic MLP
  - resnet: ResNet only (image-only 비교용)
"""
from .sk2res2net_dynamic_mlp import *
from .sk2res2net import *
from .resnet_dynamic_mlp import *
from .resnet import *
