#!/usr/bin/env python3
"""
NIA29 물류 이미지 데이터셋 모듈.

본 모듈은 NIA 29차 물류 이미지 데이터를 PyTorch Dataset 형태로 제공한다.
원본 DynamicMLP는 iNaturalist 데이터(위도/경도/날짜)를 사용하지만,
본 프로젝트에서는 상품의 물리적 가로/세로 크기(width/height)를
Dynamic MLP의 보조 메타데이터로 사용한다.
"""
import datetime
import json
import math
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

# ImageNet 사전학습 모델이 학습된 평균/표준편차로 정규화.
# 사전학습 가중치를 가져다 쓰기 때문에 동일한 정규화 파라미터를 사용해야
# 백본이 올바른 feature를 추출할 수 있다.
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def get_label_from_kan_code(kan_code, kan_code_list):
    """
    KAN_code 문자열에서 4자리 코드를 추출해 클래스 인덱스(0~106)로 변환한다.

    KAN_code는 물류 표준 분류 코드로, 앞 4자리가 대분류+중분류를 의미한다.
    8자리 전체를 쓰면 클래스 수가 너무 많아지고 샘플이 부족하므로
    앞 4자리만 사용하여 107개 클래스로 그룹화한다.
    """
    kan_value = kan_code[:4]  # KAN_code의 앞 4자리만 사용 (예: '01010110' → '0101')
    if kan_value in kan_code_list:
        # 리스트 내 인덱스가 곧 클래스 라벨(0부터 시작).
        # +1을 하지 않는 이유: CrossEntropyLoss는 0부터 시작하는 인덱스를 요구하므로.
        label = kan_code_list.index(kan_value)
    else:
        raise ValueError(f"KAN_code {kan_value} not found in predefined list.")
    return label


class NiaDataset(Dataset):
    """
    NIA29 물류 이미지 데이터셋.

    이미지(.jpg)와 어노테이션(.json)을 쌍으로 로드하여
    (이미지 텐서, 클래스 라벨, 메타데이터 벡터, 이미지 경로)를 반환한다.
    메타데이터 벡터는 상품의 가로/세로 크기를 sin/cos 인코딩한 4차원 벡터로,
    Dynamic MLP의 조건 입력으로 사용된다.
    """

    def __init__(self, root, train=True, transform=None, args=None):
        """
        데이터셋을 초기화하고 이미지 파일 목록을 구성한다.

        Args:
            root: 데이터 루트 디렉토리 (train/, train_ann/, tta/, tta_ann/ 포함)
            train: True면 학습용, False면 검증용 (tta/ 폴더 사용)
            transform: 이미지 전처리 파이프라인
            args: 학습 설정 (mlp_cin 등 메타데이터 차원 정보 포함)
        """
        self.transform = transform
        self.args = args
        self.root = root

        # KAN_code 앞 4자리 기준 107개 클래스 정의 리스트.
        # 이 리스트의 순서가 곧 클래스 인덱스(0~106)가 된다.
        # dataset.py와 eval.py의 리스트가 동일해야 한다.
        self.kan_code_list = [
            '0101', '0102', '0103', '0104', '0105', '0106', '0107', '0108', '0109', '0111',
            '0112', '0114', '0115', '0116', '0117', '0120', '0121', '0122', '0123', '0188',
            '0199', '0201', '0202', '0203', '0301', '0302', '0303', '0304', '0311', '0312',
            '0313', '0314', '0315', '0317', '0318', '0319', '0320', '0399', '0501', '0504',
            '0505', '0506', '0601', '0602', '0603', '0604', '0606', '0699', '0701', '0713',
            '0714', '0715', '0717', '0719', '0720', '0721', '0722', '0723', '0725', '0726',
            '0727', '0728', '0729', '0799', '0801', '0802', '0803', '0804', '0805', '0806',
            '0809', '0810', '0811', '0899', '0901', '0902', '0903', '0905', '0907', '0999',
            '1001', '1002', '1003', '1005', '1008', '1009', '1014', '1019', '1021', '1022',
            '1023', '1026', '1027', '1029', '1030', '1031', '1032', '1099', '1101', '1102',
            '1105', '1106', '1107', '1108', '1109', '1110', '1112',
        ]

        # 학습/검증에 따라 다른 하위 폴더를 사용.
        # 검증용으로 tta/를 사용하는 것은 원본 코드의 명명 규칙을 따른 것.
        # (TTA = Test-Time Augmentation의 약자이나, 여기서는 단순히 검증 폴더명)
        if train:
            self.img_dir = os.path.join(root, 'train/')
            self.ann_dir = os.path.join(root, 'train_ann/')
        else:
            self.img_dir = os.path.join(root, 'tta/')
            self.ann_dir = os.path.join(root, 'tta_ann/')

        # 이미지 디렉토리 하위의 모든 .jpg 파일을 재귀적으로 수집.
        # prepare_data.py로 평면 구조로 복사했으므로 직접 파일만 있지만,
        # 원본 구조(하위 폴더)에도 대응하기 위해 os.walk를 사용한다.
        self.img_list = []
        for subdir, _, files in os.walk(self.img_dir):
            for file in files:
                if file.endswith('.jpg'):
                    self.img_list.append(os.path.join(subdir, file))

    def __len__(self):
        """데이터셋의 전체 이미지 수를 반환한다."""
        return len(self.img_list)

    def __getitem__(self, idx):
        """
        idx번째 샘플을 로드하여 (이미지, 라벨, 메타데이터, 경로) 튜플을 반환한다.

        메타데이터는 상품의 물리적 가로/세로(width/height)를
        이미지 픽셀 크기로 정규화한 후 sin/cos 인코딩한 4차원 벡터이다.
        """
        # 이미지 경로에서 파일명을 추출하고, 동일한 이름의 JSON 어노테이션 경로를 생성.
        # 예: 01010110_..._1_1.jpg → 01010110_..._1_1.json
        img_path = self.img_list[idx]
        img_name = os.path.basename(img_path)
        ann_name = img_name.replace('.jpg', '.json')
        ann_path = os.path.join(self.ann_dir, ann_name)

        # 어노테이션 파일이 없으면 학습이 불가능하므로 즉시 오류를 발생시킨다.
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")

        # Windows(cp949) 환경에서 한글이 포함된 JSON을 읽기 위해 UTF-8 인코딩을 명시.
        with open(ann_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)

        # COCO 형식의 어노테이션에서 attributes 필드를 추출.
        # attributes 안에 KAN_code, width, height 등 메타데이터가 있다.
        if 'annotations' in annotation and len(annotation['annotations']) > 0:
            attributes = annotation['annotations'][0].get('attributes', None)
            if attributes is not None:
                # KAN_code 앞 4자리로부터 클래스 라벨(0~106)을 결정.
                kan_code = attributes.get('KAN_code', None)
                if kan_code is None:
                    raise KeyError(f"KAN_code not found in attributes for {ann_path}")
                label = get_label_from_kan_code(kan_code, self.kan_code_list)
            else:
                raise KeyError(f"'attributes' field not found in {ann_path}")
        else:
            raise KeyError(f"No annotations found in {ann_path}")

        # PIL로 이미지를 로드. RGB 변환은 transform에서 처리.
        img = Image.open(img_path)

        # 상품의 물리적 가로/세로 크기(cm)를 메타데이터로 사용.
        # 원본 DynamicMLP에서는 위도/경도를 사용했으나,
        # 물류 데이터에서는 상품 크기가 분류에 유의미한 정보를 제공한다.
        width = attributes.get('width', None)
        height = attributes.get('height', None)

        # 이미지의 픽셀 크기를 가져온다. 메타데이터 정규화에 사용.
        img_width, img_height = img.size

        # 상품의 물리적 크기를 이미지 픽셀 크기로 나누어 0~1 사이로 정규화.
        # 이는 상품 크기의 절대값이 아니라 이미지 내에서 차지는 비율이
        # 분류에 더 의미있는 정보이기 때문이다.
        width = float(width) / img_width
        height = float(height) / img_height

        # 0~1 범위를 -1~1 범위로 변환.
        # sin/cos 인코딩은 -1~1 입력을 가정하므로 이 변환이 필요하다.
        width = (width * 2) - 1
        height = (height * 2) - 1

        # 메타데이터가 존재하면 sin/cos 인코딩을 수행하여 4차원 벡터를 생성.
        # 메타데이터가 없으면 0벡터를 사용하여 Dynamic MLP에 영향을 주지 않도록 한다.
        if width is not None and height is not None:
            # 주의: 아래 줄에서 width/height를 다시 img_width/img_height로 나누고 있다.
            # 이미 위에서 정규화를 한 상태이므로 이중 정규화가 발생한다.
            # 기존 체크포인트와의 호환성을 위해 이 로직을 유지한다.
            # (수정하려면 처음부터 학습을 다시 해야 한다.)
            width = float(width) / img_width
            height = float(height) / img_height
            size = []
            size += [width, height]  # 항상 width/height를 메타데이터로 사용
            size = np.array(size)
            size = encode_loc_time(size)
        else:
            size = np.zeros(self.args.mlp_cin, float)

        # 이미지에 전처리 파이프라인(Resize, Crop, Flip, ToTensor, Normalize)을 적용.
        if self.transform is not None:
            img = self.transform(img)

        # (이미지 텐서, 클래스 라벨, 메타데이터 벡터, 이미지 경로)를 반환.
        # 이미지 경로는 eval.py에서 틀린 예측을 추적할 때 사용된다.
        return img, label, size, img_path


def encode_loc_time(product_size):
    """
    메타데이터 벡터를 sin/cos 인코딩하여 차원을 2배로 확장한다.

    단순 선형 값보다 sin/cos 인코딩을 사용하면:
    1. 값의 순환성(cyclic nature)을 표현할 수 있다
    2. 연속적인 값 사이의 거리가 더 잘 보존된다
    3. 신경망이 비선형 관계를 더 쉽게 학습할 수 있다

    입력: [width, height] (각 -1~1 범위, 2차원)
    출력: [sin(πw), sin(πh), cos(πw), cos(πh)] (4차원)
    """
    feats = np.concatenate((np.sin(math.pi * product_size), np.cos(math.pi * product_size)))
    return feats


def load_train_dataset(args):
    """
    학습용 DataLoader를 생성한다.

    학습용 transform은 데이터 증강을 포함하여 일반화 성능을 높인다.
    """
    # 데이터셋 이름에 따라 클래스 수를 설정.
    # NIA29_input은 KAN_code 앞 4자리 기준 107개 클래스.
    if args.data == 'NIA29_input':
        args.num_classes = 107
    else:
        raise NotImplementedError

    dataset = NiaDataset(
        root=args.data_dir,
        train=True,
        transform=transforms.Compose([
            # 256으로 리사이즈 후 224로 CenterCrop.
            # RandomResizedCrop보다 보수적이지만 샘플이 적은 환경에서 안정적.
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # transforms.RandomResizedCrop(224),  # 데이터가 충분하면 이 증강을 활성화
            transforms.RandomHorizontalFlip(),  # 좌우 반전으로 데이터 증강
            transforms.ToTensor(),
            normalize,  # ImageNet 평균/표준편차로 정규화
        ]),
        args=args,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,  # 학습 시 에포크마다 데이터 순서를 섞어 일반화 성능 향상
        num_workers=args.num_workers,
        # GPU가 있을 때만 pin_memory를 활성화하여 CPU→GPU 전송 속도를 높인다.
        pin_memory=getattr(args, 'cuda', True),
    )
    return train_loader


def load_val_dataset(args):
    """
    검증용 DataLoader를 생성한다.

    검증용 transform은 증강을 제외하고 재현 가능한 결과를 보장한다.
    """
    if args.data == 'NIA29_input':
        args.num_classes = 107
    else:
        raise NotImplementedError

    if args.tencrop:
        # TenCrop: 이미지를 4모서리 + 중앙 × 2(좌우반전) = 10장으로 증강하여 앙상블.
        # 현재는 TenCrop이 주석 처리되어 있어 실제로는 단일 CenterCrop만 동작.
        dataset = NiaDataset(
            root=args.data_dir,
            train=False,
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                # transforms.TenCrop(224),  # 10장으로 증강 (현재 비활성화)
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
            ]),
            args=args,
        )
        val_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,  # 검증 시 순서를 섞지 않아 재현 가능한 결과 보장
            num_workers=args.num_workers,
            pin_memory=getattr(args, 'cuda', True),
        )
    else:
        # 일반 검증: 증강 없이 단일 CenterCrop만 적용.
        dataset = NiaDataset(
            root=args.data_dir,
            train=False,
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]),
            args=args,
        )
        val_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=getattr(args, 'cuda', True),
        )
    return val_loader
