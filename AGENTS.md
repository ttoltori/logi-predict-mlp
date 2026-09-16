# AGENTS.md — DynamicMLP (NIA29 물류 이미지 분류 적용판)

이 파일은 본 프로젝트를 매번 재분석하지 않아도 되도록 핵심 구조·실행 방법·주의사항을 정리한 참조 문서입니다.
원본 프로젝트: https://github.com/ylingfeng/DynamicMLP (논문 "Dynamic MLP for Fine-Grained Image Classification by Leveraging Geographical and Temporal Information")

> 본 저장소는 원본(iNaturalist 생물 분류용)을 **한국 NIA 29차 물류 이미지 데이터(택배/바코드, KAN_code 기준 107개 클래스)** 분류용으로 개조한 fork입니다. 메타데이터로 위도/경도/날짜 대신 **상품의 물리적 가로/세로 크기(width/height)** 를 Dynamic MLP의 조건 입력으로 사용합니다.

AI와의 작업 세션에서 소스 변경이 발생하면 반드시 아래의 파일들도 없데이트 한다.
- AGENTS.md
- README.md
docs/학습및검증절차.md

---

## 1. 프로젝트 개요

- **목표**: 물류 이미지(포장/바코드)를 KAN_code 앞 4자리 기준 107개 클래스로 세분류(fine-grained classification).
- **핵심 아이디어**: 이미지 특징만 쓰는 백본에 **Dynamic MLP** 를 결합해, 보조 메타데이터(상품 가로/세로)로부터 동적으로 가중치를 생성해 이미지 feature를 변조(fusion). 원본의 geo/temporal 임베딩 자리를 상품 size 임베딩으로 치환.
- **입력**: (1) 224×224 RGB 이미지, (2) 4차원 메타데이터 벡터(`encode_loc_time`으로 sin/cos 인코딩된 width/height).
- **출력**: 107-way 클래스 로짓.
- **백본**: ResNet-50/101 또는 SK-Res2Net-101 (주로 `sk2res2net101` 사용).

---

## 2. 디렉토리 구조

```
DynamicMLP/
├── train.py                     # 학습 엔트리포인트 (argparse + train/validate 루프)
├── eval.py                      # 평가 스크립트 (confusion matrix, 클래스별 로깅)
├── dataset.py                   # ★ NIA29 커스텀 데이터셋 (NiaDataset) — 현재 사용 중
├── dataset_oryginal.py          # 원본 INatDataset (iNaturalist용, 참고용)
├── prepare_data.py              # ★ 전처리: samples → datasets 재분포 (--src 세미콜론 다중 지정)
├── utils.py                     # mixup, LabelSmoothing, accuracy, LR 스케줄, 체크포인트
├── requirements.txt             # ★ Linux 서버용 패키지 목록 (pip install -r)
├── .gitignore                   # Git 제외 설정 (.venv/, *.pth, datasets/ 등)
├── .venv/                       # Python 가상환경 (로컬, Git 제외)
├── models/
│   ├── __init__.py              # 4개 모델 파일을 모두 import
│   ├── dynamic_mlp.py           # ★ Dynamic MLP 코어 (variant A/B/C, FusionModule, FCNet)
│   ├── resnet.py                # ResNet 백본 (image-only)
│   ├── resnet_dynamic_mlp.py    # ResNet + Dynamic MLP 결합 모델
│   ├── sk2res2net.py            # SK-Res2Net 백본 (image-only)
│   └── sk2res2net_dynamic_mlp.py# ★ 주력 모델: SK2Res2Net + Dynamic MLP
├── checkpoints/                 # 사전학습 가중치 (sk2res2net101_epoch_300.pth 권장)
├── datasets/                    # 학습용 데이터 (prepare_data.py가 생성, Git 제외)
├── samples/                     # NIA29 원본 샘플 데이터 (Git 제외)
├── logs/                        # 학습 로그/체크포인트 ({save_dir}/{data}/{name}/)
├── eval/                        # eval.py 결과 로그 ({save_dir}/{data}_log_{timestamp}/)
├── figs/structure.svg           # 모델 구조도
├── docs/학습및검증절차.md        # 학습/검증 절차 가이드
└── README.md                    # 프로젝트 설명
```

---

## 3. 데이터 형식 (NIA29)

### 3.1 디렉토리 레이아웃 (`--data_dir` 로 지정하는 경로)

```
<data_dir>/
├── train/                       # 학습 이미지 (하위 폴더 재귀 탐색, .jpg)
│   └── <대분류>/<중분류>/<소분류>/<KAN_id>/<file>.jpg
├── train_ann/                   # 학습 어노테이션 (.json, 이미지 파일명과 1:1)
├── tta/                         # 검증 이미지 (val/ 가 아님 — dataset.py가 tta/ 사용)
└── tta_ann/                     # 검증 어노테이션
```

> **주의**: `dataset.py`의 검증 경로는 `val/`·`val_ann/`이 아니라 `tta/`·`tta_ann/`입니다. 주석에 "검증시 검증 데이터로변경 val/" 라는 안내가 있음.

> **입고/출고 분리**: 원본 데이터는 `01_입고물품`/`02_출고물품`으로 분류되며,
> 서로 다른 모델의 학습 대상. `prepare_data.py --category inbound|outbound`로 선택.
> 기본값은 `inbound`(입고물품). 카테고리 전환 시 `--clean` 필수.

이미지 파일명 규칙(샘플 기준): `{KAN_code}_{id}_{shot_id}_{cam_id}.jpg`
예: `01010110_8809251219122_1_1.jpg` → KAN_code=`01010110`

### 3.2 어노테이션 JSON (COCO-like, 이미지 1장당 1개 파일)

```json
{
  "info": { "contributor": "...", "project_name": "29. 물류 이미지 데이터" },
  "images": [{
    "id": 53635, "file_name": "01010110_..._1_1.jpg",
    "width": 1920, "height": 1080, "shot_id": 1, "cam_id": 1
  }],
  "annotations": [{
    "id": 53635, "image_id": 53635, "category_id": 1,
    "segmentation": [[...]],
    "attributes": {
      "length": 8.5, "width": 4.3, "height": 21.4, "weight": 0.564,
      "fragile": true, "KAN_code": "01010110", "size_id": 1, "caption": "..."
    }
  }]
}
```

### 3.3 라벨 매핑

- `dataset.py` 상단의 `kan_code_list` (107개 항목)가 클래스 순서를 정의.
- 라벨 = `kan_code_list.index(KAN_code[:4])` (0-based).
- `eval.py` 상단의 `number_labels`는 동일한 107개 리스트.
- **리스트 외 KAN_code 자동 제외**: `dataset.py` `__init__`에서 파일명 앞 4자리가 `kan_code_list`에 없는 이미지를 필터링하고 개수를 출력한다(2026-09 추가). `prepare_data.py`의 `scan_products`도 `KAN_CODE_LIST`로 상품을 필터링한다. 현재 데이터셋에는 리스트 외 코드 5개(0611, 1025, 1114, 1115, 1116 — train 183장/tta 53장)가 존재하며 자동 제외된다.
- `args.num_classes = 107` (`dataset.py`의 `load_train_dataset`/`load_val_dataset`에서 `args.data == 'NIA29_input'` 일 때 하드코딩).

### 3.4 메타데이터 처리 (dataset.py `__getitem__`)

1. `attributes.width`, `attributes.height` (상품 물리 크기)를 읽음.
2. 이미지 픽셀 크기(`img_width`, `img_height`)로 나누어 0~1 정규화.
3. `*2 - 1` 로 -1~1 스케일링.
4. `encode_loc_time([width, height])` → `[sin(πw), sin(πh), cos(πw), cos(πh)]` (4차원).
5. 이 4차원 벡터가 Dynamic MLP의 `loc` 입력(`args.mlp_cin = 4`)으로 사용.

> **주의(코드 결함)**: `dataset.py` 106~111줄에서 width/height를 이미 정규화한 뒤, 113~115줄에서 다시 `float(width)/img_width` 등으로 한 번 더 나눕니다. 결과적으로 의도한 스케일이 아니지만 현재 학습된 체크포인트는 이 로직 기준이므로 함부로 수정하면 재학습 필요. 메타데이터가 없으면 `np.zeros(args.mlp_cin)` 사용.

---

## 4. 모델 아키텍처

### 4.1 Dynamic MLP (`models/dynamic_mlp.py`)

- `FCNet`: 메타데이터(loc)를 임베딩하는 작은 FC 네트워크(Linear → ReLU → FCResLayer×4 → Linear). 출력 차원 = `args.mlp_d`(256).
- `Dynamic_MLP_A/B/C`: 이미지 feature와 loc 임베딩으로부터 **동적 가중치**를 생성해 `torch.bmm` 으로 feature를 변조. variant C가 기본(`--mlp_type c`)이며 img+loc을 concat해서 가중치 생성.
- `RecursiveBlock`: Dynamic MLP 1층 래퍼.
- `FusionModule`: `conv1`(2048→256) → RecursiveBlock×`mlp_n`(기본 2) → `conv3`(256→2048) + residual. 백본 feature(2048차원)를 loc 조건부로 변조.
- `get_dynamic_mlp(inplanes, args)`: args(`mlp_d`, `mlp_h`, `mlp_n`, `mlp_type`)로 FusionModule 생성.

### 4.2 백본 + Dynamic MLP 결합 (`resnet_dynamic_mlp.py`, `sk2res2net_dynamic_mlp.py`)

forward 흐름:
```
image → conv1/stem → layer1~4 → avgpool → flatten → (2048차원 feature)
loc   → loc_net(FCNet) → loc_fea(256차원)
feature, loc_fea → loc_att(FusionModule) → fused feature(2048차원)
fused → fc → logits(num_classes)
```

- `resnet_dynamic_mlp.py`: torchvision ResNet 구조에 `loc_net`, `loc_att` 추가. `resnet50`/`resnet101` 제공. `--pretrained` 시 ImageNet 가중치 로드(fc 제외).
- `sk2res2net_dynamic_mlp.py`: SK2Res2Net(Res2Net + SK attention) 백본에 동일 결합. `sk2res2net101` 제공. **`--pretrained` 시 `checkpoints/sk2res2net101_epoch_300.pth` (ImageNet 사전학습)를 로드** — checkpoints/README.md 링크에서 다운로드 필요. `strict=False`로 로드되므로 백본 가중치만 적용되고 `loc_net`/`loc_att`/`fc`는 랜덤 초기화됨.

### 4.3 image-only 모드

`--image_only` 시 `net(images)`만 호출하지만, 모델 forward 시그니처가 `(x, loc)`이므로 image-only 전용 백본(`resnet.py`, `sk2res2net.py`)을 `--model_file`로 지정해야 함.

---

## 5. 학습 파이프라인 (`train.py`)

### 5.1 주요 인자 (argparse)

| 인자 | 기본값 | 비고 |
|---|---|---|
| `--name` | (필수) | 실험명, 로그/체크포인트 폴더명 |
| `--data` | `inat21_mini` | **NIA29에서는 `NIA29_input` 지정** |
| `--data_dir` | `datasets/inat21` | 데이터 루트 |
| `--save_dir` | `./logs` | 로그/체크포인트 저장 루트 |
| `--model_file` | `sk2res2net_dynamic_mlp` | `models/__init__.py`의 키 |
| `--model_name` | `sk2res2net101` | 모델 팩토리 함수명 |
| `--batch_size` | 512 | |
| `--warmup` | 2 | warmup epoch |
| `--start_lr` | 0.04 | SGD 초기 lr |
| `--stop_epoch` | 100 | 총 epoch (eval.py는 90) |
| `--num_workers` | 32 | DataLoader workers |
| `--image_only` | False | image-only 백본 사용 시 |
| `--metadata` | `geo_temporal` | 현재 dataset.py에서 미사용(항상 width/height) |
| `--pretrained` | False | 사전학습 로드 |
| `--resume` | `''` | 체크포인트 경로 또는 `best`/`latest` |
| `--evaluate` | False | 평가만 수행 |
| `--mlp_type` | `c` | a/b/c |
| `--mlp_d` | 256 | FusionModule 출력 차원 |
| `--mlp_h` | 64 | 중간 hidden |
| `--mlp_n` | 2 | RecursiveBlock 개수 |
| `--random_seed` | 37 | |
| `--fold` | 1 | 체크포인트 파일명에 사용 |

`args.mlp_cin = 4` 가 코드 내에서 하드코딩(train.py, eval.py 모두).

### 5.2 학습 로직

- Optimizer: SGD(momentum=0.9, weight_decay=1e-4).
- Loss: `LabelSmoothingLoss(smoothing=0.1)` + **mixup(alpha=0.4)** (이미지와 loc 모두 mixup 적용).
- LR: warmup(선형) → cosine decay (`utils.adjust_learning_rate`).
- AMP(`torch.cuda.amp`) 혼합정밀도 사용.
- `torch.nn.DataParallel` 다중 GPU.
- 매 epoch마다 `latest.pth` 저장, best acc1 갱신 시 `best.pth` 저장(`utils.save_checkpoint`).
- 체크포인트 경로: `logs/{data}/{name}/fold{fold}_{best|latest}.pth`.

### 5.3 검증(`validate`)

`train.py` 내 `validate`는 단순 acc1/acc5/loss만 계산. 상세 평가는 `eval.py` 사용.

### 5.4 실행 예시

```bash
# 학습 (SK2Res2Net + Dynamic MLP)
python train.py \
  --name sk2_dynamic_mlp \
  --data NIA29_input \
  --data_dir path/to/data \
  --model_file sk2res2net_dynamic_mlp \
  --model_name sk2res2net101 \
  --pretrained --batch_size 512 --start_lr 0.04

# train.py로 평가만
python train.py \
  --name sk2_dynamic_mlp \
  --data NIA29_input \
  --data_dir path/to/data \
  --model_file sk2res2net_dynamic_mlp \
  --model_name sk2res2net101 \
  --resume path/to/checkpoint.pth --evaluate
```

---

## 6. 평가 스크립트 (`eval.py`)

- `train.py`와 비슷한 인자 + 상세 로깅.
- `--save_dir`(기본 `./logs`) 아래에 `{data}_log_{timestamp}/` 디렉토리 생성 후 `{name}.log`, `results_{timestamp}.txt` 저장.
- `validate()`: 배치별로 acc1/acc5/loss 계산, 클래스별 TP/FP/TN/FN confusion matrix 누적, 이미지별 예측/정답 로그 출력.
- `cudnn.deterministic=True`, `benchmark=False` (재현성 중시).
- **주의**: 파일 내에 두 번째 버전이 triple-quoted 문자열(`"""..."""`)로 주석처리되어 있음. 활성화된 코드는 위쪽(1~361줄).
- 틀린 예측 이미지 저장 코드는 주석 처리되어 있음(필요시 해제).

```bash
python eval.py \
  --name sk2_dynamic_mlp \
  --data NIA29_input \
  --data_dir path/to/data \
  --model_file sk2res2net_dynamic_mlp \
  --model_name sk2res2net101 \
  --resume path/to/checkpoint.pth --evaluate
```

---

## 7. 유틸리티 (`utils.py`)

- `mixup(x, y, alpha=0.4)`: 입력과 타겟을 섞음, lam과 index 반환.
- `LabelSmoothingLoss`: label smoothing CE.
- `accuracy(output, target, topk=(1,5))`: Top-k 정확도. 반환값이 `np.array` (원본 주석 처리된 버전과 다름).
- `adjust_learning_rate`: warmup + cosine.
- `save_checkpoint`: `{path_log}/fold{fold}_{save_name}.pth` 저장(epoch, model, optimizer).
- `create_logging`: logger 설정.
- `get_flops`: thop 기반 FLOPs/params 측정(옵션).

---

## 8. 환경 요구사항

### 8.1 가상환경 (로컬 Windows)

프로젝트 루트에 `.venv/` 가상환경을 사용한다.

```bash
# 생성
python -m venv .venv

# 활성화 (PowerShell)
.\.venv\Scripts\Activate.ps1
# 활성화 (CMD)
.\.venv\Scripts\activate.bat

# 패키지 설치 (RTX 50xx Blackwell — CUDA 12.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy pillow tqdm scikit-learn matplotlib thop
```

### 8.2 패키지 목록 (`requirements.txt`)

Linux 서버 등 다른 환경으로 소스만 가져갈 때 사용. 직접 의존성만 명시되어 있으며,
전체 목록과 CUDA 인덱스 URL 안내는 `requirements.txt` 파일 헤더 주석 참조.

```bash
# Linux 서버 (CUDA 12.8 — RTX 50xx)
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu128

# Linux 서버 (CUDA 12.1 — RTX 30xx/40xx)
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121

# CPU 전용
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
```

### 8.3 검증된 환경 (로컬 Windows)

- Python 3.14.6, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU (Blackwell, capability 12.0)
- numpy 2.5.2, pillow 12.3.0, tqdm 4.70.0, scikit-learn 1.9.0, matplotlib 3.11.1, thop 0.1.1

> 원본 README 기준(참고용): Python 3.6, PyTorch 1.7.1+cu101, torchvision 0.8.2.
> 본 fork는 최신 PyTorch 2.x에서 동작하도록 검증됨.

Windows 환경에서 실행 중이므로 `num_workers`를 0~4 정도로 낮추는 것을 권장(기본 32는 리눅스 다중 GPU 가정).

---

## 9. 자체 데이터로 학습시키기 위한 체크리스트

새 데이터를 본 프레임워크로 학습하려면 다음을 확인/수정:

1. **데이터 레이아웃 맞추기**: `<data_dir>/train/`, `<data_dir>/train_ann/`, `<data_dir>/tta/`, `<data_dir>/tta_ann/` 구조로 배치. 이미지는 `.jpg`, 어노테이션은 이미지 파일명과 동일(확장자만 `.json`). `prepare_data.py --category inbound|outbound --clean`으로 자동 생성 가능. **소스 경로는 세미콜론(`;`)으로 여러 개 지정 가능** (`--src "samples;D:/more_samples;E:/extra"`). 동일 상품 폴더명이 여러 소스에 중복될 경우 `(src, folder)` 키로 중복 제거됨.
2. **어노테이션 스키마**: `annotations[0].attributes`에 `KAN_code`, `width`, `height` 필드 포함. (다른 메타데이터를 쓰려면 `dataset.py`의 `__getitem__` 수정.)
3. **클래스 리스트**: `dataset.py`의 `kan_code_list`와 `eval.py`의 `number_labels`를 새 클래스 목록으로 교체. 두 리스트가 동일해야 함.
4. **num_classes**: `dataset.py`의 `load_train_dataset`/`load_val_dataset`에서 `args.num_classes` 값을 클래스 수에 맞게 수정 (또는 `args.data` 분기 추가).
5. **`--data` 이름**: 새 데이터 이름을 쓰려면 `dataset.py`의 `if args.data == 'NIA29_input'` 분기를 추가하거나 교체.
6. **사전학습 경로**: `sk2res2net_dynamic_mlp.py`의 `model_path`는 `checkpoints/sk2res2net101_epoch_300.pth`로 설정됨(2026-09 수정). 해당 파일이 없으면 `checkpoints/README.md` 링크에서 다운로드.
7. **메타데이터 정규화 로직**: `dataset.py` 106~115줄의 중복 정규화 버그 확인. 기존 체크포인트를 이어 쓸 경우 수정 금지(재학습 필요).
8. **mlp_cin**: 메타데이터 차원이 바뀌면 `args.mlp_cin`과 `encode_loc_time` 출력 차원을 일치시킬 것. 현재는 입력 2차원 → sin/cos → 4차원.
9. **Windows 설정**: `--num_workers` 축소, 경로에 한글/공백 주의.
10. **검증 디렉토리**: `val/`이 아니라 `tta/`를 사용함을 명심.

---

## 10. 알려진 이슈 / 코드 특이사항

- `dataset.py`의 width/height 정규화가 두 번 적용됨(106~111줄과 113~115줄). 의도된 것인지 버그인지 불명. 기존 체크포인트와 호환성 주의.
- `sk2res2net_dynamic_mlp.py`의 사전학습 경로가 `checkpoints/sk2res2net101_epoch_300.pth`로 수정됨(2026-09). 파일이 없으면 `checkpoints/README.md`에서 다운로드. 다른 체크포인트를 쓰려면 `model_path` 수정 필요.
- `dataset_oryginal.py`는 원본 참고용이며 실행에 사용되지 않음.
- `eval.py`는 한 파일에 두 버전이 있고 두 번째는 문자열 주석처리됨.
- `train.py`의 `validate`와 `eval.py`의 `validate`가 다름(후자가 상세).
- `args.metadata` 인자는 dataset.py에서 실제로 참조되지 않음(주석 처리됨). 항상 width/height 사용.
- `accuracy()` 반환 타입이 `np.array` (원본은 주석처리된 버전 사용).
- 클래스 수 관련 주석에 "51"이라는 오래된 값이 남아있으나 실제 리스트는 107개, `num_classes=107`.

---

## 11. 빠른 참조: 파일별 핵심 라인

- `train.py`
  - 인자 정의: 17~50줄
  - 메인 루프/체크포인트: 89~133줄
  - `train()`: 136~209줄 (mixup/AMP/LR)
  - `validate()`: 212~254줄
- `eval.py`
  - 활성 코드: 1~361줄
  - `validate()`(상세): 94~233줄
  - `main()`: 235~361줄
- `prepare_data.py`
  - `KAN_CODE_LIST` 정의: 65~77줄
  - `scan_products` (KAN_CODE_LIST 필터 포함): 97~182줄
  - `split_products`: 211~253줄
  - `copy_files`: 256~304줄
  - `main()` / `--src` 인자(세미콜론 다중 지정): 345줄~
- `dataset.py`
  - `kan_code_list`: 72~84줄
  - 리스트 외 코드 필터링: 105~116줄
  - `NiaDataset.__getitem__`: 122~204줄
  - `encode_loc_time`: 207~221줄
  - `load_train_dataset`/`load_val_dataset`: 223~281줄
- `utils.py`
  - `mixup`: 27~45줄
  - `LabelSmoothingLoss`: 48~62줄
  - `accuracy`: 116~131줄
  - `adjust_learning_rate`: 143~153줄
- `models/dynamic_mlp.py`
  - `FCNet`: 25~54줄
  - `Dynamic_MLP_C`: 124~155줄
  - `FusionModule`: 175~216줄
  - `get_dynamic_mlp`: 219~224줄
- `models/sk2res2net_dynamic_mlp.py`
  - `SK2Res2Net.forward`: 1031~1052줄
  - `sk2res2net101` 팩토리: 1055~1073줄 (사전학습 경로 하드코딩 1070줄)
- `models/resnet_dynamic_mlp.py`
  - `ResNet.forward`: 237~256줄
  - `resnet50`/`resnet101`: 270~303줄
