#!/usr/bin/env python3
"""
prepare_data.py
================
samples/ 디렉토리의 원천데이터(이미지)와 라벨링데이터(JSON)를
dataset.py 가 기대하는 datasets/ 구조로 재분포합니다.

samples 구조 (입력):
  samples/
    01.원천데이터/          ← 이미지 (.jpg)
      01_입고물품/01_가공식품/01_조미료/01010110_8809251219122/
        01010110_8809251219122_1_1.jpg
        01010110_8809251219122_1_2.jpg
        ...
      ...
    02.라벨링데이터/        ← 어노테이션 (.json, 동일 폴더 구조)
      01_입고물품/01_가공식품/01_조미료/01010110_8809251219122/
        01010110_8809251219122_1_1.json
        ...

datasets 구조 (출력):
  datasets/
    train/        ← 학습 이미지 (평면 구조, 파일명만 사용)
    train_ann/    ← 학습 어노테이션 (평면 구조)
    tta/          ← 검증 이미지  (dataset.py 가 tta/ 를 val 로 사용)
    tta_ann/      ← 검증 어노테이션

물품 카테고리:
  - 입고물품(01_입고물품)과 출고물품(02_출고물품)은 서로 다른 모델의 학습 대상이므로
    --category 파라미터로 선택. 기본값: inbound (입고물품).

분할 기준:
  - 상품 단위(leaf 폴더 = {KAN_code}_{product_id})로 train / val 분할.
    → 동일 상품의 여러 이미지가 train과 val에 섞이지 않음 (data leakage 방지).
  - KAN_code(앞 4자리) 기준 stratified 분할.
  - 상품 폴더가 1개뿐인 KAN_code는 train에만 배치.

사용법:
  python prepare_data.py                                    # 기본: 입고물품, samples → datasets, 8:2
  python prepare_data.py --category outbound                # 출고물품
  python prepare_data.py --src samples --dst datasets --val_ratio 0.2
  python prepare_data.py --clean                            # 기존 datasets/ 하위 4폴더 삭제 후 생성
  python prepare_data.py --dry_run                          # 복사 없이 분할 결과만 출력
"""

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# Windows 콘솔(cp949)에서 한글/특수문자 출력 보장
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ── dataset.py 와 동일한 KAN_code 리스트 (107개) ──────────────────────────
KAN_CODE_LIST = [
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

# dataset.py 가 검증에 사용하는 폴더명 (val 이 아니라 tta)
TRAIN_DIR = 'train'
TRAIN_ANN_DIR = 'train_ann'
VAL_DIR = 'tta'
VAL_ANN_DIR = 'tta_ann'

# samples 내 원천/라벨링 폴더명
RAW_DIR = '01.원천데이터'
ANN_DIR = '02.라벨링데이터'

# 물품 카테고리 별칭 → 실제 폴더명 매핑
CATEGORY_MAP = {
    'inbound': '01_입고물품',    # 입고물품
    'outbound': '02_출고물품',   # 출고물품
}
CATEGORY_DEFAULT = 'inbound'


def scan_products(src: Path, category: str = CATEGORY_DEFAULT):
    """
    samples 디렉토리 중 지정한 카테고리(입고/출고) 하위만 스캔하여
    상품(leaf 폴더) 단위로 이미지·어노테이션을 수집.

    Args:
        src: samples 루트
        category: 'inbound' | 'outbound' | 또는 실제 폴더명(예: '01_입고물품')

    Returns:
        products: list of dict, 각 상품:
          {
            'kan_code': '0101',
            'product_folder': '01010110_8809251219122',
            'images': [Path, ...],
            'annotations': [Path, ...],
          }
    """
    # 별칭 → 실제 폴더명 해석 (이미 폴더명 형태면 그대로 사용)
    category_folder = CATEGORY_MAP.get(category, category)

    raw_root = src / RAW_DIR
    ann_root = src / ANN_DIR

    if not raw_root.exists():
        print(f'[오류] 원천데이터 디렉토리가 없습니다: {raw_root}')
        sys.exit(1)
    if not ann_root.exists():
        print(f'[오류] 라벨링데이터 디렉토리가 없습니다: {ann_root}')
        sys.exit(1)

    # 카테고리 하위 폴더로 제한
    raw_cat = raw_root / category_folder
    ann_cat = ann_root / category_folder
    if not raw_cat.exists():
        print(f'[오류] 카테고리 폴더가 원천데이터에 없습니다: {raw_cat}')
        print(f'  사용 가능한 카테고리: {list(CATEGORY_MAP.keys())} '
              f'또는 실제 폴더명 {list(CATEGORY_MAP.values())}')
        sys.exit(1)
    if not ann_cat.exists():
        print(f'[오류] 카테고리 폴더가 라벨링데이터에 없습니다: {ann_cat}')
        sys.exit(1)

    # 이미지가 들어있는 leaf 폴더(=상품 폴더) 수집
    # (카테고리 폴더 하위만 탐색)
    product_folders = []
    for dirpath, _, filenames in os.walk(raw_cat):
        jpgs = [f for f in filenames if f.lower().endswith('.jpg')]
        if jpgs:
            product_folders.append(Path(dirpath))

    products = []
    for pf in product_folders:
        folder_name = pf.name  # 예: 01010110_8809251219122
        kan_code = folder_name[:4]

        # 라벨링 데이터에서 대응하는 폴더 찾기 (동일한 폴더명)
        # 카테고리 하위의 상대경로를 ann_cat 에 붙임
        rel = pf.relative_to(raw_cat)
        ann_pf = ann_cat / rel

        images = sorted(pf.glob('*.jpg'))
        annotations = sorted(ann_pf.glob('*.json')) if ann_pf.exists() else []

        products.append({
            'kan_code': kan_code,
            'product_folder': folder_name,
            'images': images,
            'annotations': annotations,
        })

    # KAN_CODE_LIST(107개 클래스)에 없는 상품은 제외.
    # dataset.py가 리스트 외 코드를 만나면 ValueError로 학습이 중단되므로
    # 여기서 미리 걸러낸다.
    n_all = len(products)
    skipped_codes = sorted({p['kan_code'] for p in products
                            if p['kan_code'] not in KAN_CODE_LIST})
    products = [p for p in products if p['kan_code'] in KAN_CODE_LIST]
    skipped = n_all - len(products)
    if skipped:
        print(f'[필터] KAN_CODE_LIST에 없는 상품 {skipped}개 제외 '
              f'(코드: {", ".join(skipped_codes)})')

    return products


def validate_annotation(ann_path: Path, kan_code: str) -> bool:
    """
    어노테이션 JSON이 dataset.py가 요구하는 필드를 갖추고 있는지 검증한다.

    왜 검증이 필요한가:
      - dataset.py는 KAN_code, width, height 필드가 없으면 런타임 오류를 발생시킨다
      - 학습 도중 오류가 발생하면 전체 학습이 중단되므로 사전에 필터링하는 것이 안전하다
      - KAN_code 앞 4자리가 폴더명과 일치하는지 확인하여 데이터 무결성을 검증한다
    """
    try:
        with open(ann_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        anns = data.get('annotations', [])
        if not anns:
            return False
        attrs = anns[0].get('attributes', {})
        if 'KAN_code' not in attrs:
            return False
        if 'width' not in attrs or 'height' not in attrs:
            return False
        # KAN_code 앞 4자리가 폴더명과 일치하는지 확인
        if attrs['KAN_code'][:4] != kan_code:
            return False
        return True
    except (json.JSONDecodeError, KeyError, IndexError):
        return False


def split_products(products, val_ratio=0.2, seed=37):
    """
    KAN_code 기준 stratified 분할을 수행한다.

    왜 상품(leaf 폴더) 단위로 분할하는가:
      - 동일 상품의 여러 이미지(여러 각도/카메라)가 train과 val에 섞이면
        데이터 누수(data leakage)가 발생하여 검증 성능이 과대평가된다
      - 상품 단위 분할은 실제 서비스 환경에서 새 상품을 분류하는 상황을
        더 잘 시뮬레이션한다

    왜 KAN_code 기준 stratified 분할인가:
      - 단순 무작위 분할 시 샘플이 적은 클래스는 val에 아예 빠질 수 있다
      - stratified 분할은 각 클래스가 train/val 양쪽에 고르게 분포하도록 보장한다

    왜 상품이 1개뿐인 클래스는 train에만 배치하는가:
      - val에 넣으면 train에 해당 클래스가 아예 없어져서 모델이 학습할 수 없다
      - 차선책으로 train에 배치하여 모델이 해당 클래스를 학습하도록 한다
    """
    # KAN_code별로 상품 그룹화
    kan_groups = defaultdict(list)
    for p in products:
        kan_groups[p['kan_code']].append(p)

    train_products = []
    val_products = []

    rng = random.Random(seed)

    for kan_code in sorted(kan_groups.keys()):
        group = kan_groups[kan_code]
        if len(group) == 1:
            # 상품이 1개뿐이면 train에 배치
            train_products.extend(group)
            continue

        # 섞은 후 val_ratio 만큼 val 로
        shuffled = list(group)
        rng.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * val_ratio))
        val_products.extend(shuffled[:n_val])
        train_products.extend(shuffled[n_val:])

    return train_products, val_products


def copy_files(products, img_dst: Path, ann_dst: Path, dry_run=False):
    """
    이미지와 어노테이션을 평면 구조(하위 폴더 없이)로 복사한다.

    왜 평면 구조로 복사하는가:
      - dataset.py는 os.walk로 모든 하위 폴더를 탐색하지만,
        평면 구조가 파일 검색과 디버깅에 더 효율적이다
      - 파일명이 고유하므로(예: 01010110_8809251219122_1_1.jpg) 폴더 구조가 불필요하다
    """
    img_dst.mkdir(parents=True, exist_ok=True)
    ann_dst.mkdir(parents=True, exist_ok=True)

    n_img = 0
    n_ann = 0
    n_skip = 0
    n_invalid = 0

    for p in products:
        kan_code = p['kan_code']

        # 이미지 복사
        for img in p['images']:
            dst_img = img_dst / img.name
            if dst_img.exists():
                n_skip += 1
                continue
            if not dry_run:
                shutil.copy2(img, dst_img)
            n_img += 1

        # 어노테이션 복사 (검증 포함)
        for ann in p['annotations']:
            dst_ann = ann_dst / ann.name
            if dst_ann.exists():
                n_skip += 1
                continue

            # 어노테이션 검증
            if not validate_annotation(ann, kan_code):
                print(f'  [경고] 어노테이션 검증 실패 (건너뜀): {ann.name} '
                      f'(KAN_code={kan_code})')
                n_invalid += 1
                continue

            if not dry_run:
                shutil.copy2(ann, dst_ann)
            n_ann += 1

    return n_img, n_ann, n_skip, n_invalid


def print_stats(train_products, val_products, n_img_train, n_ann_train,
                n_img_val, n_ann_val, n_invalid):
    """분할 통계 출력."""
    train_kans = set(p['kan_code'] for p in train_products)
    val_kans = set(p['kan_code'] for p in val_products)
    only_train = train_kans - val_kans
    only_val = val_kans - train_kans

    print()
    print('=' * 60)
    print('데이터 분할 결과')
    print('=' * 60)
    print(f'  상품 폴더 수  — train: {len(train_products)}, val: {len(val_products)}')
    print(f'  이미지 수     — train: {n_img_train}, val: {n_img_val}')
    print(f'  어노테이션 수 — train: {n_ann_train}, val: {n_ann_val}')
    print(f'  검증 실패 어노테이션: {n_invalid}')
    print()
    print(f'  KAN_code 수 — train: {len(train_kans)}, val: {len(val_kans)}, '
          f'전체: {len(train_kans | val_kans)}')
    if only_train:
        print(f'  train에만 있는 KAN_code ({len(only_train)}개): '
              f'{sorted(only_train)}')
    if only_val:
        print(f'  [주의] val에만 있는 KAN_code ({len(only_val)}개): '
              f'{sorted(only_val)}')
    print('=' * 60)


def clean_datasets(dst: Path):
    """datasets/ 하위의 train, train_ann, tta, tta_ann 삭제."""
    subdirs = [TRAIN_DIR, TRAIN_ANN_DIR, VAL_DIR, VAL_ANN_DIR]
    for sd in subdirs:
        target = dst / sd
        if target.exists():
            print(f'  삭제: {target}')
            shutil.rmtree(target)


def main():
    parser = argparse.ArgumentParser(
        description='samples/ 데이터를 dataset.py용 datasets/ 구조로 재분포')
    parser.add_argument('--src', default='samples', type=str,
                        help='원본 데이터 루트. 세미콜론(;)으로 여러 경로 지정 가능 '
                             '(예: "samples;D:/more_samples") (기본: samples)')
    parser.add_argument('--dst', default='datasets', type=str,
                        help='출력 데이터 루트 (기본: datasets)')
    parser.add_argument('--category', default=CATEGORY_DEFAULT, type=str,
                        help='물품 카테고리: inbound(입고물품) | outbound(출고물품) '
                             '또는 실제 폴더명 (기본: inbound)')
    parser.add_argument('--val_ratio', default=0.2, type=float,
                        help='검증셋 비율 (기본: 0.2)')
    parser.add_argument('--seed', default=37, type=int,
                        help='랜덤 시드 (기본: 37)')
    parser.add_argument('--clean', action='store_true',
                        help='기존 datasets/ 하위 4개 폴더 삭제 후 생성')
    parser.add_argument('--dry_run', action='store_true',
                        help='복사 없이 분할 결과만 출력')
    args = parser.parse_args()

    # 세미콜론으로 구분된 여러 소스 경로 파싱
    src_paths = [Path(s).resolve() for s in args.src.split(';') if s.strip()]
    if not src_paths:
        print('[오류] --src 경로가 비어 있습니다.')
        sys.exit(1)
    dst = Path(args.dst).resolve()

    # 카테고리 별칭 해석
    category_folder = CATEGORY_MAP.get(args.category, args.category)

    print(f'원본: {"; ".join(str(p) for p in src_paths)}')
    print(f'출력: {dst}')
    print(f'카테고리: {args.category} → {category_folder}')
    print(f'검증 비율: {args.val_ratio} (시드: {args.seed})')
    print(f'모드: {"dry-run" if args.dry_run else "실제 복사"}')
    print()

    # 1. 스캔 (여러 소스를 순회하며 병합)
    print(f'[1/4] samples 스캔 중... (소스 {len(src_paths)}개)')
    products = []
    seen_folders = set()  # (src, product_folder) 중복 방지
    for src in src_paths:
        print(f'  - 스캔: {src}')
        src_products = scan_products(src, category=args.category)
        # 동일 상품 폴더명이 여러 소스에 중복될 수 있으므로 (src, folder) 키로 dedupe
        for p in src_products:
            key = (str(src), p['product_folder'])
            if key in seen_folders:
                continue
            seen_folders.add(key)
            products.append(p)
        print(f'    상품 폴더: {len(src_products)}개 (누적: {len(products)}개)')
    print(f'  전체 상품 폴더: {len(products)}개')
    total_img = sum(len(p['images']) for p in products)
    total_ann = sum(len(p['annotations']) for p in products)
    print(f'  이미지: {total_img}개, 어노테이션: {total_ann}개')

    # KAN_code 분포
    kan_counts = defaultdict(int)
    for p in products:
        kan_counts[p['kan_code']] += 1
    print(f'  KAN_code 종류: {len(kan_counts)}개')

    # 2. 분할
    print()
    print('[2/4] train / val 분할 중...')
    train_products, val_products = split_products(
        products, val_ratio=args.val_ratio, seed=args.seed)
    print(f'  train 상품: {len(train_products)}개, val 상품: {len(val_products)}개')

    # 3. 클린 (옵션)
    if args.clean and not args.dry_run:
        print()
        print('[3/4] 기존 데이터 삭제 중...')
        clean_datasets(dst)

    # 4. 복사
    print()
    print('[4/4] 파일 복사 중...')
    train_img_dst = dst / TRAIN_DIR
    train_ann_dst = dst / TRAIN_ANN_DIR
    val_img_dst = dst / VAL_DIR
    val_ann_dst = dst / VAL_ANN_DIR

    print(f'  → train 이미지 → {train_img_dst}')
    n_img_train, n_ann_train, skip_t, inv_t = copy_files(
        train_products, train_img_dst, train_ann_dst, dry_run=args.dry_run)
    print(f'  → val 이미지   → {val_img_dst}')
    n_img_val, n_ann_val, skip_v, inv_v = copy_files(
        val_products, val_img_dst, val_ann_dst, dry_run=args.dry_run)

    # 5. 통계
    print_stats(train_products, val_products,
                n_img_train, n_ann_train, n_img_val, n_ann_val,
                inv_t + inv_v)

    # 6. 다음 단계 안내
    print()
    print('다음 단계:')
    print(f'  python train.py --name sk2_dynamic_mlp --data NIA29_input '
          f'--data_dir "{dst}" \\')
    print(f'    --model_file sk2res2net_dynamic_mlp --model_name sk2res2net101 '
          f'--batch_size 32 --num_workers 0')
    print()
    print(f'  ※ 카테고리: {args.category} ({category_folder})')
    print(f'  ※ 다른 카테고리를 준비하려면: '
          f'python prepare_data.py --category {"outbound" if args.category == "inbound" else "inbound"} --clean')


if __name__ == '__main__':
    main()
