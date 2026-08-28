#!/usr/bin/env python3
"""
DynamicMLP 학습 엔트리포인트.

본 스크립트는 NIA29 물류 이미지 데이터로 Dynamic MLP 모델을 학습한다.
학습 과정:
  1. 데이터 로더 구성 (이미지 + 메타데이터)
  2. 모델 생성 (백본 + Dynamic MLP)
  3. mixup 데이터 증강 + Label Smoothing 손실로 학습
  4. 매 epoch마다 검증 및 체크포인트 저장

CPU/GPU를 자동으로 감지하여 동작한다.
"""
import argparse
import datetime
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim

import dataset
import models
import utils


def main():
    """
    학습 파이프라인의 메인 진입점. 인자 파싱, 환경 설정, 학습 루프를 관리한다.
    """
    # ── 명령줄 인자 정의 ──────────────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, type=str,
                        help='실험명 (로그/체크포인트 폴더명으로 사용)')
    parser.add_argument('--data', default='inat21_mini', type=str,
                        help='데이터셋 이름 (NIA29_input 사용)')
    parser.add_argument('--data_dir', default='datasets/inat21', type=str,
                        help='데이터 루트 디렉토리')
    parser.add_argument('--save_dir', default='./logs', type=str,
                        help='로그 및 체크포인트 저장 디렉토리')
    parser.add_argument('--model_file', default='sk2res2net_dynamic_mlp', type=str,
                        help='models/ 내의 모델 파일명')
    parser.add_argument('--model_name', default='sk2res2net101', type=str,
                        help='모델 팩토리 함수명')
    parser.add_argument('--fold', default=1, type=int,
                        help='교차검증 fold 번호 (체크포인트 파일명에 사용)')
    parser.add_argument('--random_seed', default=37, type=int,
                        help='재현성을 위한 랜덤 시드')

    # 학습 하이퍼파라미터
    parser.add_argument('--batch_size', default=512, type=int,
                        help='미니배치 크기 (GPU 메모리에 따라 조절)')
    parser.add_argument('--warmup', default=2, type=int,
                        help='warmup epoch 수 (초기 학습률을 선형적으로 증가)')
    parser.add_argument('--start_lr', default=0.04, type=float,
                        help='초기 학습률 (warmup 후 도달할 최대 학습률)')
    parser.add_argument('--stop_epoch', default=100, type=int,
                        help='총 학습 epoch 수')
    parser.add_argument('--num_workers', default=32, type=int,
                        help='DataLoader 워커 수 (Windows에서는 0 권장)')

    # 데이터 옵션
    parser.add_argument('--tencrop', action='store_true', default=False,
                        help='검증 시 TenCrop 증강 사용 여부')
    parser.add_argument('--image_only', action='store_true', default=False,
                        help='메타데이터 없이 이미지만 사용 (image-only 백본 필요)')
    parser.add_argument('--metadata', default='geo_temporal', type=str,
                        help='메타데이터 타입 (현재 dataset.py에서 미사용, 항상 width/height)')

    # 모델 옵션
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='사전학습 가중치 로드 여부')
    parser.add_argument('--resume', default='', type=str,
                        help='체크포인트 경로 (best/latest 키워드 가능)')
    parser.add_argument('--evaluate', action='store_true',
                        help='학습 없이 평가만 수행')

    # Dynamic MLP 구조 하이퍼파라미터
    parser.add_argument('--mlp_type', default='c', type=str,
                        help='Dynamic MLP 변형: a(단순) | b(중간) | c(복합, 기본값)')
    parser.add_argument('--mlp_d', default=256, type=int,
                        help='FusionModule 출력 차원 (백본 feature와 잔차 연결을 위해 2048→256→2048)')
    parser.add_argument('--mlp_h', default=64, type=int,
                        help='RecursiveBlock 중간 hidden 차원')
    parser.add_argument('--mlp_n', default=2, type=int,
                        help='RecursiveBlock 반복 횟수 (Dynamic MLP 깊이)')

    args = parser.parse_args()

    # mlp_cin: Dynamic MLP에 입력되는 메타데이터 벡터의 차원 수.
    # encode_loc_time이 [width, height] (2차원)을 sin/cos로 4차원으로 확장하므로 4로 고정.
    args.mlp_cin = 4

    # ── 디바이스 설정 ──────────────────────────────────────────────────
    # CUDA가 사용 가능하면 GPU를, 그렇지 않으면 CPU를 사용한다.
    # 이를 통해 동일한 코드로 GPU 서버와 CPU 환경에서 모두 실행할 수 있다.
    args.cuda = torch.cuda.is_available()
    args.device = torch.device('cuda' if args.cuda else 'cpu')
    args.use_amp = args.cuda  # AMP(혼합정밀도)는 GPU에서만 의미가 있다

    # ── 랜덤 시드 고정 ──────────────────────────────────────────────────
    # 실험 재현성을 위해 모든 난수 생성기를 고정한다.
    # 단, cudnn.benchmark=True는 속도를 위해 결정성을 포기한다 (GPU만).
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        # benchmark=True: 입력 크기가 일정하면 cuDNN이 최적 알고리즘을 자동 선택.
        # deterministic=False: 재현성보다 속도를 우선 (학습 시 일반적 선택).
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # 사용 가능한 GPU 수 (DataParallel 다중 GPU 학습에 사용)
    args.nprocs = torch.cuda.device_count() if args.cuda else 0

    # ── 로거 설정 ──────────────────────────────────────────────────────
    # 로그와 체크포인트는 {save_dir}/{data}/{name}/ 아래에 저장된다.
    creat_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    args.path_log = os.path.join(args.save_dir, f'{args.data}', f'{args.name}')
    os.makedirs(args.path_log, exist_ok=True)
    logger = utils.create_logging(os.path.join(args.path_log, '%s_train.log' % creat_time))

    # ── 데이터 로더 구성 ────────────────────────────────────────────────
    train_loader = dataset.load_train_dataset(args)
    val_loader = dataset.load_val_dataset(args)

    # 현재 사용 중인 디바이스 정보를 로그에 기록.
    if args.cuda:
        logger.info('Using CUDA ({} GPU(s))'.format(args.nprocs))
    else:
        logger.info('CUDA not available - using CPU. (학습 속도가 느릴 수 있습니다)')

    # 모든 인자를 로그에 기록하여 나중에 어떤 설정으로 학습했는지 추적 가능.
    for param in sorted(vars(args).keys()):
        logger.info('--{0} {1}'.format(param, vars(args)[param]))

    # ── 모델 생성 ──────────────────────────────────────────────────────
    # models/__init__.py를 통해 model_file(예: sk2res2net_dynamic_mlp)의
    # model_name(예: sk2res2net101) 함수를 동적 호출하여 모델을 생성한다.
    net = models.__dict__[args.model_file].__dict__[args.model_name](logger, args)
    net = net.to(args.device)

    # GPU가 2장 이상이면 DataParallel로 다중 GPU 학습을 수행한다.
    # 단일 GPU이면 DataParallel을 적용하지 않아 오버헤드를 줄인다.
    if args.nprocs > 1:
        net = torch.nn.DataParallel(net)

    # ── 손실 함수 ──────────────────────────────────────────────────────
    # Label Smoothing: 정답 클래스에 0.9, 나머지에 0.1/(N-1)을 할당하여
    # 모델이 과도하게 확신하는 것을 방지하고 일반화 성능을 높인다.
    criterion = utils.LabelSmoothingLoss(classes=args.num_classes, smoothing=0.1).to(args.device)

    # ── 옵티마이저 ──────────────────────────────────────────────────────
    # SGD with momentum: 대규모 분류 태스크에서 Adam보다 더 잘 일반화되는 경향이 있다.
    # weight_decay=1e-4: L2 정규화로 과적합을 억제한다.
    optimizer = optim.SGD(net.parameters(), lr=args.start_lr, momentum=0.9, weight_decay=1e-4)

    # ── 체크포인트 로드 (학습 재개 시) ──────────────────────────────────
    start_epoch = 1
    if args.resume:
        # 'best' 또는 'latest' 키워드를 실제 파일 경로로 변환.
        if args.resume in ['best', 'latest']:
            args.resume = os.path.join(args.path_log, 'fold%s_%s.pth' % (args.fold, args.resume))
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            # map_location: GPU에서 학습한 체크포인트를 CPU에서 로드할 수 있도록 함.
            state_dict = torch.load(args.resume, map_location=args.device)
            if 'model' in state_dict:
                # epoch, model, optimizer가 모두 저장된 체크포인트 (학습 재개용)
                start_epoch = state_dict['epoch'] + 1
                net.load_state_dict(state_dict['model'])
                optimizer.load_state_dict(state_dict['optimizer'])
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, state_dict['epoch']))
            else:
                # 가중치만 저장된 체크포인트 (추론용)
                net.load_state_dict(state_dict)
                logger.info("=> loaded checkpoint '{}'".format(args.resume))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # ── 평가 전용 모드 ──────────────────────────────────────────────────
    # --evaluate 플래그가 있으면 학습을 하지 않고 검증만 수행한다.
    if args.evaluate:
        epoch = start_epoch - 1
        acc1, acc5, outputs = validate(val_loader, net, criterion, epoch, logger, args)
        logger.info('\t'.join(outputs))
        logger.info('Exp path: %s' % args.path_log)
        return

    # ── 학습 루프 ──────────────────────────────────────────────────────
    best_acc1 = 0.0
    best_acc5 = 0.0
    args.time_sec_tot = 0.0
    args.start_epoch = start_epoch

    for epoch in range(start_epoch, args.stop_epoch + 1):
        # 1. 학습: 모델 가중치를 업데이트
        train(train_loader, net, criterion, optimizer, epoch, logger, args)

        # 2. latest 체크포인트 저장: 학습 중단 시 언제든 재개할 수 있도록 매 epoch 저장
        utils.save_checkpoint(epoch, net, optimizer, args, save_name='latest')

        # 3. 검증: 현재 epoch의 모델 성능을 평가
        acc1, acc5, outputs = validate(val_loader, net, criterion, epoch, logger, args)

        # 4. best 체크포인트 저장: Top-1 정확도가 최고일 때만 저장
        if acc1 > best_acc1:
            best_acc1 = acc1
            best_acc5 = acc5
            utils.save_checkpoint(epoch, net, optimizer, args, save_name='best')

        # 최고 기록을 로그에 기록 (Copypaste 라벨은 결과 파싱을 편하게 하기 위함)
        outputs += [
            'best_acc1: {:.4f}'.format(best_acc1), 'best_acc5: {:.4f}'.format(best_acc5),
            'Copypaste: {:.4f}, {:.4f}'.format(best_acc1, best_acc5)
        ]
        logger.info('\t'.join(outputs))
        logger.info('Exp path: %s' % args.path_log)


def train(train_loader, net, criterion, optimizer, epoch, logger, args):
    """
    한 epoch 동안 학습을 수행한다. mixup 증강과 AMP를 적용한다.
    """
    # 학습 모드 전환: Dropout/BN 등이 학습용 동작을 하도록 함.
    net.train()
    minibatch_count = len(train_loader)

    # GradScaler: AMP(자동 혼합정밀도)에서 그래디언트 스케일링을 수행.
    # float16 사용 시 그래디언트 언더플로우를 방지하기 위해 스케일을 조정한다.
    # CPU 환경에서는 enabled=False로 비활성화된다.
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    tstart = time.time()
    for i, (images, target, location, img_path) in enumerate(train_loader):
        # 학습률 스케줄링: warmup(선형 증가) 후 cosine decay(점진적 감소).
        # 매 미니배치마다 학습률을 조정하여 부드러운 감소를 구현한다.
        learning_rate = utils.adjust_learning_rate(optimizer, i, epoch, minibatch_count, args)

        # 데이터 로딩 시간 측정 (병목 진단용)
        tdata = time.time() - tstart

        # 데이터를 GPU/CPU로 이동. non_blocking=True는 pin_memory와 함께
        # 데이터 전송과 연산을 오버랩하여 속도를 높인다.
        images = images.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        location = location.to(args.device, non_blocking=True).float()

        # mixup: 두 이미지를 선형 결합하여 새로운 학습 샘플을 생성.
        # 정답 라벨도 두 클래스의 혼합으로 표현하여 모델이 클래스 간의
        # 보간된 특징을 학습하도록 유도한다. 과적합 방지에 효과적.
        images, target_a, target_b, lam, index = utils.mixup(images, target, alpha=0.4)

        # 메타데이터(location)도 동일한 비율(lam)로 mixup하여
        # 이미지와 메타데이터의 일관성을 유지한다.
        location = lam * location + (1 - lam) * location[index]

        # 순전파: AMP 컨텍스트 내에서 실행하여 자동으로 float16 연산을 적용.
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            if args.image_only:
                output = net(images)
            else:
                # Dynamic MLP 모델: 이미지와 메타데이터를 함께 입력.
                output = net(images, location)

            # mixup 손실: 두 타겟에 대한 손실을 lam 비율로 가중 평균.
            loss = lam * criterion(output, target_a) + (1 - lam) * criterion(output, target_b)

        # 정확도 측정: mixup된 두 타겟 각각에 대해 정확도를 계산 후 가중 평균.
        acc1_a, acc5_a = utils.accuracy(output, target_a, topk=(1, 5))
        acc1_b, acc5_b = utils.accuracy(output, target_b, topk=(1, 5))
        acc1 = lam * acc1_a + (1 - lam) * acc1_b
        acc5 = lam * acc5_a + (1 - lam) * acc5_b

        # 역전파 및 가중치 업데이트.
        # AMP 사용 시 scaler를 통해 그래디언트를 스케일링한 후 역전파하고,
        # 옵티마이저 스텝 전에 스케일을 되돌린다.
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 경과 시간 측정 및 ETA 계산.
        tend = time.time()
        ttrain = tend - tstart
        tstart = tend

        args.time_sec_tot += ttrain
        time_sec_avg = args.time_sec_tot / ((epoch - args.start_epoch) * minibatch_count + i + 1)
        eta_sec = time_sec_avg * ((args.stop_epoch + 1 - epoch) * minibatch_count - i - 1)
        eta_str = str(datetime.timedelta(seconds=int(eta_sec)))

        # 학습 진행 상황 로그 (20배치마다 출력하여 로그 과부하 방지)
        outputs = [
            "e: {}/{},{}/{}".format(epoch, args.stop_epoch, i, minibatch_count),
            "{:.2f} mb/s".format(1. / ttrain),
            'eta: {}'.format(eta_str),
            'time: {:.3f}'.format(ttrain),
            'data_time: {:.3f}'.format(tdata),
            'lr: {:.4f}'.format(learning_rate),
            'acc1: {:.4f}'.format(acc1.item()),
            'acc5: {:.4f}'.format(acc5.item()),
            'loss: {:.4f}'.format(loss.item()),
        ]

        # 데이터 로딩이 전체 시간의 5% 이상이면 데이터 파이프라인 병목 의심.
        if tdata / ttrain > .05:
            outputs += [
                "dp/tot: {:.4f}".format(tdata / ttrain),
            ]

        if i % 20 == 0:
            logger.info('\t'.join(outputs))


def validate(val_loader, net, criterion, epoch, logger, args):
    """
    검증 데이터셋으로 모델을 평가하여 Top-1/Top-5 정확도와 loss를 반환한다.
    """
    logger.info('eval epoch {}'.format(epoch))

    # 평가 모드 전환: Dropout을 비활성화하고 BN은 고정 통계값을 사용.
    net.eval()

    acc1_sum = 0
    acc5_sum = 0
    loss = 0
    valdation_num = 0

    for i, (images, target, location, img_path) in enumerate(val_loader):
        images = images.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        location = location.to(args.device, non_blocking=True).float()

        # no_grad: 검증 시 그래디언트를 계산하지 않아 메모리와 시간을 절약.
        with torch.no_grad():
            if args.image_only:
                output = net(images)
            else:
                output = net(images, location)

        # 정확도를 배치 크기로 가중 평균하여 전체 정확도를 계산.
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        num = target.size(0)
        valdation_num += num
        acc1_sum += acc1.item() * num
        acc5_sum += acc5.item() * num
        loss += criterion(output, target).item()
        if i % 20 == 0:
            logger.info('iter {}/{}'.format(i, len(val_loader)))

    # 전체 검증셋에 대한 평균 지표 계산.
    loss = loss / len(val_loader)
    acc1 = acc1_sum / valdation_num
    acc5 = acc5_sum / valdation_num

    outputs = [
        "val e: {}".format(epoch),
        'acc1: {:.4f}'.format(acc1),
        'acc5: {:.4f}'.format(acc5),
        'loss: {:.4f}'.format(loss),
    ]

    return acc1, acc5, outputs


if __name__ == '__main__':
    main()
