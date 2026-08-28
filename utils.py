#!/usr/bin/env python3
"""
학습 유틸리티 모듈.

본 모듈은 학습에 사용되는 보조 기능들을 제공한다:
  - mixup 데이터 증강
  - Label Smoothing 손실 함수
  - Top-k 정확도 계산
  - 학습률 스케줄링 (warmup + cosine decay)
  - 체크포인트 저장
  - 로깅 설정
"""
import logging
import os

import numpy as np
import torch
import torch.nn as nn


def get_flops(model, logger=None, loc_cin=6, img_size=224, image_only=False):
    """
    모델의 FLOPs(연산량)와 파라미터 수를 계산하여 출력한다.
    thop 라이브러리가 필요하며, 모델 규모 비교 시 사용한다.
    """
    from thop import clever_format, profile
    bs = 2
    img = torch.randn(bs, 3, img_size, img_size)
    loc = torch.randn(bs, loc_cin)
    if image_only:
        flops, params = profile(model, inputs=(img, ))
    else:
        flops, params = profile(model, inputs=(img, loc))
    flops = flops / bs
    flops, params = clever_format([flops, params], "%.3f")

    if logger is not None:
        logger.info('{:<30}  {:<8}'.format('Computational complexity: ', flops))
        logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params))


def mixup(x, y, alpha=0.4):
    """
    mixup 데이터 증강: 두 샘플을 선형 결합하여 새로운 학습 샘플을 생성한다.

    논문: "mixup: Beyond Empirical Risk Minimization" (ICLR 2018)
    https://arxiv.org/pdf/1710.09412.pdf

    왜 사용하는가:
      - 클래스 간의 보간된 샘플을 생성하여 결정 경계를 부드럽게 만든다
      - 과적합을 방지하고 일반화 성능을 향상시킨다
      - 라벨 노이즈에 대한 robustness를 높인다

    Args:
        x: 입력 이미지 배치 [B, C, H, W]
        y: 타겟 라벨 배치 [B]
        alpha: Beta 분포 파라미터 (클수록 더 강한 mixup)

    Returns:
        mixed_x: mixup된 이미지
        y: 원본 타겟 (target_a)
        y[index]: 섞인 타겟 (target_b)
        lam: 혼합 비율 (1.0이면 mixup 없음)
        index: 섞인 샘플의 인덱스
    """
    if alpha > 0:
        # Beta(alpha, alpha) 분포에서 혼합 비율을 샘플링.
        # alpha가 작으면 lam이 1에 가까워지고(약한 mixup),
        # alpha가 크면 lam이 0.5에 가까워진다(강한 mixup).
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    # 배치 내에서 무작위로 페어를 만든다. 같은 이미지가 선택될 수도 있지만
    # 확률적으로 무시할 수 있다.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    # 이미지를 lam 비율로 선형 결합.
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y, y[index], lam, index


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing 교차 엔트로피 손실.

    왜 사용하는가:
      - 일반 CE 손실은 정답 클래스에 1.0, 나머지에 0을 할당하여
        모델이 과도하게 확신하도록 만든다
      - Label Smoothing은 정답에 (1-smoothing), 나머지에 smoothing/(N-1)을
        할당하여 모델의 확신을 완화한다
      - 이는 과적합을 방지하고 캘리브레이션(예측 확률의 신뢰도)을 개선한다
    """

    def __init__(self, classes, smoothing=0.0, dim=-1):
        """
        Args:
            classes: 전체 클래스 수
            smoothing: 스무딩 비율 (0.1이면 정답에 0.9, 나머지에 0.1/(N-1) 할당)
            dim: 연산할 차원 (기본값 -1, 마지막 차원)
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing  # 정답 클래스에 할당할 확률
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        """
        예측 로짓과 정답 라벨로부터 label smoothing 손실을 계산한다.
        """
        # log_softmax: 로짓을 로그 확률로 변환 (CE 손실의 표준 단계).
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # 정답 분포 생성: 모든 클래스에 smoothing/(N-1)을 깔고,
            # 정답 클래스 위치에 confidence를 추가.
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # CE 손실 = -sum(정답분포 * 로그예측분포)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def create_logging(log_file=None, log_level=logging.INFO, file_mode='w'):
    """
    학습용 로거를 생성한다. 콘솔과 파일에 동시에 로그를 출력한다.

    Args:
        log_file: 로그 파일 경로 (None이면 콘솔만 출력)
        log_level: 로그 레벨 (INFO가 일반적)
        file_mode: 파일 모드 ('w' 덮어쓰기, 'a' 이어쓰기)
    """
    logger = logging.getLogger()

    handlers = []
    # StreamHandler: 콘솔에 로그를 출력.
    stream_handler = logging.StreamHandler()
    handlers.append(stream_handler)

    # rank 0 (메인 프로세스)만 파일에 로그를 기록.
    # 분산 학습 시 여러 프로세스가 동시에 파일에 쓰는 것을 방지.
    rank = 0

    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    return logger


def accuracy(output, target, topk=(1, )):
    """
    Top-k 정확도를 계산한다.

    Top-1: 가장 확률이 높은 예측이 정답인 비율
    Top-5: 상위 5개 예측 중 정답이 포함된 비율
    (클래스 수가 많은 세분류 태스크에서 Top-5도 유용한 지표)

    Args:
        output: 모델 예측 로짓 [B, num_classes]
        target: 정답 라벨 [B]
        topk: 계산할 k 값들의 튜플

    Returns:
        각 k에 대한 정확도 배열 (단위: %)
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # topk: 상위 k개 예측의 값과 인덱스
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()  # [k, B]로 전치
        # 정답과 비교하여 정답 위치를 True로 표시
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # 상위 k개 중 정답이 있는 샘플 수를 세고 백분율로 변환
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        # numpy 배열로 변환하여 반환 (mixup 가중 계산을 위해)
        return np.array([r.cpu().squeeze().numpy() for r in res])


def save_checkpoint(epoch, model, optimizer, args, save_name='latest'):
    """
    학습 상태(모델, 옵티마이저, epoch)를 체크포인트 파일로 저장한다.

    latest: 매 epoch마다 저장하여 학습 중단 시 재개할 수 있도록 함
    best: Top-1 정확도가 최고일 때 저장하여 최고 성능 모델을 보존

    optimizer 상태도 함께 저장하는 이유:
      SGD momentum은 이전 그래디언트를 누적하므로,
      옵티마이저 상태를 복원하지 않으면 재개 시 학습이 불안정해진다.
    """
    state_dict = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state_dict, os.path.join(args.path_log, 'fold%s_%s.pth' % (args.fold, save_name)))


def adjust_learning_rate(optimizer, idx, epoch, minibatch_count, args):
    """
    학습률을 warmup + cosine annealing 스케줄로 조정한다.

    왜 이 스케줄을 사용하는가:
      - warmup: 초기에 학습률을 낮게 시작하여 모델이 안정적으로 수렴하도록 돕는다.
        초기 가중치가 무작위에 가까울 때 큰 학습률은 그래디언트 폭발을 유발할 수 있다.
      - cosine decay: warmup 후 학습률을 코사인 곡선을 따라 점진적으로 감소시킨다.
        후반부에 학습률이 낮아지면서 미세 조정이 이루어져 더 나은 최적값에 도달한다.

    매 미니배치마다 학습률을 조정하여 부드러운 감소 곡선을 만든다.

    Args:
        optimizer: 학습률을 설정할 옵티마이저
        idx: 현재 epoch 내의 미니배치 인덱스
        epoch: 현재 epoch (1부터 시작)
        minibatch_count: 전체 미니배치 수
        args: start_lr, warmup, stop_epoch 포함

    Returns:
        현재 학습률
    """
    if epoch <= args.warmup:
        # warmup 단계: 학습률을 0에서 start_lr까지 선형적으로 증가.
        # epoch 내 미니배치 단위로 세밀하게 증가시켜 부드러운 warmup을 구현.
        lr = args.start_lr * ((epoch - 1) / args.warmup + idx / (args.warmup * minibatch_count))
    else:
        # cosine annealing: 학습률을 코사인 곡선을 따라 start_lr에서 0으로 감소.
        # 0.5*(1+cos) 공식을 사용하여 0~1 사이의 감소 비율을 계산.
        decay_rate = 0.5 * (1 + np.cos((epoch - 1) * np.pi / args.stop_epoch))
        lr = args.start_lr * decay_rate

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
