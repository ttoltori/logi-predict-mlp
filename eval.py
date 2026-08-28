#!/usr/bin/env python3
"""
모델 상세 평가 스크립트.

본 스크립트는 학습된 체크포인트로 검증 데이터셋을 평가하여
이미지별 예측 결과, 클래스별 classification report, confusion matrix 등을 출력한다.

train.py --evaluate보다 훨씬 상세한 정보를 제공하며,
어떤 클래스에서 오분류가 많이 발생하는지 파악하는 데 사용한다.
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
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report
import logging
import sys

# KAN_code 앞 4자리 107개 클래스 라벨.
# dataset.py의 kan_code_list와 동일한 순서여야 한다.
# 인덱스(0~106)가 클래스 라벨 번호와 대응된다.
number_labels = ['0101', '0102', '0103', '0104', '0105', '0106', '0107', '0108', '0109', '0111',
                '0112', '0114', '0115', '0116', '0117', '0120', '0121', '0122', '0123', '0188',
                '0199', '0201', '0202', '0203', '0301', '0302', '0303', '0304', '0311', '0312',
                '0313', '0314', '0315', '0317', '0318', '0319', '0320', '0399', '0501', '0504',
                '0505', '0506', '0601', '0602', '0603', '0604', '0606', '0699', '0701', '0713',
                '0714', '0715', '0717', '0719', '0720', '0721', '0722', '0723', '0725', '0726',
                '0727', '0728', '0729', '0799', '0801', '0802', '0803', '0804', '0805', '0806',
                '0809', '0810', '0811', '0899', '0901', '0902', '0903', '0905', '0907', '0999',
                '1001', '1002', '1003', '1005', '1008', '1009', '1014', '1019', '1021', '1022',
                '1023', '1026', '1027', '1029', '1030', '1031', '1032', '1099', '1101', '1102',
                '1105', '1106', '1107', '1108', '1109', '1110', '1112']

def setup_logger(log_dir, name):
    """
    평가용 로거를 생성한다. 파일과 콘솔에 동시에 상세 로그(DEBUG 레벨)를 출력한다.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 파일 핸들러: 로그 파일에 기록
    fh = logging.FileHandler(os.path.join(log_dir, f'{name}.log'))
    fh.setLevel(logging.DEBUG)

    # 콘솔 핸들러: stdout에 출력
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)

    # 포맷터: 시간 - 메시지 형식
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def get_label_from_index(label_idx, class_to_number_label):
    """클래스 인덱스(0~106)를 KAN_code 문자열(예: '0101')로 변환한다."""
    if label_idx in class_to_number_label:
        return class_to_number_label[label_idx]
    else:
        return "Unknown Label"

def denormalize(tensor, mean, std):
    """
    정규화된 이미지 텐서를 원래 픽셀 값 범위로 되돌린다 (시각화용).

    dataset.py에서 Normalize(mean, std)를 적용했으므로,
    이미지를 화면에 표시하려면 역변환이 필요하다: pixel = tensor * std + mean
    """
    mean = torch.tensor(mean).reshape(1, 3, 1, 1)
    std = torch.tensor(std).reshape(1, 3, 1, 1)
    tensor = tensor * std + mean
    return tensor

def create_unique_dir(directory):
    """
    중복되지 않는 고유한 디렉토리를 생성한다.

    같은 이름의 디렉토리가 이미 존재하면 _1, _2, ... 접미사를 붙인다.
    이전 평가 결과를 덮어쓰지 않기 위해 사용한다.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        return directory

    count = 1
    while True:
        new_directory = f"{directory}_{count}"
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)
            return new_directory
        count += 1

def get_existing_classes(val_loader):
    """
    검증 데이터셋에 실제로 존재하는 클래스 인덱스들을 수집한다.

    왜 필요한가:
      - 107개 클래스 중 일부는 검증셋에 없을 수 있다 (특히 소수 클래스)
      - classification_report는 실제 존재하는 클래스에 대해서만 계산해야 한다
    """
    existing_classes = set()
    for _, target, _, _ in tqdm(val_loader, desc="Checking existing classes"):
        existing_classes.update(target.cpu().numpy())
    return sorted(existing_classes)

def log_gpu_info(logger):
    """GPU 메모리 사용량을 로그에 기록한다 (메모리 누수 진단용)."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"Memory Allocated: {torch.cuda.memory_allocated(i)/1024**2:.2f} MB")
            logger.info(f"Memory Cached: {torch.cuda.memory_reserved(i)/1024**2:.2f} MB")

def validate(val_loader, net, criterion, epoch, logger, args):
    """
    검증 데이터셋으로 상세 평가를 수행한다.

    train.py의 validate보다 상세한 정보를 출력한다:
      - 이미지별 예측/정답/TP/FP/TN/FN
      - 클래스별 confusion matrix
      - classification_report (precision, recall, f1-score)
      - Top-1/Top-5 정확도
    """
    logger.info('='*50)
    logger.info(f'[{datetime.datetime.now()}] Starting evaluation for epoch {epoch}')
    logger.info('Evaluation Parameters:')
    logger.info(f'- Model: {args.model_name}')
    logger.info(f'- Batch size: {args.batch_size}')
    logger.info(f'- Data directory: {args.data_dir}')
    logger.info(f'- Image only mode: {args.image_only}')
    logger.info(f'- MLP type: {args.mlp_type}')
    logger.info(f'- MLP dimensions: d={args.mlp_d}, h={args.mlp_h}, n={args.mlp_n}')
    log_gpu_info(logger)
    logger.info('='*50)
    
    net.eval()
    torch.set_printoptions(precision=4, sci_mode=False, linewidth=120)
    
    # 메트릭 초기화
    acc1_sum = 0
    acc5_sum = 0
    loss_sum = 0
    validation_num = 0
    all_predictions = []
    all_targets = []
    
    # Confusion Matrix 값 초기화
    confusion_matrix = {}  # 각 클래스별 TP, FP, TN, FN 저장
    
    # 클래스 정보 설정
    existing_classes = get_existing_classes(val_loader)
    class_to_number_label = {i: number_labels[i] for i in existing_classes}
    logger.info(f'Found {len(existing_classes)} classes in validation data')
    
    # 배치 처리
    for i, (images, target, location, image_paths) in enumerate(tqdm(val_loader, desc="Validation Progress")):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        batch_size = images.size(0)
        logger.info(f'\n[{timestamp}] Processing batch {i+1}/{len(val_loader)}')
        
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        location = location.cuda(non_blocking=True).float()
        
        with torch.no_grad():
            output = net(images, location) if not args.image_only else net(images)
            predictions = output.argmax(dim=1)
            
            # 각 이미지에 대해 confusion matrix 계산
            for idx, (pred, true) in enumerate(zip(predictions, target)):
                pred_label = get_label_from_index(pred.item(), class_to_number_label)
                true_label = get_label_from_index(true.item(), class_to_number_label)
                
                # confusion matrix 값 업데이트
                if true_label not in confusion_matrix:
                    confusion_matrix[true_label] = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
                
                # 현재 클래스에 대한 TP, FP, TN, FN 계산
                for cls in confusion_matrix:
                    if cls == true_label:
                        if pred_label == true_label:
                            confusion_matrix[cls]['TP'] += 1
                        else:
                            confusion_matrix[cls]['FN'] += 1
                    else:
                        if pred_label == cls:
                            confusion_matrix[cls]['FP'] += 1
                        else:
                            confusion_matrix[cls]['TN'] += 1
                
                # 로깅
                logger.info(f'[{timestamp}] Image: {image_paths[idx]}')
                logger.info(f'- Ground Truth: {true_label}')
                logger.info(f'- Prediction: {pred_label}')
                logger.info(f'- TP: {confusion_matrix[true_label]["TP"]} '
                          f'FP: {confusion_matrix[true_label]["FP"]} '
                          f'TN: {confusion_matrix[true_label]["TN"]} '
                          f'FN: {confusion_matrix[true_label]["FN"]}')
                
                # # 틀린 예측 이미지 저장
                # if pred_label != true_label:
                #     img = denormalize(images[idx].cpu().unsqueeze(0), 
                #                     [0.485, 0.456, 0.406], 
                #                     [0.229, 0.224, 0.225])
                #     img = img.squeeze(0).numpy().transpose(1, 2, 0)
                #     img = np.clip(img, 0, 1)
                    
                #     fig, ax = plt.subplots(figsize=(5, 5))
                #     ax.imshow(np.ones((img.shape[0] + 50, img.shape[1], 3), dtype=np.float32))
                #     ax.imshow(img, extent=[0, img.shape[1], 50, img.shape[0] + 50])
                #     ax.axis('off')
                #     ax.text(10, 35, f'True: {true_label}', color='blue', fontsize=12)
                #     ax.text(10, 15, f'Pred: {pred_label}', color='red', fontsize=12)
                    
                #     save_path = os.path.join(pred_fail_dir, f"{os.path.basename(image_paths[idx])}")
                #     plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                #     plt.close(fig)
            
            # 배치 메트릭 계산
            batch_loss = criterion(output, target).item()
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            
            validation_num += batch_size
            acc1_sum += acc1.item() * batch_size
            acc5_sum += acc5.item() * batch_size
            loss_sum += batch_loss * batch_size
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # 배치 결과 로깅
            logger.info(f'\nBatch {i+1} Results:')
            logger.info(f'- Loss: {batch_loss:.4f}')
            logger.info(f'- Top-1 Accuracy: {acc1.item():.2f}%')
            logger.info(f'- Top-5 Accuracy: {acc5.item():.2f}%')
    
    # 최종 결과 로깅
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info('='*50)
    logger.info(f'[{timestamp}] Final Validation Results:')
    logger.info(f'Total samples: {validation_num}')
    # logger.info('\nConfusion Matrix Results per Class:')
    # for cls in confusion_matrix:
    #     logger.info(f'\nClass {cls}:')
    #     logger.info(f'TP: {confusion_matrix[cls]["TP"]} FP: {confusion_matrix[cls]["FP"]} '
    #                f'TN: {confusion_matrix[cls]["TN"]} FN: {confusion_matrix[cls]["FN"]}')
    
    final_loss = loss_sum / validation_num
    final_acc1 = acc1_sum / validation_num
    final_acc5 = acc5_sum / validation_num
    
    logger.info(f'\nAverage loss: {final_loss:.4f}')
    logger.info(f'Top-1 accuracy: {final_acc1:.2f}%')
    logger.info(f'Top-5 accuracy: {final_acc5:.2f}%')
    logger.info('='*50)
    
    return final_acc1, final_acc5, [
        f"val e: {epoch}",
        f'acc1: {final_acc1:.4f}',
        f'acc5: {final_acc5:.4f}',
        f'loss: {final_loss:.4f}',
    ]

def main():
    """
    평가 파이프라인의 메인 진입점.

    체크포인트를 로드하고 검증 데이터셋으로 상세 평가를 수행한 후
    결과를 로그 파일과 콘솔에 출력한다.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, type=str)
    parser.add_argument('--data', default='inat21_mini', type=str, help='inat21_mini|inat21_full')
    parser.add_argument('--data_dir', default='datasets/inat21', type=str)
    parser.add_argument('--save_dir', default='./logs', type=str)
    parser.add_argument('--model_file', default='sk2res2net_dynamic_mlp', type=str, help='model file name')
    parser.add_argument('--model_name', default='sk2res2net101', type=str, help='model type in detail')
    parser.add_argument('--fold', default=1, type=int, help='training fold')
    parser.add_argument('--random_seed', default=37, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--warmup', default=2, type=int)
    parser.add_argument('--start_lr', default=0.04, type=float)
    parser.add_argument('--stop_epoch', default=90, type=int)
    parser.add_argument('--num_workers', default=32, type=int)
    parser.add_argument('--tencrop', action='store_true', default=False)
    parser.add_argument('--image_only', action='store_true', default=False)
    parser.add_argument('--metadata', default='geo_temporal', type=str, help='geo_temporal|geo|temporal')
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--mlp_type', default='c', type=str, help='dynamic mlp versions: a|b|c')
    parser.add_argument('--mlp_d', default=256, type=int)
    parser.add_argument('--mlp_h', default=64, type=int)
    parser.add_argument('--mlp_n', default=2, type=int)
    args = parser.parse_args()
    args.mlp_cin = 4
    
    # 실행 커맨드 로깅을 위한 문자열 생성
    command = ' '.join(sys.argv)
    
    # 랜덤 시드 설정
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 디렉토리 설정
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    #pred_fail_dir = create_unique_dir(os.path.join(args.save_dir, f'{args.data}_pred-fail_{timestamp}'))
    log_dir = create_unique_dir(os.path.join(args.save_dir, f'{args.data}_log_{timestamp}'))
    
    # 로거 설정
    logger = setup_logger(log_dir, args.name)
    
    # 시작 시 실행 정보 로깅
    logger.info('='*50)
    logger.info(f'[{timestamp}] Starting evaluation script')
    logger.info(f'Execution command: {command}')
    logger.info(f'Arguments: {args}')
    logger.info('='*50)
    
    # 데이터 로더 설정
    logger.info('Loading validation dataset...')
    val_loader = dataset.load_val_dataset(args)
    logger.info(f'Validation dataset size: {len(val_loader.dataset)}')
    logger.info(f'Number of batches: {len(val_loader)}')
    
    # 모델 설정
    logger.info('Building model...')
    net = models.__dict__[args.model_file].__dict__[args.model_name](logger, args)
    logger.info(f'\nModel Architecture:\n{net}')
    
    # GPU 설정
    if torch.cuda.is_available():
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        logger.info(f'Using {torch.cuda.device_count()} GPUs')
    
    # Loss function 설정
    criterion = utils.LabelSmoothingLoss(classes=args.num_classes, smoothing=0.1)
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    
    # 체크포인트 로드
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume)
            
            if 'model' in checkpoint:
                net.load_state_dict(checkpoint['model'])
                logger.info("Loaded model state from checkpoint")
            else:
                net.load_state_dict(checkpoint)
                logger.info("Loaded entire checkpoint")
                
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
                logger.info(f"Resuming from epoch {start_epoch}")
        else:
            logger.error(f"No checkpoint found at {args.resume}")
            return
    
    # 평가 실행
    logger.info("Starting evaluation...")
    acc1, acc5, outputs = validate(val_loader, net, criterion, 0, logger, args)
    
    # 결과 저장
    results = {
        'timestamp': timestamp,
        'acc1': acc1,
        'acc5': acc5,
        'args': vars(args)
    }
    
    results_file = os.path.join(log_dir, f'results_{timestamp}.txt')
    with open(results_file, 'w') as f:
        for key, value in results.items():
            f.write(f'{key}: {value}\n')
    
    logger.info('\n' + '='*50)
    logger.info('Evaluation Complete!')
    logger.info(f'Top-1 Accuracy: {acc1:.2f}%')
    logger.info(f'Top-5 Accuracy: {acc5:.2f}%')
    logger.info(f'Results saved to: {results_file}')
    #logger.info(f'Predictions saved to: {pred_fail_dir}')
    logger.info(f'Logs saved to: {log_dir}')
    logger.info('='*50)

if __name__ == '__main__':
    main()

"""
###################로그 출력 안하게 하려면 아래 주석을 푸세요###################
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
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# 실제 라벨
number_labels = ['0101', '0102', '0103', '0104', '0105', '0106', '0107', '0108', '0109', '0111', 
                                '0112', '0114', '0115', '0116', '0117', '0120', '0121', '0122', '0123', '0188', 
                                '0199', '0201', '0202', '0203', '0301', '0302', '0303', '0304', '0311', '0312', 
                                '0313', '0314', '0315', '0317', '0318', '0319', '0320', '0399', '0501', '0504', 
                                '0505', '0506', '0601', '0602', '0603', '0604', '0606', '0699', '0701', '0713', 
                                '0714', '0715', '0717', '0719', '0720', '0721', '0722', '0723', '0725', '0726', 
                                '0727', '0728', '0729', '0799', '0801', '0802', '0803', '0804', '0805', '0806', 
                                '0809', '0810', '0811', '0899', '0901', '0902', '0903', '0905', '0907', '0999', 
                                '1001', '1002', '1003', '1005', '1008', '1009', '1014', '1019', '1021', '1022', 
                                '1023', '1026', '1027', '1029', '1030', '1031', '1032', '1099', '1101', '1102', 
                                '1105', '1106', '1107', '1108', '1109', '1110', '1112']

# 숫자 라벨을 변환
def get_label_from_index(label_idx, number_labels):
    if 0 <= label_idx < len(number_labels):
        return number_labels[label_idx]
    else:
        return "Unknown Label"
    
# 디노멀라이즈 함수
def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).reshape(1, 3, 1, 1)
    std = torch.tensor(std).reshape(1, 3, 1, 1)
    tensor = tensor * std + mean  # 반대로 표준편차를 곱하고 평균을 더함
    return tensor

# --data로 받은 디렉토리가 중복되면 +1
def create_unique_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        count = 1
        new_directory = f"{directory}_{count}"
        while os.path.exists(new_directory):
            count += 1
            new_directory = f"{directory}_{count}"
        os.makedirs(new_directory)
        directory = new_directory
    return directory

def get_existing_classes(val_loader):
    existing_classes = set()
    for _, target, _, _ in val_loader:
        existing_classes.update(target.cpu().numpy())
    return sorted(existing_classes)  # 실제 존재하는 클래스만 정렬하여 반환

# 숫자 라벨을 변환하는 함수 (실제 val 데이터에 있는 클래스만 매핑)
def get_label_from_index(label_idx, class_to_number_label):
    if label_idx in class_to_number_label:
        return class_to_number_label[label_idx]
    else:
        return "Unknown Label"

def validate(val_loader, net, criterion, epoch, logger, args, pred_fail_dir):
    logger.info('eval epoch {}'.format(epoch))
    net.eval()

    acc1_sum = 0
    acc5_sum = 0
    loss = 0
    valdation_num = 0

    # 실제로 val 데이터에 존재하는 클래스와 number_labels 간 매핑 생성
    existing_classes = get_existing_classes(val_loader)
    class_to_number_label = {i: number_labels[i] for i in existing_classes}

    # tqdm을 val_loader에 적용하여 진행 상황 표시
    val_loader = tqdm(val_loader, desc="Validation Progress(배치 단위)")

    for i, (images, target, location, image_paths) in enumerate(val_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        location = location.cuda(non_blocking=True).float()

        with torch.no_grad():
            if args.image_only:
                output = net(images)
            else:
                output = net(images, location)
            #print('output:', output)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        num = target.size(0)
        valdation_num += num
        acc1_sum += acc1.item() * num
        acc5_sum += acc5.item() * num
        loss += criterion(output, target).item()

        pred = output.argmax(dim=1)
        incorrect_indices = (pred != target).nonzero(as_tuple=True)[0]

        # 잘못된 예측에 대한 이미지 저장
        # for idx in incorrect_indices:
        #     # 디노멀라이즈
        #     orig_image = images[idx].cpu()
        #     orig_image = denormalize(orig_image.unsqueeze(0), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     orig_image = orig_image.squeeze(0).numpy().transpose(1, 2, 0)
        #     orig_image = np.clip(orig_image, 0, 1)

        #     true_label_idx = target[idx].item()
        #     pred_label_idx = pred[idx].item()

        #     # 실제 라벨과 예측 라벨을 val에 있는 실제 클래스 기반으로 숫자 라벨로 변환
        #     true_label_text = get_label_from_index(true_label_idx, class_to_number_label)
        #     pred_label_text = get_label_from_index(pred_label_idx, class_to_number_label)

        #     # 이미지의 원본 파일 이름 추출
        #     image_filename = os.path.basename(image_paths[idx])  # 경로에서 파일명만 추출
            
        #     # 흰색 배경에 이미지 및 텍스트 표시
        #     fig, ax = plt.subplots(figsize=(5, 5))
        #     ax.imshow(np.ones((orig_image.shape[0] + 50, orig_image.shape[1], 3), dtype=np.float32))  # 흰색 배경
        #     ax.imshow(orig_image, extent=[0, orig_image.shape[1], 50, orig_image.shape[0] + 50])  # 이미지 위에 표시
        #     ax.axis('off')

        #     # 텍스트 배치 수정: True가 Pred 위로 오도록 조정
        #     ax.text(10, 35, f'True: {true_label_text}', color='blue', fontsize=12)  # True를 더 아래에 표시
        #     ax.text(10, 15, f'Pred: {pred_label_text}', color='red', fontsize=12)   # Pred를 더 위에 표시

        #     # 이미지 저장 (pred와 true가 다른 경우만)
        #     save_path = os.path.join(pred_fail_dir, f"{image_filename}")  # pred-fail 디렉토리에 저장
        #     plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        #     plt.close(fig)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, type=str)
    parser.add_argument('--data', default='inat21_mini', type=str, help='inat21_mini|inat21_full')
    parser.add_argument('--data_dir', default='datasets/inat21', type=str)
    parser.add_argument('--save_dir', default='./logs', type=str)
    parser.add_argument('--model_file', default='sk2res2net_dynamic_mlp', type=str, help='model file name')
    parser.add_argument('--model_name', default='sk2res2net101', type=str, help='model type in detail')
    parser.add_argument('--fold', default=1, type=int, help='training fold')
    parser.add_argument('--random_seed', default=37, type=int)
    # train
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--warmup', default=2, type=int)
    parser.add_argument('--start_lr', default=0.04, type=float)
    parser.add_argument('--stop_epoch', default=90, type=int)
    parser.add_argument('--num_workers', default=32, type=int)
    # data
    parser.add_argument('--tencrop', action='store_true', default=False)
    parser.add_argument('--image_only', action='store_true', default=False)
    parser.add_argument('--metadata', default='geo_temporal', type=str, help='geo_temporal|geo|temporal')
    # model
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--evaluate', action='store_true', help='evaluate model on validation set')
    # dynamic MLP
    parser.add_argument('--mlp_type', default='c', type=str, help='dynamic mlp versions: a|b|c')
    parser.add_argument('--mlp_d', default=256, type=int)
    parser.add_argument('--mlp_h', default=64, type=int)
    parser.add_argument('--mlp_n', default=2, type=int)
    args = parser.parse_args()
    args.mlp_cin = 4  #

    # set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    args.nprocs = torch.cuda.device_count()

    # 1. --data로 받은 데이터셋 이름에 _pred-fail을 붙여서 디렉토리 생성
    pred_fail_dir = create_unique_dir(os.path.join(args.save_dir, f'{args.data}_pred-fail'))

    # 2. log 디렉토리를 생성하고 중복 시 +1 추가
    creat_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
    log_dir = create_unique_dir(os.path.join(args.save_dir, f'{args.data}_log'))  # 로그 디렉토리 생성
    args.path_log = os.path.join(log_dir, f'{args.name}')  # 로그 파일 이름 설정
    os.makedirs(args.path_log, exist_ok=True)  # 로그 디렉토리 생성

    # 로그 파일 생성
    logger = utils.create_logging(os.path.join(args.path_log, '%s_eval.log' % creat_time))

    # get datasets
    val_loader = dataset.load_val_dataset(args)

    # get net
    net = models.__dict__[args.model_file].__dict__[args.model_name](logger, args)
    net.cuda()
    net = torch.nn.DataParallel(net)

    # get criterion
    criterion = utils.LabelSmoothingLoss(classes=args.num_classes, smoothing=0.1).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            state_dict = torch.load(args.resume)
            if 'model' in state_dict:
                net.load_state_dict(state_dict['model'])
                logger.info("=> loaded checkpoint '{}'".format(args.resume))
            else:
                net.load_state_dict(state_dict)
                logger.info("=> loaded checkpoint '{}'".format(args.resume))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # Perform validation
    epoch = 0  # Set to 0 or any other epoch number
    acc1, acc5, outputs = validate(val_loader, net, criterion, epoch, logger, args, pred_fail_dir)
    logger.info('\t'.join(outputs))
    logger.info('Exp path: %s' % args.path_log)

if __name__ == '__main__':
    main()
"""