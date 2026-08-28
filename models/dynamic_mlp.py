"""
Dynamic MLP 코어 모듈.

본 모듈은 Dynamic MLP의 핵심 컴포넌트를 구현한다.
Dynamic MLP는 메타데이터(본 프로젝트에서는 상품의 가로/세로 크기)로부터
동적으로 가중치를 생성하여 이미지 feature를 변조하는 구조이다.

핵심 아이디어:
  일반적인 CNN은 이미지 feature에 고정된 FC 레이어를 적용한다.
  Dynamic MLP는 메타데이터에 따라 FC 레이어의 가중치를 동적으로 생성하여,
  동일한 이미지라도 메타데이터에 따라 다른 변환이 적용된다.
  이를 통해 이미지만으로는 구별하기 어려운 유사한 클래스를
  메타데이터 정보를 활용해 구별할 수 있다.

구조:
  1. FCNet: 메타데이터 → 임베딩 벡터 (loc_net)
  2. Dynamic_MLP_A/B/C: 이미지 feature + loc 임베딩 → 동적 변환
  3. FusionModule: 백본 feature(2048차원)에 Dynamic MLP를 잔차 형태로 적용
"""
import torch
import torch.nn as nn


class FCResLayer(nn.Module):
    """
    잔차 연결이 있는 FC 레이어.

    왜 잔차 연결을 사용하는가:
      - 깊은 네트워크에서 그래디언트 소실을 방지한다
      - 입력을 출력에 더함으로써 학습이 쉬워진다
      - 메타데이터 임베딩 네트워크(FCNet)의 기본 building block
    """

    def __init__(self, linear_size=256):
        """
        Args:
            linear_size: 입력/출력 차원 (동일해야 잔차 덧셈이 가능)
        """
        super(FCResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()  # 정규화를 위한 드롭아웃
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        """잔차 연결: y = x + W2(ReLU(W1(x))) 형태로 입력을 보존하며 변환."""
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y  # 잔차 더하기
        return out


class FCNet(nn.Module):
    """
    메타데이터 인코더 네트워크 (loc_net).

    메타데이터 벡터(4차원: sin/cos 인코딩된 width/height)를
    더 높은 차원의 임베딩(256차원)으로 변환한다.
    이 임베딩은 Dynamic MLP에서 동적 가중치를 생성하는 데 사용된다.

    왜 단순 선형 변환이 아닌 깊은 FC 네트워크를 사용하는가:
      - 메타데이터와 클래스 간의 관계가 비선형적일 수 있다
      - 깊은 네트워크가 더 풍부한 표현을 학습할 수 있다
    """

    def __init__(self, num_inputs, num_classes=1000, num_filts=256):
        """
        Args:
            num_inputs: 입력 차원 (mlp_cin=4, sin/cos 인코딩된 메타데이터)
            num_classes: 출력 차원 (mlp_d=256, FusionModule에 전달될 임베딩 차원)
            num_filts: 내부 hidden 차원
        """
        super(FCNet, self).__init__()
        self.inc_bias = False
        self.feats = nn.Sequential(
            nn.Linear(num_inputs, num_filts),  # 4 → 256 차원으로 확장
            nn.ReLU(inplace=True),
            # 4개의 잔차 FC 레이어로 깊은 표현 학습
            FCResLayer(num_filts),
            FCResLayer(num_filts),
            FCResLayer(num_filts),
            FCResLayer(num_filts),
        )
        # 최종 임베딩 출력 레이어. bias를 사용하지 않는 이유:
        # Dynamic MLP에서 내적 연산을 깔끔하게 유지하기 위해.
        self.class_emb = nn.Linear(num_filts, num_classes, bias=self.inc_bias)

    def forward(self, x, return_feats=False, class_of_interest=None):
        """
        메타데이터를 임베딩 벡터로 변환한다.

        Args:
            x: 메타데이터 벡터 [B, mlp_cin]
            return_feats: True면 중간 feature 반환 (디버깅/분석용)
            class_of_interest: 특정 클래스에 대한 예측만 계산 (추론 최적화용)

        Returns:
            임베딩 벡터 [B, mlp_d] 또는 클래스 예측
        """
        loc_emb = self.feats(x)
        if return_feats:
            return loc_emb  # [B, num_filts]
        if class_of_interest is None:
            class_pred = self.class_emb(loc_emb)
        else:
            class_pred = self.eval_single_class(loc_emb, class_of_interest)

        return class_pred  # [B, num_classes]

    def eval_single_class(self, x, class_of_interest):
        """
        특정 클래스에 대한 예측만 계산한다 (추론 시 효율성).
        전체 클래스에 대한 행렬 곱 대신 하나의 클래스 가중치만 사용.
        """
        if self.inc_bias:
            return torch.matmul(x, self.class_emb.weight[class_of_interest, :]) + self.class_emb.bias[class_of_interest]
        else:
            return torch.matmul(x, self.class_emb.weight[class_of_interest, :])


class Basic1d(nn.Module):
    """
    1D 기본 블록: Linear → (LayerNorm) → ReLU.

    bias=False인 경우 LayerNorm을 추가하여 정규화를 수행한다.
    bias=True인 경우 ReLU만 추가한다 (bias가 이미 오프셋을 제공).
    """

    def __init__(self, in_channels, out_channels, bias=True):
        """
        Args:
            in_channels: 입력 차원
            out_channels: 출력 차원
            bias: Linear 레이어의 bias 사용 여부
        """
        super().__init__()
        conv = nn.Linear(in_channels, out_channels, bias)
        self.conv = nn.Sequential(conv, )
        if not bias:
            # bias가 없으면 LayerNorm으로 정규화하여 학습 안정성 확보.
            self.conv.add_module('ln', nn.LayerNorm(out_channels))
        self.conv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        """Linear → 정규화 → ReLU 순서로 변환."""
        out = self.conv(x)
        return out


class Dynamic_MLP_A(nn.Module):
    """
    Dynamic MLP 변형 A: 가장 단순한 버전.

    메타데이터(loc)로부터 직접 변환 행렬을 생성하여 이미지 feature에 적용한다.
    구조: loc → Linear → weight [inplanes × planes] → bmm(img_fea, weight)

    왜 동적 가중치를 사용하는가:
      - 고정된 가중치는 모든 메타데이터에 동일한 변환을 적용한다
      - 동적 가중치는 메타데이터에 따라 다른 변환을 적용하여
        메타데이터 정보를 feature에 주입할 수 있다
    """

    def __init__(self, inplanes, planes, loc_planes):
        """
        Args:
            inplanes: 이미지 feature 입력 차원
            planes: 출력 차원
            loc_planes: 메타데이터 임베딩 차원
        """
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes

        # loc 임베딩으로부터 변환 행렬(inplanes × planes)을 직접 생성.
        self.get_weight = nn.Linear(loc_planes, inplanes * planes)
        self.norm = nn.LayerNorm(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, img_fea, loc_fea):
        """
        이미지 feature를 메타데이터 기반 동적 가중치로 변환한다.

        Args:
            img_fea: 이미지 feature [B, inplanes]
            loc_fea: 메타데이터 임베딩 [B, loc_planes]

        Returns:
            변환된 feature [B, planes]
        """
        # loc 임베딩으로부터 변환 행렬 생성: [B, inplanes*planes] → [B, inplanes, planes]
        weight = self.get_weight(loc_fea)
        weight = weight.view(-1, self.inplanes, self.planes)

        # 배치 행렬 곱: img_fea [B, 1, inplanes] × weight [B, inplanes, planes] → [B, 1, planes]
        img_fea = torch.bmm(img_fea.unsqueeze(1), weight).squeeze(1)
        img_fea = self.norm(img_fea)
        img_fea = self.relu(img_fea)

        return img_fea


class Dynamic_MLP_B(nn.Module):
    """
    Dynamic MLP 변형 B: 중간 복잡도 버전.

    A와의 차이: 이미지 feature와 loc 임베딩을 각각 변환한 후 결합한다.
    이미지 feature에서도 가중치 생성에 기여하도록 하여
    더 풍부한 동적 변환을 가능하게 한다.
    """

    def __init__(self, inplanes, planes, loc_planes):
        """
        Args:
            inplanes: 이미지 feature 입력 차원
            planes: 출력 차원
            loc_planes: 메타데이터 임베딩 차원
        """
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes

        # 이미지 feature 변환 경로
        self.conv11 = Basic1d(inplanes, inplanes, True)
        self.conv12 = nn.Linear(inplanes, inplanes)

        # loc 임베딩 변환 경로 (가중치 생성)
        self.conv21 = Basic1d(loc_planes, inplanes, True)
        self.conv22 = nn.Linear(inplanes, inplanes * planes)

        # 정규화 + 활성화
        self.br = nn.Sequential(
            nn.LayerNorm(planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = Basic1d(planes, planes, False)

    def forward(self, img_fea, loc_fea):
        """
        이미지 feature와 loc 임베딩을 각각 변환 후 결합하여 동적 변환을 수행한다.
        """
        # 이미지 feature 경로
        weight11 = self.conv11(img_fea)
        weight12 = self.conv12(weight11)

        # loc 임베딩 경로: 가중치 행렬 생성
        weight21 = self.conv21(loc_fea)
        weight22 = self.conv22(weight21).view(-1, self.inplanes, self.planes)

        # 두 경로의 결과를 배치 행렬 곱으로 결합
        img_fea = torch.bmm(weight12.unsqueeze(1), weight22).squeeze(1)
        img_fea = self.br(img_fea)
        img_fea = self.conv3(img_fea)

        return img_fea


class Dynamic_MLP_C(nn.Module):
    """
    Dynamic MLP 변형 C: 가장 복잡한 버전 (기본값, --mlp_type c).

    B와의 차이: 이미지 feature와 loc 임베딩을 concat하여
    가중치 생성에 모두 동시에 사용한다.
    이를 통해 이미지와 메타데이터의 상호작용을 더 직접적으로 모델링한다.
    """

    def __init__(self, inplanes, planes, loc_planes):
        """
        Args:
            inplanes: 이미지 feature 입력 차원
            planes: 출력 차원
            loc_planes: 메타데이터 임베딩 차원
        """
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes

        # 이미지 feature와 loc 임베딩을 concat하여 사용하므로 입력 차원이 inplanes + loc_planes.
        self.conv11 = Basic1d(inplanes + loc_planes, inplanes, True)
        self.conv12 = nn.Linear(inplanes, inplanes)

        self.conv21 = Basic1d(inplanes + loc_planes, inplanes, True)
        self.conv22 = nn.Linear(inplanes, inplanes * planes)

        self.br = nn.Sequential(
            nn.LayerNorm(planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = Basic1d(planes, planes, False)

    def forward(self, img_fea, loc_fea):
        """
        이미지 feature와 loc 임베딩을 concat하여 동적 가중치를 생성하고 변환한다.
        """
        # 이미지 feature와 메타데이터 임베딩을 concat.
        # 이를 통해 가중치 생성 시 두 정보의 상호작용을 직접 반영할 수 있다.
        cat_fea = torch.cat([img_fea, loc_fea], 1)

        # concat된 feature로부터 두 개의 중간 표현을 생성
        weight11 = self.conv11(cat_fea)
        weight12 = self.conv12(weight11)

        weight21 = self.conv21(cat_fea)
        weight22 = self.conv22(weight21).view(-1, self.inplanes, self.planes)

        # 배치 행렬 곱으로 동적 변환 수행
        img_fea = torch.bmm(weight12.unsqueeze(1), weight22).squeeze(1)
        img_fea = self.br(img_fea)
        img_fea = self.conv3(img_fea)

        return img_fea


class RecursiveBlock(nn.Module):
    """
    Dynamic MLP 1층을 래퍼로 감싸는 블록.

    FusionModule에서 여러 번 반복 적용하기 위해 별도 블록으로 분리.
    mlp_type에 따라 A/B/C 중 하나를 선택하여 사용한다.
    """

    def __init__(self, inplanes, planes, loc_planes, mlp_type='c'):
        """
        Args:
            inplanes: 입력 차원
            planes: 출력 차원
            loc_planes: 메타데이터 임베딩 차원
            mlp_type: 'a' | 'b' | 'c' (기본값 'c')
        """
        super().__init__()
        # mlp_type에 따라 적절한 Dynamic MLP 변형 선택
        if mlp_type.lower() == 'a':
            MLP = Dynamic_MLP_A
        elif mlp_type.lower() == 'b':
            MLP = Dynamic_MLP_B
        elif mlp_type.lower() == 'c':
            MLP = Dynamic_MLP_C

        self.dynamic_conv = MLP(inplanes, planes, loc_planes)

    def forward(self, img_fea, loc_fea):
        """Dynamic MLP를 한 번 적용하고 (img_fea, loc_fea)를 반환."""
        img_fea = self.dynamic_conv(img_fea, loc_fea)
        return img_fea, loc_fea


class FusionModule(nn.Module):
    """
    Dynamic MLP 융합 모듈.

    백본에서 추출한 이미지 feature(2048차원)에 메타데이터 기반 Dynamic MLP를
    잔차(residual) 형태로 적용한다.

    구조:
      1. conv1: 2048 → 256 차원으로 축소 (연산량 감소)
      2. RecursiveBlock × mlp_n: 256차원에서 Dynamic MLP 반복 적용
      3. conv3: 256 → 2048 차원으로 복원
      4. 원본 feature에 더함 (잔차 연결)

    왜 잔차 연결을 사용하는가:
      - Dynamic MLP가 feature를 완전히 대체하지 않고 보완하도록 함
      - 학습 초기에 Dynamic MLP가 의미있는 변환을 학습하지 못해도
        원본 feature가 보존되어 학습이 안정적으로 진행됨
    """

    def __init__(self, inplanes=2048, planes=256, hidden=64, num_layers=2, mlp_type='c'):
        """
        Args:
            inplanes: 백본 feature 차원 (2048)
            planes: Dynamic MLP 작업 차원 (256)
            hidden: 중간 hidden 차원 (64)
            num_layers: RecursiveBlock 반복 횟수
            mlp_type: Dynamic MLP 변형 ('a'|'b'|'c')
        """
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.hidden = hidden

        # 2048 → 256 차원 축소. Dynamic MLP 연산을 저차원에서 수행하여 효율성 확보.
        self.conv1 = nn.Linear(inplanes, planes)

        # RecursiveBlock 스택 구성.
        # num_layers=1이면 단일 블록, 2 이상이면 첫/마지막 블록은 차원 변환,
        # 중간 블록들은 hidden 차원에서 동작.
        conv2 = []
        if num_layers == 1:
            conv2.append(RecursiveBlock(planes, planes, loc_planes=planes, mlp_type=mlp_type))
        else:
            # 첫 블록: planes(256) → hidden(64)
            conv2.append(RecursiveBlock(planes, hidden, loc_planes=planes, mlp_type=mlp_type))
            # 중간 블록들: hidden → hidden
            for _ in range(1, num_layers - 1):
                conv2.append(RecursiveBlock(hidden, hidden, loc_planes=planes, mlp_type=mlp_type))
            # 마지막 블록: hidden(64) → planes(256)
            conv2.append(RecursiveBlock(hidden, planes, loc_planes=planes, mlp_type=mlp_type))
        self.conv2 = nn.ModuleList(conv2)

        # 256 → 2048 차원 복원. 잔차 연결을 위해 원본 차원으로 되돌림.
        self.conv3 = nn.Linear(planes, inplanes)
        self.norm3 = nn.LayerNorm(inplanes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, img_fea, loc_fea):
        """
        이미지 feature에 Dynamic MLP를 잔차 형태로 적용한다.

        Args:
            img_fea: 백본 feature [B, 2048] (avgpool 후 flatten된 벡터)
            loc_fea: 메타데이터 임베딩 [B, 256] (FCNet 출력)

        Returns:
            융합된 feature [B, 2048]
        """
        # 원본 feature를 보존 (잔차 연결용)
        identity = img_fea

        # 저차원 공간으로 투영
        img_fea = self.conv1(img_fea)

        # RecursiveBlock을 순차적으로 적용.
        # 각 블록은 img_fea를 변환하고 loc_fea는 그대로 전달.
        for m in self.conv2:
            img_fea, loc_fea = m(img_fea, loc_fea)

        # 원래 차원으로 복원
        img_fea = self.conv3(img_fea)
        img_fea = self.norm3(img_fea)

        # 잔차 더하기: 원본 feature + Dynamic MLP 변환
        img_fea += identity

        return img_fea


def get_dynamic_mlp(inplanes, args):
    """
    args 설정에 따라 FusionModule을 생성하여 반환한다.

    Args:
        inplanes: 백본 feature 차원 (ResNet/SK2Res2Net 모두 2048)
        args: mlp_d, mlp_h, mlp_n, mlp_type 포함

    Returns:
        FusionModule 인스턴스
    """
    return FusionModule(inplanes=inplanes,
                        planes=args.mlp_d,      # 작업 차원 (기본 256)
                        hidden=args.mlp_h,      # 중간 hidden (기본 64)
                        num_layers=args.mlp_n,  # 반복 횟수 (기본 2)
                        mlp_type=args.mlp_type) # 변형 (기본 'c')
