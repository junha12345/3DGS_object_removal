# 실행 방법

## 1. 환경 준비

3DGS 학습·렌더링에는 NVIDIA GPU와 CUDA 확장을 빌드할 CUDA Toolkit, C++ 컴파일러가 필요합니다. 이 실험의 3DGS 환경은 Python 3.10 / PyTorch 1.13.1 / CUDA 11.6을 사용했습니다.

```bash
git clone https://github.com/junha12345/3DGS_object_removal.git
cd 3DGS_object_removal
conda env create -f environment.yml
conda activate gaussian_splatting_object
python -m pip install --no-build-isolation 'git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git@9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0'
python -m pip install --no-build-isolation 'git+https://gitlab.inria.fr/bkerbl/simple-knn.git@86710c2d4b46680c02301765dd79e465819c8f19'
```

CUDA 확장은 외부 저장소에서 별도로 설치합니다. `fused-ssim`은 선택 사항이며, 설치하지 않으면 기본 SSIM 구현을 사용합니다.

SAM2는 PyTorch 2.x를 사용하는 별도 환경에 설치합니다. 설치와 체크포인트 다운로드는 [SAM2 저장소](https://github.com/facebookresearch/sam2)의 안내를 따릅니다. 아래 SAM2 명령은 그 환경에서 실행하고, 3DGS 학습 전에 다시 `gaussian_splatting_object` 환경을 활성화합니다.

## 2. 입력 데이터

COLMAP으로 준비한 장면을 다음 구조로 둡니다.

```text
data/scene/
├── images/             # 카메라 이미지
└── sparse/0/           # COLMAP 카메라·점군
```

`convert.py`로 COLMAP 변환을 실행할 수도 있습니다. 이 경우 `data/scene/input/`에 추출한 프레임을 넣고 `python convert.py -s data/scene`을 실행합니다. COLMAP은 별도 설치가 필요합니다.

`labels/class_masks/`에는 실제 학습에 사용한 클래스 마스크 438장이 있습니다. 이미지 이름의 확장자만 `.png`로 바꾼 이름이며, 픽셀 값이 클래스 번호입니다. 원본 촬영 영상, 전체 COLMAP 데이터와 학습된 PLY·체크포인트는 포함하지 않았으므로 기존 장면을 그대로 재현하려면 해당 파일이 별도로 필요합니다. 새로운 장면에는 그 장면에 맞는 마스크를 생성해야 합니다.

| 라벨 | 객체 | 처리 |
| --- | --- | --- |
| 0 | 미지정 영역 | 클래스 손실에서 제외 |
| 1 | 파란 토끼 인형 | 제거 대상 |
| 2 | 컵 | 유지 |
| 3 | 보라색 인형 | 제거 대상 |
| 4 | 테이블 | 유지 |

`num_objects=5`는 라벨 0을 포함한 클래스 점수의 수입니다. 색을 입힌 시각화 이미지가 아니라 단일 채널 클래스 마스크를 학습에 사용합니다. 마스크는 학습 카메라 해상도에 맞춰 최근접 보간으로 조정합니다. 라벨 0은 배경 정답을 학습한 클래스가 아니라 미지정 영역입니다.

## 3. SAM2 마스크 생성과 클래스 병합

먼저 한 프레임의 후보 마스크를 확인합니다.

```bash
python local_tools/sam2_segment_image.py \
  --image data/scene/images/frame_000107.jpg \
  --output_dir output/sam2_preview \
  --checkpoint checkpoints/sam2.1_hiera_large.pt
```

선택한 객체를 점 또는 박스로 지정하고 영상 전체에 전파합니다. 아래 좌표는 명령 형식 예시이며 실제 이미지의 객체 위치에 맞게 바꿉니다. 네 객체에 대해 `--obj_id`를 1~4로 바꿔 각각 실행합니다.

```bash
python local_tools/sam2_track_object.py \
  --input_dir data/scene/images \
  --output_dir output/sam2_masks \
  --checkpoint checkpoints/sam2.1_hiera_large.pt \
  --prompt_frame 106 --obj_id 1 \
  --point 800,400,1 --reverse \
  --offload_video_to_cpu --offload_state_to_cpu
```

`--prompt_frame`은 정렬된 이미지 목록에서 0부터 시작하는 인덱스입니다. `--reverse`는 선택 프레임 이전 시점에도 마스크를 전파합니다.

```bash
python local_tools/merge_sam2_object_masks.py \
  --sam2_mask_root output/sam2_masks \
  --output_dir output/class_masks \
  --object object_004:4 --object object_001:1 \
  --object object_002:2 --object object_003:3
```

`frame_mapping.tsv`는 SAM2의 숫자 프레임 이름과 원래 이미지 이름을 연결합니다. 객체 마스크가 겹치면 뒤에 지정한 객체의 라벨이 덮어씁니다. 테이블을 먼저 채운 뒤 인형·컵을 채우는 위 순서로 병합하면 실제 학습 라벨이 재현됩니다. 원본 SAM2 마스크를 다시 병합해 438장의 픽셀 값이 모두 일치하는 것을 확인했습니다. `labels/seed_class_mask.png`는 대표 프레임에서 선택한 마스크이며, 학습 마스크는 `labels/class_masks/`에 따로 저장했습니다.

## 4. Object Class Loss를 추가한 학습

```bash
conda activate gaussian_splatting_object
bash local_tools/train_scene_object.sh \
  data/scene models/object_scene labels/class_masks 5 0 15000 0.05
```

새로 생성한 마스크를 사용한다면 `labels/class_masks`를 `output/class_masks`로 바꿉니다. 마지막 세 인자는 GPU 번호, 학습 횟수, 클래스 손실 가중치입니다.

`train.py`는 Gaussian별 클래스 점수를 3채널씩 나눠 렌더링하고, 픽셀의 정답 라벨과 교차 엔트로피를 계산합니다.

```text
전체 손실 = RGB 재구성 손실 + lambda_object × Object Class Loss
```

클래스 점수는 `scene/gaussian_model.py`의 `_object_logits`로 관리하고, Gaussian 복제·분할·가지치기에도 함께 반영합니다. 저장한 PLY에는 `object_logit_*` 필드가 포함됩니다. 학습 결과의 `cameras.json`은 여러 시점 검증에서도 사용합니다.

## 5. 라벨에 따른 기존 객체 제거

```bash
python local_tools/filter_ply_by_object_logit.py \
  --input models/object_scene/point_cloud/iteration_15000/point_cloud.ply \
  --output output/label_only/point_cloud.ply \
  --remove-label 1 --remove-label 3
```

클래스 점수의 argmax가 라벨 1 또는 3인 Gaussian을 삭제합니다. 이 방법에서 객체 주변의 배경까지 함께 사라지는 문제가 나타났습니다.

## 6. 여러 시점 마스크 검증 후 객체 제거

```bash
python local_tools/validate_removal_multiview.py \
  --model-dir models/object_scene \
  --mask-dir labels/class_masks \
  --output-dir output/multiview \
  --iteration 15000 --labels 1 3 \
  --views 48 --vote-width 640 --margin 8 \
  --support-threshold 0.8 --min-positive-views 3 \
  --max-negative-fraction 0.1 \
  --eval-frames frame_000107.jpg frame_000146.jpg frame_000219.jpg \
    frame_000292.jpg frame_000375.jpg frame_000417.jpg
```

다른 장면에는 `--eval-frames`를 실제 카메라 이름으로 바꿉니다. 위 명령은 학습한 장면을 다시 학습하지 않고 삭제 후보를 검증합니다.

1. 지정한 라벨의 Gaussian을 삭제 후보로 선택합니다.
2. 평가 시점을 제외한 여러 카메라에서 객체 마스크 안쪽과 바깥쪽을 구분합니다.
3. 렌더링 색에 대한 미분으로 각 Gaussian의 가시성과 합성 기여량을 계산합니다.
4. 객체 안쪽에 일관되게 기여하고, 바깥쪽 영역에 기여하는 시점이 적은 후보만 삭제합니다.
5. 기존 라벨 삭제와 검증 후 삭제를 동일한 평가 시점에서 렌더링해 비교합니다.

객체 경계의 마스크 오차를 줄이기 위해 안쪽 영역은 축소하고 바깥쪽 영역은 확장합니다. 기본 조건은 객체 안쪽 기여 비율 0.8 이상, 객체 지지 시점 3개 이상, 반대 시점 비율 0.1 이하입니다. 시점별 지지·반대 판정 기준은 각각 0.7과 0.3입니다.

출력 파일:

| 파일 | 내용 |
| --- | --- |
| `point_cloud_multiview.ply` | 마스크 검증을 통과한 후보만 제거한 점군 |
| `evidence.npz` | Gaussian별 기여량, 시점 수, 검증 결과 |
| `results.json` | 사용한 시점·설정과 비교 지표 |
| `assets/*_before.png` | 제거 전 렌더링 |
| `assets/*_label_only.png` | 기존 라벨 기반 제거 |
| `assets/*_multiview.png` | 여러 시점 검증 후 제거 |

`mask_only_diagnostic` 출력은 클래스 조건 없이 마스크 검증만 적용한 비교 결과입니다. README에서 소개하는 방법은 클래스 후보와 마스크 검증을 함께 사용한 `multiview`입니다.

실제 실행 기록은 [results/multiview_validation.json](../results/multiview_validation.json)에 있습니다. 바깥쪽 영역의 변화는 크게 줄었지만 인형의 일부 흔적은 남았습니다. 대상 영역의 이미지 변화량은 객체 제거 정확도를 뜻하지 않습니다.
