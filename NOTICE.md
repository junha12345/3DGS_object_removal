# 기반 코드

학습·렌더링 코드는 [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)의 커밋 `54c035f7834b564019656c3e3fcc3646292f727d`를 기반으로 합니다. 원본 코드의 저작권 표시와 [LICENSE.md](LICENSE.md)를 유지했습니다.

이 저장소에서 추가한 내용은 SAM2 클래스 마스크 생성·병합, Gaussian별 객체 클래스 점수와 Object Class Loss, 라벨 기반 객체 제거, 여러 시점 마스크를 이용한 삭제 후보 검증입니다. COLMAP 변환에는 순차 매칭 및 GPU 선택 옵션을 추가했습니다.

CUDA 확장은 실험 환경에서 사용한 커밋으로 고정했습니다.

| 확장 | 커밋 |
| --- | --- |
| diff-gaussian-rasterization | `9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0` |
| simple-knn | `86710c2d4b46680c02301765dd79e465819c8f19` |
| fused-ssim | `1272e21a282342e89537159e4bad508b19b34157` |

SAM2는 별도 설치하는 [facebookresearch/sam2](https://github.com/facebookresearch/sam2)를 사용합니다.
