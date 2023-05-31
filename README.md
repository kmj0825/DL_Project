# DL_Project
2023-1 딥러닝 텀 프로젝트 
## **AI generated Image Classification**
---

## **1.개요**

### 1. 문제 분석
- 현재 생성 AI 를 이용하여 만들어지는 이미지가 엄청나게 증가하고 있는 상황
- 생성된 이미지 인지 실제 이미지 인지 사람이 구별하기 어려운 문제가 다수 발생


<img src = "https://drive.google.com/uc?id=1gwrWY9N39PqSKCEif3mC7YpCwZs7u6nr"  width = 1000>

<img src = "https://drive.google.com/uc?id=1Qb7plPSZKF3OGt291cmvdn3DMc7FZlhn"  width = 1000>

- 위와 같은 상황에서 AI 생성 이미지를 판별하는 것은 중요한 문제점 중 하나라고 판단하였습니다. 

### 2. 주제 선정
#### **생성 이미지 판별에 적합한 모델은 무엇일까?**
- 현재 고화질로 생성되는 이미지에 대해서 판별해주는 모델은 명확히 존재하지 않습니다. 

<img src = "https://drive.google.com/uc?id=1kJApbqhukRK94XehZHht1c88ISO2jUPs" width = 1000>
<img src = "https://drive.google.com/uc?id=1A5TvoI3GNFytZkx0WmcLE8Kl_UMQnZR1" width = 1000>

- 위와 같은 사이트에서도 진위 판별만 제공하고 있고, 어떤 모델을 사용하여 판별하는지는 제공하지 않고 있는 상황
- 현재 Image Classification 문제에서 가장 성능이 좋은 모델(CIFAR-10)은 ViT-L/16, DINOv2 과 같은 Vision Transformer 기반
  - ImageNet Dataset 의 경우에도 BASIC-L, CoCa 와 같은 Transformer 기반 방식이 성능이 좋음
- Generated Image Classification 에서도 Vision Transformer 방식이 다른 방식의 모델들 보다 성능이 뛰어날 것인지 검증
- 생성 이미지를 판단할 때, 어떤 부분을 보고 결정하는지에 대한 시각화의 필요성 - CAM 사용

### 3. 진행 과정 

#### 데이터 셋 설명
##### **CIFAKE 데이터셋**

<img src = "https://drive.google.com/uc?id=1t45T4v6W5QFQD5xUuKIM0FW1eGXkb25I"  width = 1000>
<img src = "https://drive.google.com/uc?id=1IKZvEkDUeROear1xwTX5c-Byv61lB4A8"  width = 1000>

- 데이터셋 선정 이유
CIFAR-10 데이터셋은 이미지 분류 딥러닝 모델에 대한 성능을 측정할 때, 가장 기본적으로 사용하는 데이터셋입니다. 기본적인 데이터부터 합성 데이터를 판별해 나가고자 하여 이 데이터셋을 선정하였고, 이후 고화질의 데이터셋으로 확장해나가고자 합니다. 
- 데이터셋 구성
    - 이 데이터셋은 10개의 클래스로 구성된 32 x 32 RGB 이미지 데이터로 구성되어 있습니다.
    - CIFAR-10 에서 수집한 60,000개의 실제 이미지와 실제 이미지를 바탕으로 생성된  60,000개의 합성(가짜)이미지로 구성된 데이터셋입니다.
    - 실제 데이터 수집 Class Label : ‘REAL’ or Positive class ‘1’  Krizhevsky & Hinton's CIFAR-10 dataset 에서 수집
    - 합성 데이터 생성  Class Label : ‘FAKE’ or Negative class ‘0’ CIFAR-10 을 바탕으로 Stable Diffusion version 1.4 를 이용하여 생성



## **2.모델 선정**
##### CIFAKE 논문에서는 기본적인 CNN 을 사용하여  Filter 수를 조정해가며 학습 및 평가 
#####  : CIFAKE 데이터셋이 단순하여 기본 CNN 이 유효한 성능을 가졌지만 고화질로 갈수록 문제가 있을 것이라 판단하여 더 높은 화질을 가진 데이터셋에서도 성능이 유효할 모델로 선정


<img src = "https://drive.google.com/uc?id=1lECwsj5YsUAInvaTVl0dQgI5j889_C4q" width = 1000>


##### 최종적으로 가장 높은 Accuracy 는 92.93 % 
#####  : CIFAKE 논문에서 제시하는 92.93 % 의 Accuracy 를 넘는 모델 학습 및 선정 후 제안





#### 1. VGGNet16
- 선정 이유 : 이미지 분류 문제를 해결할 때 가장 기본적으로 사용하는 모델
  - 기본적인 CNN 보다 깊은 구조를 가지므로 더 복잡한 패턴과 추상적인 특징을 인식
  - ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 대회에서 우수한 성능

#### 2. EfficientNet
- 현재 CIFAR-10 데이터셋 분류 문제에서 Transformer 계열을 제외하고 성능과 효율성이 뛰어난 모델
- 파라미터수가 적어 효율적인 모델로 빠른 학습과 판별이 가능할 것이라 예상
  - 성능과 모델 크기, 계산 및 메모리 요구 사항 사이의 균형을 잘 맞춘다는 특징
  - 네트워크 구조를 효율적으로 설계함으로써 더 적은 파라미터와 계산 비용으로 높은 성능을 달성  

#### 3. ViT 
- 현재 모든 이미지 처리 분야에서 가장 성능이 좋은 모델로 CIFAKE 문제에서도 가장 좋은 성능을 낼 것으로 예상
  - 전역적인 정보를 캡처하기 위해 self-attention 메커니즘을 사용하며, 이미지의 다양한 위치 간의 상호 작용을 고려
  - 유연한 구조 (입력 이미지의 크기나 해상도에 상관없이) 및 모듈성을 가짐 

<img src = "https://drive.google.com/uc?id=1s8sKnwDGgtCTNrz1iS6Cd015L3VGtbkY" width = 1000>


#### 주제에 대한 HyperParameter 
1. 사용하는 모델 
2. 시각화하는 CAM 방식 (어떤 부분을 보고 Fake 인지 판단하기 위해)


## **3. 전처리 및 학습 과정**

### 1. Preprocess Data (Binary Classification Preprocess)
- Kaggle 에서 데이터 zip 파일을 다운 받은 후, colab 및 gcp 에 업로드
    - 120000개 데이터 전체 unzip 과정 중에 지속적인 오류 발생 
- 사용 데이터 셋 개수 줄이기 (120000개 전체 사용에 어려움- unzip 문제 등) 
    - Train dataset : 10000
    - Validation dataset : 5000
    - Test dataset : 5000

- 생성 이미지에 대한 변형이 발생할 것이라 생각하여 기본적인 정규화 이외 다른 전처리 수행 하지 않았다. 

### 2. 모델별 학습 및 분석 과정

#### 1. VGGNET16 
<img src = "https://drive.google.com/uc?id=1p8AP7Vhf_SWth6Je3LqEfKU79orFJTHq" height = 300 width = 1000>


#### 2-1. EfficientNet
 
- MBConv2D : Mobile Inverted Bottleneck (중간에서 channel 수를 늘리고, 마지막에서 감소)

<img src = "https://drive.google.com/uc?id=19IBtvBexnifKo3fsnxHcDvykuUjePve7" height = 500 width = 1000>

- Compound scaling 
  - 이 depth, width, resolution이라는 세 가지 변수는 밀접하게 연관되어 있으며, 이를 같이 움직이는 것이 도움이 될 것
  - compound coefficient ϕ 를 사용함으로써 network의 w, d, r을 아래의 수식과 같이 scaling함
  - Scaling Method : 작은 baseline network에 대해서(compound coefficient ϕ
를 1로 고정해서) 먼저 좋은 α,β, γ 를 찾고(STEP 1) 그 다음에 전체적인 크기를 키운다(STEP 2). 

`<figure class="third">`
    <img src = "https://drive.google.com/uc?id=1TB-QJcFB8xRgjd5Aln-ptNzh07f0EOsc" width=300>
    <img src = "https://drive.google.com/uc?id=1cpX9DAP-jqBm7afDHpBIV3lLieoCZNGJ" width=200>
    <img src = "https://drive.google.com/uc?id=1_dR4EBoXr2P9fCnFRMePEDXUIMNwOEtI" width=300>
<figure>
<img src = "https://drive.google.com/uc?id=1g6db6TkDz8z_7HgnlO-aLEcgUwZb3Oxu" width = 500>



#### 2-2. EfficientNetV2
- Progressive Resizing with adaptive regularization : 작은 크기의 이미지에서 큰 크기의 이미지의 순서로 이미지 크기를 동적으로 조절하여 모델을 학습하는 방법
  - 이전 EfficientNet 은 B0~B7 에 포함되지 않는 사이즈를 가진 이미지는 작은 이미지를 큰 이미지로 크기 변환해서 학습하는 문제점 발생 
  <img src = "https://drive.google.com/uc?id=166GwSWTG0p1sYTP53swks1XWcUzcM1On">
  - 이미지의 크기가 512인 경우가 380인 경우보다 단위 시간동안 적은 수의 이미지를 학습
  - 작은 크기의 이미지로 학습할 때는 정보량이 많지 않아서 작은 용량의 모델로도 충분히 학습이 가능하기 때문에 regularization의 강도를 줄이고 큰 이미지의 경우 강도를 키워야함 
- Fused-MBConv
   - 여러 라이브러리들(cuDNN, MKL 등)에서 최적화가 잘 되어있어 depthwise 3x3 convolution 연산을 이용하였을 때 오히려 overhead가 발생
   - depthwise 3x3 convolution 연산을 3x3 convolution 연산으로 전환

<figure class="half">
     <img src = "https://drive.google.com/uc?id=1cpX9DAP-jqBm7afDHpBIV3lLieoCZNGJ" width = 400>
     <img src = "https://drive.google.com/uc?id=13naFZVGdhxlmHoZLFLyebNI5LfdgQ9R7" width = 400>
<figure>


#### 3. ViT (Vision Transformer)
-  ViT는 Transformer의 Encoder부분(Self-Attention)을 그대로 응용
<img src = "https://drive.google.com/uc?id=1YzkV8oqRUP5-7lCI_73FnFs82ZPccYwA">
- Patch Embedding : Image patch 를 통해 1D embedding을 만듦
- HyperParameter : {'patch_size': (2,4), 'embed_dim': 128, 'num_heads': (8,12), 'sequence_length': (64, 256)}
변경해 가면서 학습 수행
- 최고 성능의 Parameter : {'patch_size': 2, 'embed_dim': 128, 'num_heads': 8, 'sequence_length': 256}

> num_attention_heads (int, optional, defaults to 12) — Number of attention heads for each attention layer in the Transformer encoder.


## **4.모델 평가**

### 1.파라미터 수 비교

1. VGGNet16
---
``` 
Total params: 14,715,714

Trainable params: 7,080,450

Non-trainable params: 7,635,264 
```


2. EfficientNetV2B0
------
``` 
Total params: 6,252,882

Trainable params: 6,189,714

Non-trainable params: 63,168
```

3. ViT(Vision Transformer)
-----
``` 
Total params: 1,760,770

Trainable params: 1,760,770

Non-trainable params: 0
```
---
### 2.학습시간 비교

VGG(2.35s) < ViT (6.20 s) < EfficientNet(10.06s) 


모든 모델의 초기 학습 속도가 오래걸리나 
EfficientNet 초기 학습 속도가 특히 오래 걸림 


### 3. accuracy 비교

1. VGGNet16 : 86.65 %
<img src = "https://drive.google.com/uc?id=1j7kfWpQ-HhzD6g4VIk0gWb5KAxBRubMW" height = 300 width = 1000>
2. EfficientNet V2 B0 : 93.97 % (EfficientNetB0 성능 : 73.8047%)
<img src = "https://drive.google.com/uc?id=1od1HZccHyytmNBGcl7NaNsLguGJ1cGI6" height = 300 width = 1000>
3. ViT : 88.45 %
<img src = "https://drive.google.com/uc?id=1rwxuV3mk1biYlSqLM0RM3YNb1tXPII5m" height = 300 width = 1000>


### 모델 선정 결과 : EfficientNetV2

### 4. 시각화 방식 비교 


####CAM :  Global Average Pooling(이하 GAP) 레이어를 통해  암묵적인 attention을 드러내는 generic localizable deep representation을 구축하는 방법
<img src = "https://drive.google.com/uc?id=1eC_Yi-2WrPqgvxUDiYScDhzXiH7Hr12D">

#### CAM 시각화

실제 이미지 판별시 확인하는 부분 


---
<figure class="third">
    <img src = "https://drive.google.com/uc?id=1RJbmau1Jb_7CoQNHdJC63WG0QGvcwsic" width=300>
    <img src = "https://drive.google.com/uc?id=1HYIFc6lNiRf_qN1AKypsScNVpHBtvQUf" width=300>
    <img src = "https://drive.google.com/uc?id=1OaBc4YjfxJ-XB1bfOn6WM4jHWoQuOPq4" width=300>
<figure>

생성 이미지 판별시 확인하는 부분 


---
<figure class="half">
    <img src = "https://drive.google.com/uc?id=1H3EYwCVj4mGUUk_QpboNIGMGn2qBeLgh" width=300>
    <img src = "https://drive.google.com/uc?id=1BgoX97AIOp014Zp2z1vAo0Au2vBYxswS" width=300>
   
<figure>

### Grad-CAM 시각화
- CAM 의 일반화 -> Grad-CAM
- 어떤 target concept일지라도 final convolutional layer로 흐르는 gradient를 사용하여 이미지의 중요한 영역을 강조하는 localization map을 제작


실제 이미지 판별시 확인하는 부분

---
<figure class="third">
    <img src = "https://drive.google.com/uc?id=1EwtdDJ1LFekfx6GpfDEv1oCqnGUQ7uZE" width=300>
    <img src = "https://drive.google.com/uc?id=1EtyZX46-LXstZhddnL-i4cK6SEX979ot" width=300>
    <img src = "https://drive.google.com/uc?id=10oSaYgNmvyS2PICrkM5jy7kqHT6_hEYT" width=300>
<figure>

## **5. 결과** 

### 1. 결과 
- ViT 보다 EfficientNet 이 더 좋은 성능
- AI 가 생성한 이미지를 판별하고 설명하는데 가장 좋은 모델은 EfficientNetV2 와 Grad-CAM 방식을 함께 사용하는 것으로 선정


### 2. 한계점
1. 학습하는 데이터셋의 낮은 해상도 
  - 같은 모델을 통해 해상도를 높인 생성 이미지에 대한 학습 및 평가 필요 
  - ViT 모델의 경우 고해상도 이미지에서 더 좋은 성능 -> 낮은 Accuracy 의 이유 중 하나(self-attention 메커니즘이 이미지의 공간적인 관계를 더 잘 이해하고 다양한 위치 간의 상호 작용에 좋음) 
2. Grad-CAM 이후의 Grad-CAM++ 미적용
  - 하나의 이미지에서 같은 클래스의 물체가 여러번 발생하는 경우 제대로 히트맵을 그리지 못하는 문제점
  - Weighted Sum 을 사용하여 Visualization 이 뭉개지는 것을 개선
  