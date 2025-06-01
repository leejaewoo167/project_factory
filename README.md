# ML Experiment

### Introduction

ML(Machine Learning)의 workflow는 크게 load data - preprocessing - model training - model validation - model test로 이루어집니다. 이 workflow는 이미 획일화된 과정으로 일괄적으로 처리를 해야합니다. 이에 대한 이유는 다음과 같습니다. 

1. model selection에 대해 pairwise하게 진행하여 제안 모델의 우수성을 파악
2. 귀찮음 방지
   - ~~굳이 code를 하나하나 selection하고 반복적으로 돌릴 필요가 있을까요?~~
3. cheating 방지
   -  cheating이란, training phase와 validation phase에서 test data를 참여시켜 모델 성능 측정하는 것입니다. 
   - 이는 연구윤리에 어긋나는 문제에 해당되며, paper가 reject되는 불상사가 발생합니다.
   - industry의 경우는 해당 product에 대한 신뢰가 떨어져 매출 하락으로 이어질 수 있습니다.
   - 참고: training phase에서는 모델이 데이터를 학습하고, validation phase에서는 model selection을 위한 성능 검증을 진행합니다. 즉, hyper-parameter searching 수행 시 validation phase의 성능을 기준으로 model을 selection합니다. 학습을 진행하지 않지만,  model selection을 하는 단계로 test phase에 사용되는 dataset을 참여시키면 안됩니다.
4. 확장성
   - 모델만 추가하면 바로 실험의 전과정을 수행할 수 있습니다.
   - 기능을 추가하여 실험 진행 및 결과 확인이 용이합니다. 

이 template은 rule based model을 대상으로,  Scikit-learn API의 object 구조에 따라 구성하였습니다.

---

### Setting

- 해당 template에 사용된 라이브러리들을 세팅합니다.
- 가상환경은 anaconda virtual environment 기준으로 진행합니다. 
- OS는 windows, terminal은 bash로 합니다. 
- terminal의 directory 경로는 이 directory의 최상단에 위치해야 합니다.

0. Required Libraries

```shell
fire==0.7.0
loguru==0.7.2
scipy==1.15.3
pandas==2.2.3
numpy==2.2.5
scikit-learn==1.6.1 
imbalanced-learn==0.13.0
lightgbm==4.6.0
xgboost==2.1.4
```

- fire는 conda에서 지원을 안해주기 때문에 pip으로 설치하셔야 합니다

1. 가상환경 세팅

- conda 가상환경을 proj_factory 이름으로 requirements.txt의 라이브러리 버전으로 설치하되 중간에 y/[n] 부분은 모두 yes로 처리하도록 합니다.
- Mac에서는 안될 가능성이 높습니다. Mac에서 하려면 따로 가상환경을 만드셔서 진행하시길 바랍니다.
- 아래와 안된다면 0번에서 소개한 library들을 수동으로 설치하시길 바랍니다.

```shell
conda create --name proj_factory --file requirements.txt -y
```

2. 가상환경 활성화

```shell
conda activate proj_factory
```

---

### Model Attachment

- 모델을 추가하는 방법에 대해서 알아봅시다
- 현재 버전 기준 크게 최상단의 config와 src/models 폴더들만 수정합니다. 

__Process__

1. src/models/model_attach_template.py를 copy & paste하고 해당 파일명을 모델명(대문자)으로 수정합니다.

2. template에 따라 parameter의 후보군과 모델을 선언해줍니다.

   - 함수명의 model  name을 수정해야 합니다.
   - param_set, classifier 변수명은 변경해도 되지만, return의 dictionary 양식을 지켜줘야 합니다. __특히, key는 수정하면 안됩니다.__
   - param_set은 src/utils.py 내에 있는 set_parameters를 사용하시면 됩니다. 필요시 적절히 custimizing 가능합니다.(VOTING.py 참고)
   - logger.info는 모델이 잘 불러왔는지 확인하기 위한 디버깅 용도로 삭제해도 좋지만, 불안하다면 info 내용을 수정해도 됩니다.

3. src/models/select_model.py 에 model을 등록합니다.

   1. 상단에 model을 import합니다.
   2. SelectModel 함수 내에 model 변수에 '[모델명 소문자]': [import한 모델명] 형식으로 작성하고 저장합니다.

4. 최상단 config 폴더에 모델명(소문자).json을 만듭니다.

   - 이미 존재하는 파일을  copy&paste해도 됩니다.

   - model, seed, export, test_split_ratio는 기본 option이고, 모델에 따라서 hyper-parameter를 추가할 수 있습니다.

   - 추가적인 hyper-parameter는 다음과 같은 rule에 따라 작성합니다.

     > __Rule__
     >
     > - hyper-parameter 명은 최대한 model의 hyper-parameter와 비슷한 의미를 가지는 이름으로 설정할 것
     > - 최소 길이 1에서 최대 길이 3의 리스트 형식으로 작성할 것
     > - 길이가 1의 경우: 해당 파라미터만 사용
     > - 길이가 2의 경우: [min, max]로 파라미터가 min~max의 범위에서 1의 증가량으로 파라미터 후보군을 만듦
     > - 길이가 3의 경우: [min, max, increment]로 min~max 의 범위로 지정한 증가량(increment)만큼 파라미터 후보군을 만듦
     > - 자세한 logic은 src>utils.py>set_parameter를 확인

---

### Demo

- 어떻게 실행하는지 알아봅시다

- __OS는 Windows 그리고 terminal은 bash 기준으로 진행합니다.__

1. 먼저 터미널에서 아래와 같은 명령어를 입력하여 src 로 이동합니다.

```shell
cd src
```

2. 그리고 아래와 같은 명령어를 실행하여 실험을 진행합니다.

```shell
python main.py --model {model name} --seed {seed number} --test_split_ratio {split ratio} ...
```

- 예를 들어,  seed number를 580로 하고, xgboost 모델의 depth 범위를 5~10을 주고, 실험을 진행하고 싶으면 아래와 같이 명령어를 작성합니다.

```shell
python main.py --model xgboost --seed 580 --depth [5,10]
```

---

### Parameters

config file 기준

| option           | Descripttion                         | Default |
| ---------------- | ------------------------------------ | ------- |
| model            | 모델명(소문자)                       | -       |
| seed             | random seed number                   | 600     |
| export           | 실험 결과와 학습된 모델 저장(미개발) | False   |
| test_split_ratio | train과 test 데이터 분할 비율        | 0.2     |

- 이외 다른 매개변수는 모델 별로 다르므로 config의 파일을 보고 적절히 customizing을 합니다.

- __voting의 경우는 parameter의 조합이 연쇄되어 있으므로 해당 모델의 config 파일의 내용을 변경하여 실험을 진행하시면 됩니다.__ 

  ```shell
  python main.py --model voting
  ```

---

### File Tree

```shell
├── config
│   ├── bagging.json
│   ├── ...
├── data
│   ├── submit.csv
│   ├── test.csv
│   └── train.csv
├── reports
├── requirements.txt
├── results
└── src
    ├── EDA
    │   ├── Count_code_time.py
    │   ├── Find_thresholds.py
    │   ├── Scaling_dist.py
    │   ├── main.py
    │   └── utils.py
    ├── load_data.py
    ├── main.py
    ├── models
    │   ├── BAGGING.py
    │   ├── LIGHTGBM.py
    │   ├── RANDOMFOREST.py
    │   ├── VOTING.py
    │   ├── XGBOOST.py
    │   ├── model_attach_template.py
    │   └── select_model.py
    ├── test.py
    ├── train.py
    └── utils.py
```

- config: 각 모델의 hyper-parameter 및 실험 option을 담은 json파일을 보관합니다.
- data: dataset을 보관합니다.
- reports: dashboard에 관한 source들을 보관합니다.
- results: 실험 파일들을 보관합니다.
- src: 실험 및 EDA 파일들을 보관합니다.
  -  EDA: EDA와 관련한 파일들을 보관합니다.
  - models: model을 작성한 파일들을 보관합니다.

---

### Recommended References

- [pbd(python debugger)](https://jh-bk.tistory.com/22)

- 



'이게 뭐지?' 하는 부분이 있으면 주저없이 저에게 연락부탁드립니다.

구경민
