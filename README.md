# 1. 프로젝트 개요

### 1.1 프로젝트 주제
<img src="https://user-images.githubusercontent.com/78770033/211195602-c5b75969-8575-43a4-b137-0b073b24b836.png">
- 일반적인 영화 추천 대회의 경우, 사용자의 영화 시청 이력을 바탕으로 해당 사용자가 다음에 시청할 영화를 예측합니다.  
- 그러나 실제 상황에서는 서버 불량 등 여러가지 이유로 데이터가 Sequential하게 적재되지 않을 가능성이 존재합니다.  
- 따라서 본 대회에서는 사용자의 영화 시청 이력을 바탕으로 당므에 시청할 영화 뿐 아니라 누락되었을 수 있는 영화 또한 예측하는 것을 목표로 합니다.  

#### 1.1.1 데이터 개요
<img src="https://user-images.githubusercontent.com/78770033/211195692-17c89cf8-0e8a-42e8-b9a1-02efb03ea04b.png">
- MovieLens 데이터를 전처리 하여 만든 Implicit Feedback 기반의 Sequential Recommendation 시나리오를 바탕으로 사용자의 Time-ordered Sequence에서 일부 Item이 누락된 상황을 상정합니다.  
- 이와 함께 영화와 관련된 Side Imformation으로 영화별 감독, 장르, 제목, 작가, 개봉 년도를 제공합니다.  
- 31,360명의 User, 6,807개의 Item, 5,154,471개의 Interaction으로 구성되어 있으며 Sparsity는 97.6%입니다.  

### 1.2 프로젝트 요약 
- 평가 Metric: Recall@10  
- 제공된 Baseline 코드 및 RecBole을 사용해 모델 구축  
- Sequential, General, Context-aware 모델 수십여 개를 학습하여 성능 실험  
- 앙상블을 활용한 성능 개선  
- 최종 결과  
    - Public LB 0.1662 (5위) -> Private LB 0.1622 (8위)  
    
### 1.3 활용 장비 및 협업 툴  

- GPU: V100 5대   
- 운영체제: Ubuntu 18.04.5 LTS  
- 협업툴: Github, Notion, Weight & Bias  

### 1.4 프로젝트 구조

```
Movie Recommendation/
│
├── train.py - main script to start training
├── inference.py - make submission with trained models
├── ensemble.py - make ensemble with submission files
│
├── config/ - holds configurations for training
|   ├──LSTM_config.json
|   ├──transformer_config.json
|   ├──transformerLSTM_config.json
|   ├──GRUtransformer_config.json
│   └──GTN_congfig.json
│
├── data_loader/ - anything about data loading goes here
│   ├── dataset.py
│   └── preprocess.py
│
├── data/ - default directory for storing input data
│
├── model/ - base, get_model, utils for model, and all of models
│   ├── base.py
│   ├── get_model.py
│   ├── utils.py
│   ├── LSTM.py
│   ├── transformer.py
│   ├── transformerLSTM.py
│   ├── transformerGRU.py
│   ├── GRUtransformer.py
│   ├── GTN.py
│   ├── GTNGRU.py
│   └── XGBoost.py
│
├── trainer/ - trainers, losses, metric, optimizer, and scheduler
│   ├── trainer.py
│   ├── loss.py
│   ├── metric.py
│   ├── optimizer.py
│   └── scheduler.py
|
├── preprocess/ - preprocess ipynb files
│
├── ensembles/ - anything about ensemble goes here
│   └── ensemble.py
|
├── ensembles_inference/ - submission files that needs to be ensembled
|
├── logger/ - module for wandb  and logging
│   └── wandb_logger.py
│
├── saved_model/
|
├── submission/
|
└── utils/ - small utility functions
    ├── util.py
    └── ...
```

---

# 2. 프로젝트 팀 구성 및 역할

- ?

---

# 3. 프로젝트 수행 결과 (Public 5위 / Private 8위)

---

# 4. References

- Boostcourse 강의 자료

---

# 5. Contributors

| <img src="https://user-images.githubusercontent.com/64895794/200263288-1d77b5f8-ed79-4548-9bc1-01aec2474aaa.png" width=200> | <img src="https://user-images.githubusercontent.com/64895794/200263509-9f564042-6da7-4410-a820-c8198037b0b3.png" width=200> | <img src="https://user-images.githubusercontent.com/64895794/200263683-37597e1d-10c1-483c-90f2-fb4749310e40.png" width=200> | <img src="https://user-images.githubusercontent.com/64895794/200263783-52ddbcf3-5e0b-431e-a84d-f7f17f3d061e.png" width=200> | <img src="https://user-images.githubusercontent.com/64895794/200264314-77728a99-9849-41e9-b13d-be120877a184.png" width=200> |
| :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: |
|                                           [류명현](https://github.com/ryubright)                                            |                                           [이수경](https://github.com/41ow1ives)                                            |                                            [김은혜](https://github.com/kimeunh3)                                            |                                         [정준환](https://github.com/Jeong-Junhwan)                                          |                                            [장원준](https://github.com/jwj51720)                                            |
