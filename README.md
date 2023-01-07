# 1. 프로젝트 개요

## Movie Recommendation

사용자의 영화 시청 이력 데이터를 바탕으로 사용자가 다음에 시청할 영화 및 좋아할 영화를 예측하는 대회입니다.   
MovieLens 데이터를 전처리 하여 implicit feedback 기반의 sequential recommendation 시나리오를 바탕으로 사용자의 time-ordered sequence에서 일부 item이 누락된 상황을 상정합니다. 아이템인 영화와 관련된 content(side-information)가 함께 제공됩니다. 위 상황에서 원본 데이터가 있다면 특정 시점 이후의 데이터(sequential)와 특정 시점 이전의 일부 데이터(static)를 임의로 추출하여 정답 데이터로 사용합니다.

## 프로젝트 목표

- ㄱ
- ㄴ

## 활용 장비 및 협업 툴

- GPU: V100 5대
- 운영체제: Ubuntu 18.04.5 LTS
- 협업툴: Github, Notion, Weight & Bias

## Folder Structure

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
