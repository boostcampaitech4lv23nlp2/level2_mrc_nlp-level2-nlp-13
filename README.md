# Open-Domain Question Answering

### Retriever 학습
```python
python train_dpr.py
```
p,q_encoder는 saved_models/DPR/encoder 경로에 저장됩니다.
학습 후 model: best_p_encoder_path, best_q_encoder_path 지정해야 합니다.

### Reader 학습
```python
python train.py
```
모델은 saved_models/klue 경로에 저장됩니다. (roberta-base를 사용했다면 saved_models/klue/roberta-base에 저장됩니다)
학습 후 config.yaml에서 model: name_or_path 지정해야 합니다. 

### Inference
```python
python inference.py
```
predict 파일은 predictions 폴더에 저장된다.

### BM25와 hybrid를 적용하고 싶다면
sparse:
  embedding_type: bm25
retriever:
  type: hybrid 로 설정
