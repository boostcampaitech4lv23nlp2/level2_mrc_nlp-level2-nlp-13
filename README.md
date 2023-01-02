# Open-Domain Question Answering

이 템플렛에서는 config.yaml 파일로 모든 훈련과 추론 설정을 조정할 수 있습니다. 사용할 config 파일은 cli 상 `--config`나 `-c`로 지정해 줄 수 있습니다 (디폴트 custom_config.yaml). 

## Reader
Reader는 주어진 context 문장에서 query 문장의 답이 될 수 있는 sub-string을 찾아냅니다. Reader는 transformers 라이브러리의 ModelForQuestionAnswering 구조로 query를 인풋으로 받아 context 각 토큰별로 답변의 시작점과 끝점이 될 확률을 계산합니다. 답변의 최대 길이는 `config.utils.max_answer_length`로 지정할 수 있습니다.

### Reader 학습
**train.py**에서 MRC(machine reading comprehension) reader를 학습하고 검증합니다 (MRC 관련 **mrc.py** 참조). Reader로 사용할 사전학습모델은 `config.model.name_or_path`로 지정할 수 있습니다. `config.model.name_or_path`에는 HuggingFace hub에 등록된 모델의 이름(e.g. nlpotato/roberta-base-e5)이나 이미 학습되어 로컬에 저장된 모델의 체크포인트 경로(e.g. saved_models/nlpotato/roberta-base-e5/LWJ_12-23-22-11/checkpoint-9500)를 명시해야 합니다. 


Trainer에 입력해야 하는 인자들은 `config.train`에서 설정할 수 있으며, optimizer 관련 설정은 `config.optimizer`를 이용하시면 됩니다. trainer 설정 관련 자세한 설명은 [HuggingFace 공식 문서](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)를 참조하시길 바랍니다. tokenizer 관련 설정은 `config.tokenizer`로 조정할 수 있으며, tokenizer 모델은 `config.model.name_or_path`에 지정된 모델과 동일합니다. 


학습된 언어모델과 토크나이저 파일들은 `config.train.output_dir`에 명시된 경로에 저장됩니다. `output_dir`를 따로 설정하지 않으면, 학습마다 사용한 사전학습모델과 학습 시작 시간이 명시된 고유의 run_id로 명명된 아웃풋 폴더가 "saved_models/model_name/run_id"에 생깁니다.
훈련을 재개하려면 기훈련된 trainer 체크포인트가 저장된 폴더의 경로를 `config.path.resume`에 입력하면 되며, 훈련된 모델과 토크나이저를 HuggingFace Hub에 업로드하려면 `config.hf_hb.push_to_hub`을 `True`로 설정하고 hub에 등록할 모델 이름을 `config.hf_hub.save_name`에 입력하면 됩니다. Hub에 공유하기 위해서는 터미널에 `huggingface-cli login'을 쳐서 HuggingFace 계정 정보를 등록해야 합니다.


## Retreiver
Retriever는 주어진 query 문장에 적합한 문서들을 데이터베이스에서 읽어옵니다. 이때 불러오는 문서의 수를 `config.retriever.topk`로 지정할 수 있습니다.
### Sparse 
Sparse Embedding을 사용하시려면 `confing.path.type`을 `sparse`로 선택해야 합니다.
#### TF-IDF
Scikit-learn의 TfidfVectorizer로 query 문장과 context 문서들을 임베딩합니다. `config.retriever.sparse.tfidf_num_features`로 tfidf 벡터의 최대 크기를 지정할 수 있습니다. fit이 끝나고 tf-idf 벡터화된 context 문서들과 TfidfVectorizer 객체는 context 문장들이 저장된 `config.path.context` 폴더에 저장됩니다.

### Dense

### Faiss
`config.retriever.faiss.use_faiss` 설정을 통하여 retrieval 시 Faiss를 사용할지 결정할 수 있습니다. `config.retriever.faiss.num_clusters`에 지정된 값으로 IndexIVFScalarQuantizer가 만들어내는 클러스터의 갯수를 조정할 수 있으며, 인덱싱 및 거리 계산에 쓰이는 quantizer 방식도 `config.retriever.faiss.metric`으로 정할 수 있습니다. 

## 주의사항
1. 