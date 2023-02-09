# Open-Domain Question Answering
## 1️⃣ Introduction
본 모델은 질문에 관련된 문서를 찾아주는 "retriever" 단계와 관련된 문서를 읽고 적절한 답변을 찾거나 만들어주는 "reader" 단계로 구성되어 있습니다.
<p align="center"><img src="https://user-images.githubusercontent.com/65378914/217729308-057c696b-6c1f-41eb-970e-14ea6281c67c.png" width="80%" height="80%"/></p>

## 2️⃣ 팀원 소개

김별희|이원재|이정아|임성근|정준녕|
:-:|:-:|:-:|:-:|:-:
<img src='https://avatars.githubusercontent.com/u/42535803?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/61496071?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/65378914?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/14817039?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/51015187?v=4' height=80 width=80px></img>
[Github](https://github.com/kimbyeolhee)|[Github](https://github.com/wjlee-ling)|[Github](https://github.com/jjeongah)|[Github](https://github.com/lim4349)|[Github](https://github.com/ezez-refer)

## 3️⃣ 
## config
이 템플렛에서는 config.yaml 파일로 모든 훈련과 추론 설정을 조정할 수 있습니다. 사용할 config 파일은 cli 상 `--config`나 `-c`로 지정해 줄 수 있습니다 (디폴트 custom_config.yaml). 
본 프로젝트는 지문이 따로 주어지지 않을 때 World Knowledge에 기반해서 질의 응답을 하는 **ODQA(Open-Domain Question Answering)** Task입니다.

## Reader
Reader는 주어진 context 문장에서 query 문장의 답이 될 수 있는 sub-string을 찾아냅니다. Reader는 transformers 라이브러리의 ModelForQuestionAnswering 구조로 query를 인풋으로 받아 context 각 토큰별로 답변의 시작점과 끝점이 될 확률을 계산합니다. 답변의 최대 길이는 `config.utils.max_answer_length`로 지정할 수 있습니다.

### Reader 학습
**train.py**에서 MRC(machine reading comprehension) reader를 학습하고 검증합니다 (MRC 관련 **mrc.py** 참조). Reader로 사용할 사전학습모델은 `config.model.name_or_path`로 지정할 수 있습니다. `config.model.name_or_path`에는 HuggingFace hub에 등록된 모델의 이름(e.g. nlpotato/roberta-base-e5)이나 이미 학습되어 로컬에 저장된 모델의 체크포인트 경로(e.g. saved_models/nlpotato/roberta-base-e5/LWJ_12-23-22-11/checkpoint-9500)를 명시해야 합니다. 


Trainer에 입력해야 하는 인자들은 `config.train`에서 설정할 수 있으며, optimizer 관련 설정은 `config.optimizer`를 이용하시면 됩니다. trainer 설정 관련 자세한 설명은 [HuggingFace 공식 문서](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)를 참조하시길 바랍니다. tokenizer 관련 설정은 `config.tokenizer`로 조정할 수 있으며, tokenizer 모델은 `config.model.name_or_path`에 지정된 모델과 동일합니다. 


학습된 언어모델과 토크나이저 파일들은 `config.train.output_dir`에 명시된 경로에 저장됩니다. `output_dir`를 따로 설정하지 않으면, 학습마다 사용한 사전학습모델과 학습 시작 시간이 명시된 고유의 run_id로 명명된 아웃풋 폴더가 "saved_models/model_name/run_id"에 생깁니다.
훈련을 재개하려면 기훈련된 trainer 체크포인트가 저장된 폴더의 경로를 `config.path.resume`에 입력하면 되며, 훈련된 모델과 토크나이저를 HuggingFace Hub에 업로드하려면 `config.hf_hb.push_to_hub`을 `True`로 설정하고 hub에 등록할 모델 이름을 `config.hf_hub.save_name`에 입력하면 됩니다. Hub에 공유하기 위해서는 터미널에 `huggingface-cli login'을 쳐서 HuggingFace 계정 정보를 등록해야 합니다.


## Retreiver
Retriever는 주어진 query 문장에 적합한 문서들을 데이터베이스에서 읽어옵니다. 이때 불러오는 문서의 수를 `config.retriever.topk`로 지정할 수 있습니다.
### 1. Sparse 
Sparse Embedding을 사용하시려면 `confing.path.type`을 `sparse`로 선택해야 합니다.
#### (1) TF-IDF
Scikit-learn의 TfidfVectorizer로 query 문장과 context 문서들을 임베딩합니다. `config.retriever.sparse.tfidf_num_features`로 tfidf 벡터의 최대 크기를 지정할 수 있습니다. fit이 끝나고 tf-idf 벡터화된 context 문서들과 TfidfVectorizer 객체는 context 문장들이 저장된 `config.path.context` 폴더에 저장됩니다.
#### (2) BM25

### 2. Dense
Dense Embedding을 사용하시려면 `config.retriever.type`을 `dense`로 입력해야 합니다.

### Faiss
`config.retriever.faiss.use_faiss` 설정을 통하여 retrieval 시 Faiss를 사용할지 결정할 수 있습니다. `config.retriever.faiss.num_clusters`에 지정된 값으로 IndexIVFScalarQuantizer가 만들어내는 클러스터의 갯수를 조정할 수 있으며, 인덱싱 및 거리 계산에 쓰이는 quantizer 방식도 `config.retriever.faiss.metric`으로 정할 수 있습니다. 

