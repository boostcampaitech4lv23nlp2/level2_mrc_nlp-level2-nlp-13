# Open-Domain Question Answering
## 1️⃣ Introduction
본 프로젝트는 지문이 따로 주어지지 않을 때 World Knowledge에 기반해서 질의 응답을 하는 **ODQA(Open-Domain Question Answering)** Task입니다.
본 모델은 질문에 관련된 문서를 찾아주는 "retriever" 단계와 관련된 문서를 읽고 적절한 답변을 찾거나 만들어주는 "reader" 단계로 구성되어 있습니다.
<p align="center"><img src="https://user-images.githubusercontent.com/65378914/217729308-057c696b-6c1f-41eb-970e-14ea6281c67c.png" width="80%" height="80%"/></p>

## 2️⃣ 팀원 소개

김별희|이원재|이정아|임성근|정준녕|
:-:|:-:|:-:|:-:|:-:
<img src='https://avatars.githubusercontent.com/u/42535803?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/61496071?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/65378914?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/14817039?v=4' height=80 width=80px></img>|<img src='https://avatars.githubusercontent.com/u/51015187?v=4' height=80 width=80px></img>
[Github](https://github.com/kimbyeolhee)|[Github](https://github.com/wjlee-ling)|[Github](https://github.com/jjeongah)|[Github](https://github.com/lim4349)|[Github](https://github.com/ezez-refer)

## 3️⃣ 데이터
![ODQA_data](https://user-images.githubusercontent.com/65378914/217733088-a82c1f7e-9739-4192-9e8c-7314fc4bcde0.png)
MRC 데이터의 경우, HuggingFace에서 제공하는 datasets 라이브러리를 이용하여 접근이 가능합니다. 해당 directory를 dataset_name 으로 저장한 후, 아래의 코드를 활용하여 불러올 수 있습니다.
```
# train_dataset을 불러오고 싶은 경우
from datasets import load_from_disk
dataset = load_from_disk("./data/train_dataset/")
print(dataset)
```
Retrieval 과정에서 사용하는 문서 집합(corpus)은 ./data/wikipedia_documents.json 으로 저장되어있습니다. 약 5만 7천개의 unique 한 문서로 이루어져 있습니다.
데이터셋은 편의성을 위해 Huggingface 에서 제공하는 datasets를 이용하여 pyarrow 형식의 데이터로 저장되어있습니다. 다음은 ./data 구조입니다.
```
# 전체 데이터
./data/
    # 학습에 사용할 데이터셋. train 과 validation 으로 구성
    ./train_dataset/
    # 제출에 사용될 데이터셋. validation 으로 구성
    ./test_dataset/
    # 위키피디아 문서 집합. retrieval을 위해 쓰이는 corpus.
    ./wikipedia_documents.json
```

<details>
    <summary><b><font size="10">데이터 예시</font></b></summary>
<div markdown="1">

```
![ex](https://user-images.githubusercontent.com/65378914/217733295-1d6a3166-3582-454b-8e9b-01409b5e8597.png)
- id: 질문의 고유 id
- question: 질문
- answers: 답변에 대한 정보. 하나의 질문에 하나의 답변만 존재함
- answer_start : 답변의 시작 위치
- text: 답변의 텍스트
- context: 답변이 포함된 문서
- title: 문서의 제목
- document_id: 문서의 고유 id
```
</div>
</details>

## 4️⃣ 모델 설명
## Reader
Reader는 주어진 context 문장에서 query 문장의 답이 될 수 있는 sub-string을 찾아냅니다. <br>
Reader는 transformers 라이브러리의 ModelForQuestionAnswering 구조로 query를 인풋으로 받아 context 각 토큰별로 답변의 시작점과 끝점이 될 확률을 계산합니다. <br>
답변의 최대 길이는 `config.utils.max_answer_length`로 지정할 수 있습니다.<br>

## Retreiver
Retriever는 주어진 query 문장에 적합한 문서들을 데이터베이스에서 읽어옵니다.  <br>
이때 불러오는 문서의 수를 `config.retriever.topk`로 지정할 수 있습니다.
### 1. Sparse 
Sparse Embedding을 사용하시려면 `confing.path.type`을 `sparse`로 선택해야 합니다.
#### (1) TF-IDF
Scikit-learn의 TfidfVectorizer로 query 문장과 context 문서들을 임베딩합니다. `config.retriever.sparse.tfidf_num_features`로 tfidf 벡터의 최대 크기를 지정할 수 있습니다. fit이 끝나고 tf-idf 벡터화된 context 문서들과 TfidfVectorizer 객체는 context 문장들이 저장된 `config.path.context` 폴더에 저장됩니다.
#### (2) BM25

### 2. Dense
Dense Embedding을 사용하시려면 `config.retriever.type`을 `dense`로 입력해야 합니다.

### Faiss
`config.retriever.faiss.use_faiss` 설정을 통하여 retrieval 시 Faiss를 사용할지 결정할 수 있습니다. `config.retriever.faiss.num_clusters`에 지정된 값으로 IndexIVFScalarQuantizer가 만들어내는 클러스터의 갯수를 조정할 수 있으며, 인덱싱 및 거리 계산에 쓰이는 quantizer 방식도 `config.retriever.faiss.metric`으로 정할 수 있습니다. 

<details>
    <summary><b><font size="10">Project Tree</font></b></summary>
<div markdown="1">

```
.
```
</div>
</details>

## 3️⃣ How to Run
## 환경 설정
```python
$ bash install_requirements.sh
```

## config
이 템플렛에서는 config.yaml 파일로 모든 훈련과 추론 설정을 조정할 수 있습니다. 사용할 config 파일은 cli 상 `--config`나 `-c`로 지정해 줄 수 있습니다 (디폴트 custom_config.yaml). 

## Reader 
**train.py**에서 MRC(machine reading comprehension) reader를 학습하고 검증합니다 (MRC 관련 **mrc.py** 참조). <br>
Reader로 사용할 사전학습모델은 `config.model.name_or_path`로 지정할 수 있습니다. <br>
`config.model.name_or_path`에는 HuggingFace hub에 등록된 모델의 이름(e.g. nlpotato/roberta-base-e5)이나 이미 학습되어 로컬에 저장된 모델의 체크포인트 경로(e.g. saved_models/nlpotato/roberta-base-e5/LWJ_12-23-22-11/checkpoint-9500)를 명시해야 합니다. <br>

Trainer에 입력해야 하는 인자들은 `config.train`에서 설정할 수 있으며, optimizer 관련 설정은 `config.optimizer`를 이용하시면 됩니다. <br>
trainer 설정 관련 자세한 설명은 [HuggingFace 공식 문서](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments)를 참조하시길 바랍니다. <br>
tokenizer 관련 설정은 `config.tokenizer`로 조정할 수 있으며, tokenizer 모델은 `config.model.name_or_path`에 지정된 모델과 동일합니다. <br>

학습된 언어모델과 토크나이저 파일들은 `config.train.output_dir`에 명시된 경로에 저장됩니다. <br>
`output_dir`를 따로 설정하지 않으면, 학습마다 사용한 사전학습모델과 학습 시작 시간이 명시된 고유의 run_id로 명명된 아웃풋 폴더가 "saved_models/model_name/run_id"에 생깁니다.<br>
훈련을 재개하려면 기훈련된 trainer 체크포인트가 저장된 폴더의 경로를 `config.path.resume`에 입력하면 되며, 훈련된 모델과 토크나이저를 HuggingFace Hub에 업로드하려면 `config.hf_hb.push_to_hub`을 `True`로 설정하고 hub에 등록할 모델 이름을 `config.hf_hub.save_name`에 입력하면 됩니다. <br>
Hub에 공유하기 위해서는 터미널에 `huggingface-cli login'을 쳐서 HuggingFace 계정 정보를 등록해야 합니다.<br>
