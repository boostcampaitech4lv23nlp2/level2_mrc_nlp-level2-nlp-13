import pandas as pd
import numpy as np

import sys
import re
import string

import json

def normalize_answer(s): # 후처리 함수
    def remove_(text):
        ''' 불필요한 기호 제거 '''
        text = re.sub("'", " ", text)
        text = re.sub('"', " ", text)
        text = re.sub('《', " ", text)
        text = re.sub('》', " ", text)
        #text = re.sub('≪', " ", text)
        #text = re.sub('≫', " ", text)
        text = re.sub('<', " ", text)
        text = re.sub('>', " ", text) 
        text = re.sub('〈', " ", text)
        text = re.sub('〉', " ", text)   
        #text = re.sub("\(", " ", text)
        #text = re.sub("\)", " ", text)
        text = re.sub("\"", " ", text)
        text = re.sub("‘", " ", text)
        text = re.sub("’", " ", text)
        text = re.sub("「", " ", text)
        text = re.sub("」", " ", text)
        #text = re.sub(r'\([^)]*\)','', text)
        return text
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    '''
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    '''
    #return white_space_fix(remove_punc(lower(remove_(s))))
    return white_space_fix(remove_(s))

# 1. 파일 불러오기
with open('predictions.json', encoding='utf-8-sig') as f:
    js = json.loads(f.read()) ## json 라이브러리 이용

# 2. pandas로 변환
result = pd.DataFrame(js, index = ['answer'])
result = result.transpose()

# 3. 후처리 실행
for i in range(600):
    result['answer'][i] =  normalize_answer(result['answer'][i])
result.to_json('temp.json',force_ascii=False)

# 4. answer column 삭제
with open('temp.json', encoding='utf-8-sig') as f:
    js = json.loads(f.read()) ## json 라이브러리 이용
js2 = str(js)
#print(js2)
js2 = js2[11:-1]
js2 = js2.replace("'", '"')

file  = open('processed_predictions.json' , 'w' ) 
file.write(js2)
file.close()