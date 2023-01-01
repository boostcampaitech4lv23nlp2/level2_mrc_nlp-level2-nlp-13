import pandas as pd
import json


#data = pd.read_csv('../data/wikipedia_documents.json')

data = ''

# with open('../data/wikipedia_documents.json', 'r', encoding='utf-8') as f:
#     wiki = json.load(f)

# print(wiki['0'])

#print(wiki['0']['text'])

# for i in range(len(wiki)):
#     idx = str(i)
#     data += wiki[idx]['text'].strip()


# txt_file = '../data/wikipedia_documents.txt'
# with open(txt_file, "w") as my_output_file:
#     my_output_file.writelines(data)
#     my_output_file.close()


# with open('../data/KorQuAD_v1.0_train.json', 'r', encoding='utf-8-sig') as f:
#     wiki = json.load(f)

# print(wiki.keys())


# print(wiki['data'][0]['paragraphs'][0]['qas'][0]['answers'][0])

# context = (wiki['data'][0]['paragraphs'][0]['context'])


# answer_end = wiki['data'][0]['paragraphs'][0]['qas'][0]['answers'][0]['answer_start'] + wiki['data'][0]['paragraphs'][0]['qas'][0]['answers'][0]






""" korquad
---'version'
---'data'
    --- paragraphs
        --- qas
            --- answers
                --- text
                --- answer_start
            --- id
            --- question
        --- context
    --- title

"""
from datasets import Dataset

#ds = Dataset.from_file("../data/train_dataset/train/dataset.arrow")
#print(ds[0]['__index_level_0__'])

a = Dataset.load_from_disk("../data/train_dataset/train")

print(pd.DataFrame(a))


#print(a['title'])


""" dataset
--- title
--- context
--- question
--- id
--- answers
    --- answer_start
    --- text
--- document_id
--- __index_level_0__

"""