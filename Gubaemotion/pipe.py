from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import pandas as pd

import numpy as np




class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings  
        #{'input_ids':句子中单词的token，'token_type_ids'区分上下句，'attention_mask'：参与attentio部分padding部分是否参与attention}

    def __getitem__(self, idx): #使用词查询索引
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}  #encoding 字典
        # print(self.labels)
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])
    
def load_data(data_file):
    df = pd.read_csv(data_file,names=['text'])
    train_texts = list(df['text'].values)

    return train_texts

import argparse


def paser_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--ernie_4',type=str,default="Guba_ernir_base/classify/ernie_4_pretrained")
    parser.add_argument('--ernie_5',type=str,default="Guba_ernir_base/classify/ernie_5_pretrained")
    parser.add_argument('--class4_file',type=str,default='Guba_ernir_base/classify/datasets/calss_4.txt')
    parser.add_argument('--class5_file',type=str,default='Guba_ernir_base/classify/datasets/class_5.txt')
    
    args = parser.parse_args()
    return args
    


class plumber():
    '''
    标签包括：
        4分类：愤怒，高兴，低落，恐惧
        5分类：愤怒，高兴，低落，恐惧，无情绪
    输入：
    text：可以输入单个句子字符串，或者将多个句子封装在列表中
    num_labels:选择分类任务（4；5）
    
    输出：每行文本以及他对应的情绪类别
    '''
    def __init__(self,text,num_labels):
        args = paser_args()
        self.num_labels = num_labels
        self.text = text
        # args = parse_args() #传入参数
        # set_seed(args.seed)  #设置种子
        #device = torch.device('cuda')
        if self.num_labels == 4:
            self.tokenizer = AutoTokenizer.from_pretrained(args.ernie_4)#分词器  四分类
            self.model = AutoModelForSequenceClassification.from_pretrained(args.ernie_4, num_labels=self.num_labels) #bert预训练模型
            self.class_list = np.array([x.strip() for x in open(
            args.class4_file).readlines()]) #要改
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(args.ernie_5)#分词器   五分类
            self.model = AutoModelForSequenceClassification.from_pretrained(args.ernie_5, num_labels=self.num_labels)
            self.class_list = np.array([x.strip() for x in open(
            args.class5_file).readlines()]) #要改
    
    def classify(self):
        train_encodings = self.tokenizer(self.text, padding='max_length', truncation=True, max_length=512,return_tensors="pt")
        output = self.model(**train_encodings)
        output_hat = F.softmax(output[0], dim=-1) #每个类别的概率  得到的是一个tenser 二维
        y_hat = output[0].argmax(dim=1) #预测标签索引  返回一个tensor
        y_hat_np = y_hat.numpy()
        y_predict = self.class_list[y_hat_np]

        return y_predict
    
    def concat(self):
        return pd.DataFrame({"text":self.text,"label":self.classify()})

        
    def __call__(self):
        return self.concat()
    
def SequenceClassification(text,num_labels=5):
    result = plumber(text,num_labels)
    return result()
    

if __name__ == '__main__':
    text =['第三行为成分，是双方实际交往的外在表现和结果，也就是我们常说的人际交往。','要做到这一点，首先要彼此认识到对方的伤害，这是宽恕的前提。有些人在没弄清楚自己的愤怒和痛苦之前就试图宽恕别人，结果自己的不满仍然无法消除。']
    c = plumber(text=text,num_labels=5)
    print(c())

    

    
    
    
    