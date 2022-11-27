import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
import os
import logging
import torch


def parse_args():  #配置参数
        parser = argparse.ArgumentParser()

        parser.add_argument('--seed', type=int, default=23)
        parser.add_argument('--random_state', type=int, default=23)
        parser.add_argument('--data_file', type=str, default='./datasets/raw_5.txt') #样本数据集
        parser.add_argument('--valid_size', type=float, default=0.2)#测试集大小
        parser.add_argument('--model_name', type=str, default='') #模型名称  1  使用其他的bert模型
        parser.add_argument('--output_dir', type=str, default='save_model') 
        parser.add_argument('--save_name', type=str, default='bert') #保存模型
        parser.add_argument('--batch_size', type=int, default=32)  #batch  3
        parser.add_argument('--lr', type=float, default=5e-4)   #4
        parser.add_argument('--weight_decay', type=float, default=0.01) #权重衰减
        parser.add_argument('--num_epochs', type=int, default=10)  #2
        parser.add_argument('--warmup_ratio', type=float, default=0.06)   #预学习率  （查）
        parser.add_argument('--logging_dir', type=str, default='logs')  #保存日志
        
        args = parser.parse_args()

        return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def load_data(data_file):
    contents = []
    labels = []
    with open(data_file, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            lin = line.strip()
            if not lin:
                continue
            content, label = lin.split('\t')
            contents.append(content)
            labels.append(label)

    return contents,labels  #得到的是两个列表

def compute_metrics(pred):  #模型结果评分
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    accuracy = accuracy_score(labels, preds)
    metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    return metrics


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings  
        #{'input_ids':句子中单词的token，'token_type_ids'区分上下句，'attention_mask'：参与attentio部分padding部分是否参与attention}
        self.labels = [int(label) for label in labels]

    def __getitem__(self, idx): #使用词查询索引
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}  #encoding 字典
        # print(self.labels)
        item['labels'] = torch.tensor(self.labels[idx])
        #print(item)
        return item

    def __len__(self):
        return len(self.labels)

logging.basicConfig(
    level=logging.INFO,
    filename=f'./logs/bert.log',
    filemode='w'
)
logger = logging.getLogger(__name__)   #记录日志

def main():
    args = parse_args() #传入参数
    set_seed(args.seed)  #设置种子
    device = torch.device('cuda')

    texts, labels = load_data(args.data_file)  #arg.data_file 里面导入的时数据的路径  text，label两个列表
 
    train_texts, valid_texts, train_labels, valid_labels = train_test_split(texts, labels, test_size=args.valid_size, random_state=args.random_state,stratify=labels)  #切分数据集  改过2 去掉了最后一个参数
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)#分词器   
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=5).to(device) #bert预训练模型
    train_encodings = tokenizer(train_texts, padding='max_length', truncation=True, max_length=512)  #训练集样本编码
    valid_encodings = tokenizer(valid_texts, padding='max_length', truncation=True, max_length=512)
    #tokenizer 返回的结果是一个字典  对于单句的输入 返回的是{'input_ids':句子中单词的token(对应到bert字典里的索引用与embeding），'token_type_ids'区分上下句，'attention_mask'：参与attentio部分padding部分是否参与attention}
    train_dataset = Dataset(train_encodings, train_labels)
    valid_dataset = Dataset(valid_encodings, valid_labels)

    training_args = TrainingArguments(    #用于设计参数的类
            output_dir=args.output_dir,    #模型预测和检查点的输出目录。必须声明的字段
            overwrite_output_dir=True,   # 如果为True，则覆盖输出目录的内容。使用此继续训练，如果`output_dir`指向检查点目录。
            #do_train（：obj：`bool`，`可选`，默认为：obj：`False`）：是否进行训练。
            #do_eval（：obj：`bool`，`optional`，默认为：obj：`False`）：是否在验证集上运行评估。
            #do_predict（：obj：`bool`，`optional`，默认为：obj：`False`）：是否在测试集上运行预测
            evaluation_strategy='epoch',
            per_device_train_batch_size=args.batch_size, 
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,  #要应用的权重衰减（如果不为零）。
            num_train_epochs=args.num_epochs,
            lr_scheduler_type='linear',
            warmup_ratio=args.warmup_ratio,
            logging_strategy='no',
            save_strategy='epoch',
            save_total_limit=1,
            report_to='none',
            seed=args.seed,
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
        )

    trainer = Trainer(    #trainer类用于训练何评估自己的数据和模型.不想使用Transformers来训练别人的预训练模型，而是想用来训练自己的模型，并且不想写训练过程代码。这时，我们可以按照一定的要求定义数据集和模型，就可以使用Trainer类来直接训练和评估模型，不需要写那些训练步骤了。
            model=model,       #上面实例化的bert模型           
            args=training_args,         #上面设计的训练所使用的参数         
            compute_metrics=compute_metrics,  #模型评估
            train_dataset=train_dataset,        #trainer在训练的时候会将dataset中的数据按照 对应的键值传入  前面的就是text的编码 后面的时候labels
            eval_dataset=valid_dataset,          #验证集
            tokenizer=tokenizer,               #分词方法
        )
        
    trainer.train()  #传入训练数据开始训练

    for log in trainer.state.log_history:   #读取训练日志  每一个epoch的训练结果
            if 'eval_loss' in log:
                epoch = int(log['epoch'])
                eval_loss = log['eval_loss']
                eval_accuracy = log['eval_accuracy']
                eval_f1 = log['eval_f1']
                eval_precision = log['eval_precision']
                eval_recall = log['eval_recall']
                logger.info(f'epoch: {epoch:02}')
                logger.info(f'\teval_loss: {eval_loss:.3f} | eval_accuracy: {eval_accuracy*100:.2f}% | eval_f1: {eval_f1*100:.2f}% | eval_precision: {eval_precision*100:.2f}% | eval_recall: {eval_recall*100:.2f}%')
    logger.info('')

    model_path = os.path.join(args.output_dir, args.save_name)
    model.save_pretrained(model_path)  #保存模型
    tokenizer.save_pretrained(model_path)

if __name__ == '__main__': 
    main()