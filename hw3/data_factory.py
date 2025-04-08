import os
import re
import jieba
import chardet 
import numpy as np
import random

class Word2id_dict():
    # 词到id的字典类
    def __init__(self, data, max_vocab_size=10000, min_freq=10):
        # 输入的data: 一个列表, 每一项对应于一个文档的词列表
        # 输出的word2id_dict: 词到id的字典
        self.word2id_dict = {}
        self.id2word_dict = {}
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.build_dict(data)

    def build_dict(self, data):
        # 构建词到id的字典
        word_count = {}
        for doc in data:
            for word in doc:
                if word not in word_count:
                    word_count[word] = 1
                else:
                    word_count[word] += 1
        # 按照词频排序
        sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        # 构建词到id的字典
        for i, (word, freq) in enumerate(sorted_word_count):
            if i >= self.max_vocab_size:
                break
            if freq < self.min_freq:
                break
            self.word2id_dict[word] = i
            self.id2word_dict[i] = word
        # 添加'<unk>'到词典中
        self.word2id_dict['<unk>'] = len(self.word2id_dict)
        self.id2word_dict[len(self.id2word_dict)] = '<unk>'

    def word2id(self, word):
        # 输入单词, 返回对应的id
        if word in self.word2id_dict:
            return self.word2id_dict[word]
        else:
            return self.word2id_dict['<unk>']

    def id2word(self, id):
        # 输入id, 返回对应的词
        if id in self.id2word_dict:
            return self.id2word_dict[id]
        else:
            return '<unk>'
        
    def __len__(self):
        # 返回词典大小
        return len(self.word2id_dict)


def convert_corpus_to_id(documents, word2id_dict):
    # 使用一个循环，将语料中的每个词替换成对应的id，以便于神经网络进行处理
    corpus = []
    for doc in documents:
        corpus.append([word2id_dict.word2id(word) for word in doc])
    return corpus

class My_DataLoader():
    # 将各种方法封装成数据获取类
    def __init__(self):
        path = '../corpus/'
        self.inf = path + 'inf.txt'
        self.stopwords_path = path +'cn_stopwords.txt'

    def get_name_list(self, inf_path):
        with open(inf_path) as f:
            data = f.read().split(',')
        data = [data[i] + '.txt' for i in range(len(data))]
        return data

    def read_stopwords(self, stopwords_path='../corpus/cn_stopwords.txt'):
        with open(stopwords_path,'r',encoding='utf-8') as f:
            stopwords = f.readlines()
        f.close()
        stopwords = [word.strip() for word in stopwords]
        return stopwords
    
    def read_words(self, data_path='../corpus/白马啸西风.txt', stopwords_path='../corpus/cn_stopwords.txt'):
        ### 读取中文文档数据, 使用jieba分词, 并移除所有的stopwords
        stop_words = self.read_stopwords(stopwords_path)
        with open(data_path, 'rb') as f:
            data = f.read()
        encoding = chardet.detect(data)['encoding']
        # print("encoding:", encoding)
        with open(data_path, 'r', encoding=encoding, errors='ignore') as f:
            data = f.read()
        f.close()
        # print("before cut:", data)
        data = re.sub(r'\n+', '\n', data) # remove multiple newlines
        data = re.sub(r'\s+','', data) # remove extra whitespace
        data = jieba.cut(data) # cut into words
        data = [word for word in data if word not in stop_words] # remove stopwords
        print("total words:", len(data))
        return data
    

    def read_words_new(self, data_path='../corpus/白马啸西风.txt', stopwords_path='../corpus/cn_stopwords.txt'):
        ### 读取中文文档数据, 使用jieba分词, 并移除所有的stopwords
        stop_words = self.read_stopwords(stopwords_path)
        with open(data_path, 'rb') as f:
            data = f.read()
        encoding = chardet.detect(data)['encoding']
        # print("encoding:", encoding)
        with open(data_path, 'r', encoding=encoding, errors='ignore') as f:
            data = f.read()
        f.close()
        # print("before cut:", data)
        # 按照句子将data进行split
        data = re.sub(r'\n+', '\n', data) # remove multiple newlines
        data = re.sub(r'\s+','', data) # remove extra whitespace
        data = re.split(r'[。！？；]', data)
        print("total sentences:", len(data))
        # 对每个句子分别进行分词，输出应当是一个分词列表的 列表
        data = [jieba.cut(sentence) for sentence in data]
        # 合并列表
        data = [word for sentence in data for word in sentence if word not in stop_words] # remove stopwords
        print("total sentences:", len(data))
        return data
    
    def read_files_words(self, dir_path = '../corpus/',inf_path='../corpus/inf.txt', stopwords_path='../corpus/cn_stopwords.txt'):
        # 读取列表中所有文档的词汇
        names = self.get_name_list(inf_path)
        data = []
        for name in names:
            print("file reading:", dir_path+name)
            words = self.read_words(dir_path+name, stopwords_path)
            data.append(words)
        print("files read successfully!")
        return data
    
    def read_word(self, data_path='../corpus/白马啸西风.txt', stopwords_path='../corpus/cn_stopwords.txt'):
        # 读取单个文档的字, 并移除stopwords
        stop_words = self.read_stopwords(stopwords_path)
        with open(data_path, 'rb') as f:
            data = f.read()
        encoding = chardet.detect(data)['encoding']
        # print("encoding:", encoding)
        with open(data_path, 'r', encoding=encoding, errors='ignore') as f:
            data = f.read()
        f.close()
        # print("before cut:", data)
        data = re.sub(r'\n+', '\n', data) # remove multiple newlines
        data = re.sub(r'\s+','', data) # remove extra whitespace
        # cut into word
        data = list(data)
        data = [word for word in data if word not in stop_words] # remove stopwords
        print("total characters:", len(data))
        return data

    def read_files_word(self, dir_path = '../corpus/',inf_path='../corpus/inf.txt', stopwords_path='../corpus/cn_stopwords.txt'):
        # 读取列表中所有文档的词汇
        names = self.get_name_list(inf_path)
        data = []
        for name in names:
            print("file reading:", dir_path+name)
            words = self.read_word(dir_path+name, stopwords_path)
            data.append(words)
        print("files read successfully!")
        return data
    


    def Sample(self, data, K=[20, 100, 500, 1000, 3000]):
        # 对单个文档的字/词进行1次随机位置采样; 每个段落的长度为K中的随机值
        # 输入的data: 字/词列表, 
        # 输出的sample: 采样结果
        random_K = random.choice(K)
        # 随机选择一个段落长度
        random_start = random.randint(0, len(data)-3000)
        random_end = random_start + random_K
        sample = data[random_start:random_end]
        return sample

    def Sample_files(self, data, N=1000, K=[20, 100, 500, 1000, 3000],
                     inf_path='../corpus/inf.txt'):
        # 对一组文件中的每个进行采样, 采样方式见Sample()函数
        # 输入的data: 一个二维列表 , 每一项对应于一个文件的字/词列表
        # 输出的samples, labels: 都是二维列表, 每一项对应于一个文件的采样后的列表 和 标签列表
        names = self.get_name_list(inf_path)
        samples = []
        labels = []
        # for i in range(len(names)):
        #     # 逐个文件进行处理
        #     print("file to sample:", names[i])
        #     sample, label = self.Sample(data[i], label=names[i], N=N, K=K)
        #     samples.append(sample)
        #     labels.append(label)
        for i in range(N):
            # 对所有文件总共采样1000个
            random_f = random.randint(0, len(names)-1)
            sample = self.Sample(data[random_f], K=K)
            samples.append(sample)
            labels.append(names[random_f])
        print("samples and labels generated!",len(samples), len(labels))
        return samples, labels
    
    def sample_sentence(self, data_path='../corpus/白马啸西风.txt', num=100):
        with open(data_path, 'rb') as f:
            data = f.read()
        encoding = chardet.detect(data)['encoding']
        # print("encoding:", encoding)
        with open(data_path, 'r', encoding=encoding, errors='ignore') as f:
            data = f.read()
        f.close()
        
        # data = re.sub(r'\n+', '\n', data) # remove multiple newlines
        # data = re.sub(r'\s+','', data) # remove extra whitespace
        data = data.split('\n')
        # 随机选择num个句子
        print("total sentences:", len(data))
        random_index = random.sample(range(len(data)), num)
        data = [data[i] for i in random_index]
        print("selected sentences:", len(data))
        return data
    
    def process_sentence(self, sentence, stopwords_path='../corpus/cn_stopwords.txt'):
        # 对句子进行分词, 并移除stopwords
        stop_words = self.read_stopwords(stopwords_path)
        # sentence = re.sub(r'\n+', '\n', sentence) # remove multiple newlines
        sentence = re.sub(r'\s+','', sentence) # remove extra whitespace
        
        sentence = list(jieba.cut(sentence))
        sentence = [word for word in sentence if word not in stop_words] # remove stopwords
        return sentence


# 文件读取示范
if __name__ == '__main__':
    # 使用手动封装好的DataLoader
    data_loader = My_DataLoader()
    # 获取文件名列表
    # path = '../corpus/'
    # inf = path + 'inf.txt'
    # names = DataLoader.get_name_list(inf)
    # print(names)

    # 获取stopwords
    # stopwords_path = path +'cn_stopwords.txt'
    # stop_words = DataLoader.read_stopwords(stopwords_path)
    # print(stop_words[:10])
    # print(len(stop_words))

    # 读取第一个文档
    # print("file to read:",path+names[0])
    # data = DataLoader.read_words(path+names[0], stopwords_path)
    # print(len(data))
    # print(data[:10])
    # print(data[-10:])

    # 遍历读取所有文档
    # for name in names:
    #     print("file to read:", path+name)
    #     data = DataLoader.read_words(path+name, stopwords_path)
    #     print(len(data))
    #     print(data[:10])
    #     print(data[-10:])

    ####### 封装后
    # 读取单个文档的字
    # data = data_loader.read_word(data_path='../corpus/白马啸西风.txt', stopwords_path='../corpus/cn_stopwords.txt')
    # print(len(data))
    # print(data[:100])
    # print(data[-100:])
    # data = data_loader.Sample(data=data, label='白马啸西风')
    # print(len(data))
    data = data_loader.read_words_new(data_path='../corpus/白马啸西风.txt', stopwords_path='../corpus/cn_stopwords.txt')
    # print(len(data))
    print(data[:10])
    # print(data[-100:])
    # data = data_loader.Sample(data=data, label='白马啸西风')
    # print(len(data))
    
    
    # 读取所有文档的词汇
    # data = data_loader.read_files_words(dir_path = '../corpus/',inf_path='../corpus/inf.txt', stopwords_path='../corpus/cn_stopwords.txt')
    # print(len(data))
    # print(len(data[0]))
    # print(data[0][:10])
    # print(data[0][-10:])
    # samples, labels = data_loader.Sample_files(data=data)
    

    
