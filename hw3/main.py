import numpy as np
from data_factory import *
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm
import gensim
from gensim.models import Word2Vec
rcParams['figure.figsize'] = 10, 10
rcParams['font.family'] = '宋体'

import torch
import os
import jieba
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# 计算段落之间的距离
def paragraph_distance(paragraphs1, paragraphs2):
    '''
    计算两个段落列表内部各个句子之间和互相之间的距离
    :param paragraphs1: 第一个段落列表;
    :param paragraphs2: 第二个段落列表;'''
    process_paragraphs1 = [Data_processer.process_sentence(sentence) for  sentence in paragraphs1]
    process_paragraphs2 = [Data_processer.process_sentence(sentence) for  sentence in paragraphs2]
    mean_distance_1 = 0 # 段落1内部各个句子的平均距离
    mean_distance_2 = 0 # 段落2内部各个句子的平均距离
    mean_distance_1_2 = 0 # 段落1和2的平均距离
    
    count1_2 = 0
    for i in range(len(process_paragraphs1)):
        for j in range(len(process_paragraphs2)):
            
            distance = model.wv.wmdistance(process_paragraphs1[i], process_paragraphs2[j])
            mean_distance_1_2 += distance
    count1_2 = len(process_paragraphs1) * len(process_paragraphs2)
    mean_distance_1_2 /= count1_2
    print(f"Mean distance between paragraphs 1 and 2: {mean_distance_1_2}")
    
    count1 = 0
    for i in range(len(process_paragraphs1)):
        for j in range(len(process_paragraphs1)):
            if i != j:
                distance = model.wv.wmdistance(process_paragraphs1[i], process_paragraphs1[j])
                mean_distance_1 += distance
    count1 = len(process_paragraphs1) * (len(process_paragraphs1) - 1)
    mean_distance_1 /= count1
    print(f"Mean distance among paragraphs 1: {mean_distance_1}")

    count2 = 0
    for i in range(len(process_paragraphs2)):
        for j in range(len(process_paragraphs2)):
            if i != j:
                distance = model.wv.wmdistance(process_paragraphs2[i], process_paragraphs2[j])
                mean_distance_2 += distance
    count2 = len(process_paragraphs2) * (len(process_paragraphs2) - 1)
    mean_distance_2 /= count2
    print(f"Mean distance among paragraphs 2: {mean_distance_2}")
    
    return mean_distance_1_2, mean_distance_1, mean_distance_2


def clustering(paragraphs):
    '''
    聚类分析
    :param paragraphs: 输入段落列表;
    :return: 聚类结果列表;
    '''
    # 1. 预处理 将文本转化为向量形式。
    data_vec = []
    word_list = []
    for text in paragraphs:
        words = jieba.lcut(text)
        vec = np.zeros(100)
        for word in words:
            if word in model.wv:
                if word not in word_list:
                    word_list.append(word)
                    vec = model.wv[word]
                    data_vec.append(vec)

   
    # 2. K-Means算法 使用K-Means算法对词向量进行聚类
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.cluster import KMeans
    # kmeans = KMeans(n_clusters=15, n_init=100)
    kmeans = MiniBatchKMeans(n_clusters=20, batch_size=10, max_iter=500)
    kmeans.fit(data_vec)

    # 3. 输出聚类结果,按照聚类标签的大小进行排列再输出. 按照: center: (word1, word2...)的格式输出
    cluster_result = sorted(zip(word_list, kmeans.labels_), key=lambda x:x[1])

    for i in range(20):
        print(f"Cluster {i}:")
        str_word = ""
        for j, (word, label) in enumerate(cluster_result):
            if label == i:
                str_word += word + " "
        print(str_word)
                
        
    
    
    # 4. 可视化, 根据聚类结果word和label 结果的可视化图
    # fig, ax = plt.subplots()
    # ax.scatter([x[1] for x in cluster_result], [x[0] for x in cluster_result], s=100)
    # ax.set_xlabel("Cluster Label")
    # ax.set_ylabel("Word")
    # plt.show()


def clustering_words(words_list):
    '''
    聚类分析
    :param paragraphs: 输入段落列表;
    :return: 聚类结果列表;
    '''
    # 1. 预处理 将文本转化为向量形式。
    data_vec = []
    word_list = []
    for word in words_list:
        if word in model.wv:
            if word not in word_list:
                word_list.append(word)
                vec = model.wv[word]
                data_vec.append(vec)

   
    # 2. K-Means算法 使用K-Means算法对词向量进行聚类
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.cluster import KMeans
    # kmeans = KMeans(n_clusters=15, n_init=100)
    kmeans = MiniBatchKMeans(n_clusters=3, batch_size=5, max_iter=20)
    kmeans.fit(data_vec)

    # 3. 输出聚类结果,按照聚类标签的大小进行排列再输出. 按照: center: (word1, word2...)的格式输出
    cluster_result = sorted(zip(word_list, kmeans.labels_), key=lambda x:x[1])

    for i in range(3):
        print(f"Cluster {i}:")
        str_word = ""
        for j, (word, label) in enumerate(cluster_result):
            if label == i:
                str_word += word + " "
        print(str_word)
                
    # 4. 可视化, 根据聚类结果word和label 结果的可视化图
    # fig, ax = plt.subplots()
    # ax.scatter([x[1] for x in cluster_result], [x[0] for x in cluster_result], s=100)
    # ax.set_xlabel("Cluster Label")
    # ax.set_ylabel("Word")
    # plt.show()


if __name__ == '__main__':
    mode = 'test'
    epochs = 100
    window = 15
    min_count = 1
    vec_size = 100
    model_path = f'./My_model_epochs_{epochs}_window_{window}_min_count{min_count}'
    print(model_path)
    
    if mode == 'train':
        split_mode = 'word'

        # 加载数据
        corpus_loader = My_DataLoader()
        if split_mode == 'word':
            # 以词为单位
            data = corpus_loader.read_files_words(dir_path = './data/',inf_path='./data/inf.txt', stopwords_path='./cn_stopwords.txt')
        elif split_mode == 'char':
            # 以字为单位
            data = corpus_loader.read_files_word(dir_path = './data/',inf_path='./data/inf.txt', stopwords_path='./cn_stopwords.txt')
        
        sentences = data
        # model = gensim.models.Word2Vec(sentences, min_count=1)
        # model = Word2Vec(LineSentence(inp), size=400, window=5, min_count=5, workers=multiprocessing.cpu_count())
        # model= Word2Vec(sentences, vector_size=400, window=5, min_count=5)
        
        model = Word2Vec(min_count=min_count, epochs=epochs, vector_size=vec_size, window=window)
        model.build_vocab(sentences)
        print(model.corpus_count)
        model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)  # train word vectors
        model.save(model_path)

    elif mode == 'test':
        Data_processer = My_DataLoader()
        model = Word2Vec.load(model_path)

        ####### E1
        testwords = ["段誉","苗人凤", "郭啸天","杀","喝","骂","书","剑","刀"]
        print("Check the top 5 similar words")
        for i in range(len(testwords)):
            res = model.wv.most_similar(testwords[i], topn=5)
            # """Find the top-N most similar keys."""
            # 希望相似度的值打印出来保留小数点3位
            for j in range(len(res)):
                res[j] = (res[j][0], round(res[j][1], 3))
            print(f"for {testwords[i]}: ", res)


        print("word similarity between '韦小宝' and '郭靖' is:", model.wv.similarity("韦小宝", "郭靖"))
        print("word similarity between '不错' and '妙' is:", model.wv.similarity("不错", "妙"))
        print("word similarity between '笑' and '说道' is:", model.wv.similarity("说道", "笑"))
        

        ###### E2
        # """Compute the Word Mover's Distance between two documents."""
        paragraphs1 = ["总舵主缓缓的道：“你可知我们天地会是干什么的？”韦小宝道：“天地会反清复明，帮汉人，杀鞑子。”总舵主点头道：“正是！你愿不愿意入我天地会做兄弟？”"
                        "韦小宝喜道：“那可好极了。”在他心目中，天地会会众个个是真正英雄好汉，想不到自己也能成为会中兄弟，又想：“连茅大哥也不是天地会的兄弟，我难道比他还行？”说道：“就怕．．．就怕我够不上格。”霎时间眼中放光，满心尽是患得患失之情，只觉这笔天外飞来的横财，多半不是真的，不过总舵主跟自己开开玩笑而已。",
                        "总舵主道：“你要入会，倒也可以。只是我们干的是反清复明的可事，以汉人的江山为重，自己的身家性命为轻。再者，会里规知严得很，如果犯了，处罚很重，你须得好好想一想。”韦小宝道：“不用想，你有什么规矩，我守着便是。总舵主，你如许我入会，我可快活死啦。”总舵主收起了笑容，正色道：“这是极要紧的大事，生死攸关，可不是小孩子们的玩意。”韦小宝道：“我当然知道。我听人说，天地会行侠仗义，做得都是惊天动地的大事，怎么会是小孩子的玩意？”",
                        "总舵主微笑道：“知道了就好，本会入会时有誓词三十六条，又有禁十刑的严规。”说到这里，脸色沉了下来，道：“这些规矩，你眼前年纪还小，还用不上，不过其中有一条：‘凡我兄弟，须当信实为本，不得谎言诈骗。’这一条，你能办到么？”",
                        "韦小宝微微一怔，道：“对你总舵主，我自然不敢说谎。可是对其余兄弟，难道什么事也都要说真话？”总舵主道：“小事不论，只论大事。”韦小宝道：“是了。好比和会中兄弟们赌钱，出手段骗可不不可以？”"]
                      
        paragraphs2 = ["段正淳送了保定帝和黄眉僧出府，回到内室，想去和王妃叙话。不料刀白凤正在为他又多了个私生女儿钟灵而生气，闭门不纳。段正淳在门外哀告良久，刀白凤发话道：“你再不走，我立刻回玉虚观去。”",
                       "段正淳无奈，只得到书房闷坐，想起钟灵为云中鹤掳去，不知钟万仇与南海鳄神是否能救得回来，褚万里等出去打探讯息，迄未回报，好生放心不下。从怀中摸准出甘宝宝交来的那只黄金钿盒，瞧着她所写那几行蝇头细字，回思十七年前和她欢聚的那段销魂蚀骨的时光，再想像她苦候自己不至而被迫与钟万仇成婚的苦楚，不由得心中大痛：“那时她还只是个十七岁的小姑娘，她父亲和后母待她向来不好，腹中怀了我的孩儿，却教她如何做人？”",
                       "越想越难过，突然之间，想起了先前刀白凤在席上对华司徒所说的那名话来：“这条地道通入钟夫人的居室，若不堵死，就怕咱们这里有一位仁兄，从此天天晚上要去钻地道。”当即召来一名亲兵，命他去把华司徒手下两名得力家将悄悄传来，不可泄漏风声。",
                       "段誉在书房中，心中翻来覆去的只是想着这些日子中的奇遇：跟木婉清订了夫妇之约，不料她竟是自己妹子，岂知奇上加奇，钟灵竟然也是自己妹子。钟灵被云中鹤掳去，不知是否已然脱险，实是好生牵挂。又想慕容博夫妇钻研‘凌波微步’，不知跟洞中的神仙姊姊是否有什么瓜葛？难道他们是‘逍遥派’的弟子？神仙姊姊吩咐我去杀了他们？这对夫妇武功这样高强，要我去杀了他们，那真是天大的笑话了。"]
        paragraph_distance(paragraphs1, paragraphs2)

        sample_sentence1 = Data_processer.sample_sentence("./data/鹿鼎记.txt", num=100)
        sample_sentence2 = Data_processer.sample_sentence("./data/天龙八部.txt", num=100)
        paragraph_distance(sample_sentence1, sample_sentence2)
        

        ###### E3
        # 用这里的word2vec模型实现一组词的聚类任务
        sample_sentence_cluster = Data_processer.sample_sentence("./data/天龙八部.txt", num=10)
        clustering(sample_sentence_cluster)

        words_list = ["西域", "中土", "丐帮", "中原", "少林", "输赢", "自知之明", "甘拜下风",  "一拳", "拍","撞","打倒"]
        clustering_words(words_list)


    
