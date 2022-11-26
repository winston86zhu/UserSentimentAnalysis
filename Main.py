from __future__ import division
from __future__ import print_function

import re, string, timeit
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.stem.porter import *
import json
import os
import nltk
import numpy as np
from builtins import str
from builtins import range
from past.utils import old_div
from builtins import object
import math
from tqdm import tqdm
stemmer = PorterStemmer()


def createSentences(content,stopwords):
    sent_word=[]
    sentences = nltk.sent_tokenize(content)
    for sent in sentences:
        words = nltk.word_tokenize(sent)
        temp = [stemmer.stem(w.lower()) for w in words if w not in string.punctuation]
        temp2 = [v for v in temp if v not in stopwords]
        if len(temp2)>0:
            sent_word.append(temp2)
    return sent_word



class createVocabularyFile:
    def create_stopwords(self):
        init_stopwords = [stemmer.stem(v) for v in stopwords.words('english')]
        additional_stopwords = ["'s","...","'ve","``","''","'m",'--',"'ll","'d"]
        self.stopwords = additional_stopwords + init_stopwords


    def createVocab(self):
        All_Contents = []
        i=0
        for hotel in self.corpus:
            print("File ."+str(i+1))
            for review in hotel.get("Reviews"):
                s= []
                for v in createSentences(review.get('Content'),self.stopwords):
                    s = v + s
                All_Contents = All_Contents + s
            i=i+1
        term_freq = FreqDist(All_Contents)
        Vocab = []
        Count = []
        VocabDict={}
        for k,v in term_freq.items():
            if v>5:
                Vocab.append(k)
                Count.append(v)
        self.Vocab = np.array(Vocab)[np.argsort(Vocab)].tolist()
        self.Count = np.array(Count)[np.argsort(Vocab)].tolist()
        self.VocabDict = dict(zip(self.Vocab,range(len(self.Vocab))))



stemmer = PorterStemmer()
def sentencesWithVocab(content,Vocab,VocabDict):
    sent_word=[]
    try:
        sentences = nltk.sent_tokenize(content)
    except:
        print('unable to tokenize: ' + content)
    try:
        for sent in sentences:
            words = nltk.word_tokenize(sent)
            temp = [stemmer.stem(w.lower()) for w in words if stemmer.stem(w.lower()) in Vocab]
            temp2 = [VocabDict.get(w) for w in temp]
            if len(temp2)>0:
                sent_word.append(temp2)
    except:
        print('unable to stem: ' + content)
    
    return sent_word


class Analyzer(object):
    def __init__(self, analyze, Vocab, VocabDict):
        self.analyze = FreqDist(analyze)
        self.unilength = len(self.analyze)
        self.label = -1   # initialize the label

class Review(object):
    def __init__(self, revData, Vocab, VocabDict):
        self.all = revData.get("all")
        self.revID = revData.get("revID")
        self.User = revData.get("User")
        self.Date = revData.get("Date")
        content = revData.get("Content")
        #self.Content = revData.get("text")
        # print("Review")
        # print("*******************************************************************************")
        # print(content)
        # print()
        # print()
        try:
            anayzedWord = sentencesWithVocab(content,Vocab,VocabDict)
            self.Analyzers = [Analyzer(analyze, Vocab, VocabDict) for analyze in anayzedWord]
            UniWord = {}
            for analyze in self.Analyzers:
                UniWord = list(UniWord) + list(analyze.analyze.keys())
                self.UniWord = np.array([w for w in UniWord])
                self.UniWord.sort()
                self.NumOfUniWord = len(self.UniWord)
        except:
            pass

    def labelreview(self):
        self.revlabel = -1
        for analyze in self.Analyzers:
            if analyze.label != -1:
                self.revlabel = 1
                break

    def analyzeDetails(self):
        self.NumOfAnnotatedAnalyzers = 0
        for analyze in self.Analyzers:
            if analyze.label != -1:
                self.NumOfAnnotatedAnalyzers = self.NumOfAnnotatedAnalyzers + 1


class Restaurant(object):
    def __init__(self, rest_data, Vocab, VocabDict):
        self.RestaurantID = rest_data.get('RestaurantInfo').get('RestaurantID')
        self.Name = rest_data.get('RestaurantInfo').get('Name')
        
        self.Address = rest_data.get('RestaurantInfo').get('Address')
        self.Reviews = [Review(review, Vocab,VocabDict) for review in rest_data.get("Reviews") ] 
        self.NumOfReviews = len(self.Reviews)

    def Calc_annotated_Reviews(self):
        self.NumOfAnnotatedReviews = 0
        for review in self.Reviews:
            if review.revlabel != -1:
                self.NumOfAnnotatedReviews = self.NumOfAnnotatedReviews + 1



def sent_aspect_match(analyze,aspects):  
    count = np.zeros(len(aspects))
    i=0
    for a in aspects:
        for w in list(analyze.analyze.keys()):
            if w in a:
                count[i]=count[i]+1
        i=i+1
    return count  


class Corpus(object):
    def __init__(self, corpus, Vocab, Count, VocabDict):
        self.Vocab = Vocab
        self.VocabDict = VocabDict
        self.VocabTF = Count
        self.V = len(Vocab)
        self.Aspect_Terms = []
        self.Restaurants = [Restaurant(rest, Vocab, VocabDict) for rest in tqdm(corpus)]
        self.NumOfRestaurants = len(corpus)



def listCombine(lists):  
    L=[]
    for l in lists:
        L=L+l
    return L


def ChisqTest(N, taDF, tDF, aDF):
    A = taDF 
    B = tDF - A 
    C = aDF - A
    D = N - A - B - C
    return N * ( A * D - B * C ) * ( A * D - B * C ) / aDF / ( B + D ) / tDF / ( C + D )

def statReview(review,aspect,Vocab):
    K = len(aspect)
    try:
        review.num_analyze_aspect_word = np.zeros((K,review.NumOfUniWord))
        review.num_analyze_aspect = np.zeros(K)
        review.num_anayzedWord = np.zeros(review.NumOfUniWord)
        review.num_analyze = 0
    except:
        pass

    try:    
        for analyze in review.Analyzers:
            if analyze.label != -1:  
                review.num_analyze = review.num_analyze + 1
                for l in analyze.label:
                    review.num_analyze_aspect[l] = review.num_analyze_aspect[l] + 1
                    for w in list(analyze.analyze.keys()):
                        z = np.where(w == review.UniWord)[0] 
                        review.num_anayzedWord[z] = review.num_anayzedWord[z] +1
                    for l in analyze.label:
                        for w in list(analyze.analyze.keys()):
                            z = np.where(w == review.UniWord)[0] 
                            review.num_analyze_aspect_word[l,z] = review.num_analyze_aspect_word[l,z]+1

    except:
        pass


class bootStrap(object):
    def sentence_label(self,corpus): 
        if len(self.Aspect_Terms)>0:
            for rest in corpus.Restaurants:
                for review in rest.Reviews:
                    for analyze in review.Analyzers:
                        count=sent_aspect_match(analyze,self.Aspect_Terms)
                        if max(count)>0:
                            s_label = np.where(np.max(count)==count)[0].tolist()
                            analyze.label = s_label
        else:
            pass


    def chiSquare(self,corpus):
        K=len(self.Aspect_Terms)
        V=len(corpus.Vocab)
        corpus.all_num_analyze_aspect_word = np.zeros((K,V))
        corpus.all_num_analyze_aspect = np.zeros(K)
        corpus.all_num_anayzedWord = np.zeros(V)
        corpus.all_num_analyze = 0
        Chi_sq = np.zeros((K,V))
        if K>0:
            for rest in corpus.Restaurants:
                for review in rest.Reviews:
                    try:
                        statReview(review,self.Aspect_Terms,corpus.Vocab)
                        corpus.all_num_analyze = corpus.all_num_analyze + review.num_analyze
                        corpus.all_num_analyze_aspect = corpus.all_num_analyze_aspect + review.num_analyze_aspect
                    except:
                        pass

                    try:
                        for w in review.UniWord:
                            z = np.where(w == review.UniWord)[0][0] # index, since the matrix for review is small
                            corpus.all_num_anayzedWord[w] = corpus.all_num_anayzedWord[w] + review.num_anayzedWord[z]
                            corpus.all_num_analyze_aspect_word[:,w] = corpus.all_num_analyze_aspect_word[:,w] + review.num_analyze_aspect_word[:,z]
                    except:
                        pass
     
            for k in range(K):
                try:
                    for w in range(V):
                        Chi_sq[k,w] = ChisqTest(corpus.all_num_analyze, corpus.all_num_analyze_aspect_word[k,w], corpus.all_num_anayzedWord[w], corpus.all_num_analyze_aspect[k])
                except:
                    pass
 
            self.Chi_sq = Chi_sq
        else:
            pass


def loadAspect(analyzer,filepath,VocabDict):
    analyzer.Aspect_Terms=[]
    f = open(filepath, "r")
    for line in f:
        aspect = [VocabDict.get(stemmer.stem(w.strip().lower())) for w in line.split(",")]
        analyzer.Aspect_Terms.append(aspect)
    f.close()
    print("Aspect Keywords loading completed")

def genAspect(analyzer, p, NumIter,c):
    for i in range(NumIter):
        analyzer.sentence_label(c)
        analyzer.chiSquare(c)
        t=0
        for cs in analyzer.Chi_sq:
            x = cs[np.argsort(cs)[::-1]] # descending order
            y = np.array([not math.isnan(v) for v in x], dtype=np.bool) # return T of F, force boolean
            words = np.argsort(cs)[::-1][y] #
            aspect_num = 0
            for w in words:
                if w not in listCombine(analyzer.Aspect_Terms):
                    analyzer.Aspect_Terms[t].append(w)
                    aspect_num = aspect_num +1
                if aspect_num > p:
                    break
            t=t+1
        print("Iteration " + str(i+1) +"/"+str(NumIter))

def save_Aspect_Keywords_to_file(analyzer,filepath,Vocab):
    try:
        f = open(filepath, 'w')
        for aspect in analyzer.Aspect_Terms:
            for w in aspect:
                try:
                    f.write(Vocab[w]+", ")
                except:
                    pass
            f.write("\n\n")
        f.close()
    except:
        pass

def genW_Review(analyzer,review,corpus):
    try:
        Nd = len(review.UniWord)
        K=len(analyzer.Aspect_Terms)
        review.W = np.zeros((K,Nd))
        for k in range(K):
            for w in range(Nd): 
                sum_row = sum(review.num_analyze_aspect_word[k])
                if  sum_row > 0:
                    review.W[k,w] = old_div(review.num_analyze_aspect_word[k,w],sum_row)
    except:
        print('unable to create W matrix for review')
        
def genW(analyzer,corpus):
    rest_num=0
    for rest in corpus.Restaurants:
        print("Creating W matrix for Restaurant "+str(rest_num+1))
        for review in rest.Reviews:
            genW_Review(analyzer,review,corpus)
        rest_num= rest_num+1

def genRating(analyzer,corpus,outputfolderpath):
    dir = outputfolderpath
    if not os.path.exists(dir):
        os.makedirs(dir)

    vocabfile = outputfolderpath+"vocab1.txt"
    f = open(vocabfile,"w")
    for w in corpus.Vocab:
        try:
            f.write(w.encode('utf8', 'replace') + ",")
        except:
            pass
    f.close()

    reviewfile = outputfolderpath + "revData.txt"
    f = open(reviewfile, 'w')
    for rest in corpus.Restaurants:
        for review in rest.Reviews:
            print(review)
            try:
                f.write(rest.RestaurantID)
                f.write(":")
                f.write(review.all)
                f.write(":")
                f.write(str(review.UniWord.tolist()))
                f.write(":")
                f.write(str(review.W.tolist()))
                f.write("\n")
            except:
                print('unable to produce data for rating for review')
    f.close()

def Stats(corpus):
    TotalNumOfRest = corpus.NumOfRestaurants
    TotalNumOfReviews = 0
    TotalNumOfAnnotatedReviews = 0
    AnalyzersperReviewList = []
    for rest in corpus.Restaurants:
        TotalNumOfReviews = TotalNumOfReviews + rest.NumOfReviews

        for review in rest.Reviews:
            try:
                review.analyzeDetails()  
                AnalyzersperReviewList.append(review.NumOfAnnotatedAnalyzers)
                review.labelreview()  # -1 or 1
                TotalNumOfAnnotatedReviews = TotalNumOfAnnotatedReviews + 1
            except:
                print('unable to calc summary stats for review.')
            
    AnalyzersperReviewList = np.array(AnalyzersperReviewList)
    m = np.mean(AnalyzersperReviewList)
    sd = np.std(AnalyzersperReviewList)
    print("TotalNumOfRest =" + str(TotalNumOfRest) +"\n")
    print("TotalNumOfReviews =" + str(TotalNumOfReviews) +"\n")
    print("TotalNumOfAnnotatedReviews =" + str(TotalNumOfAnnotatedReviews) +"\n")
    print("AnalyzersperReview=" + str(m) + "+-" + str(sd) + "\n")






def main():
    cv_obj = createVocabularyFile()
    loadfilepath = "./data/yelp_mp1_corpus.npy"
    (cv_obj.corpus,cv_obj.Vocab,cv_obj.Count,cv_obj.VocabDict)=np.load(loadfilepath,allow_pickle=True)

    # otherwise
    data = Corpus(cv_obj.corpus, cv_obj.Vocab, cv_obj.Count,cv_obj.VocabDict)

    BSanalyzer = bootStrap()
    loadfilepath = "./init_aspect_word.txt"
    loadAspect(BSanalyzer,loadfilepath,cv_obj.VocabDict)

    #### expand aspect keywords
    genAspect(BSanalyzer, 5, 5, data)

    savefilepath = './output/final_aspect_words.txt'
    save_Aspect_Keywords_to_file(BSanalyzer,savefilepath,cv_obj.Vocab)

    # genW(BSanalyzer,data)
    # W_outputfolderpath = "./output/"
    # genRating(BSanalyzer,data,W_outputfolderpath)

    # Stats(data)

if __name__ == "__main__":
    main()
