from numpy import *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWord2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if(word in vocabList):
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def bagOfWords2VecNN(VocabList, inputSet):
    returnVec = [0] * len(VocabList)
    for word in inputSet:
        if (word in VocabList):
            returnVec[VocabList.index(word)] += 1
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  #多少个文档
    numwords = len(trainMatrix[0])   #每个文档的词向量长度
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numwords) ; p1Num = ones(numwords)  #计算类别0（或1）中每个词出现的次数，为一个数组，大小为词向量大小
    p0Denom = 2.0 ; p1Denom = 2.0   #计算类别0（或1)有多少个词
    for i in range(numTrainDocs):
        if(trainCategory[i] == 1):   #这个文档属于类别1
            p1Num += trainMatrix[i]  #所有词向量中相应的词出现次数+1
            p1Denom += sum(trainMatrix[i])  #类别1中词的数量相应增加，增加的个数为文档中包含的词的数量
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)   #数组中每个元素表示p(wi|c1)的值
    p0Vect = log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWord2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(trainMat, listClasses)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWord2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWord2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []; classList = []; fullText = [] #第一个记录所有文档被分割后的词，一个词一个列表，第二个记录标签
                                                #第三个记录所有文档中的词
    for i in range(1, 26):
        wordList = textParse(open('D:/Python/untitled1/DataSet/email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('D:/Python/untitled1/DataSet/email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50)); testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWord2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0.0
    for docuIndex in testSet:
        wordVector = setOfWord2Vec(vocabList, docList[docuIndex])
        if(classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docuIndex]):
            print(docList[docuIndex])
            errorCount += 1
    print('the error rate is: ', errorCount/len(testSet))
