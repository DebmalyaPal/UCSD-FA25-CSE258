#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import defaultdict
import math
import scipy.optimize
import numpy
import string
from sklearn import linear_model
import random


# In[2]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[3]:


def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r


# In[4]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom > 0:
        return numer/denom
    return 0


# In[5]:


##################################################
# Rating prediction                              #
##################################################


# In[6]:


def getGlobalAverage(trainRatings):
    total = sum(trainRatings)
    count = len(trainRatings)
    return total / count if count > 0 else 0


# In[7]:


def trivialValidMSE(ratingsValid, globalAverage):
    squared_error_sum = 0
    for (_, _, actual_rating) in ratingsValid:
        error = actual_rating - globalAverage
        squared_error_sum += error ** 2
    return squared_error_sum / len(ratingsValid) if ratingsValid else 0


# In[8]:


def alphaUpdate(ratingsTrain, alpha, betaU, betaI, lamb):
    # Update equation for alpha
    total = 0.0
    for (u, i, r) in ratingsTrain:
        total += r - (betaU[u] + betaI[i])
    return total / len(ratingsTrain)


# In[9]:


def betaUUpdate(ratingsPerUser, alpha, betaU, betaI, lamb):
    # Update equation for betaU
    newBetaU = {}
    for u in ratingsPerUser:
        total = 0.0
        for (i, r) in ratingsPerUser[u]:
            total += r - (alpha + betaI[i])
        newBetaU[u] = total / (lamb + len(ratingsPerUser[u]))
    return newBetaU


# In[10]:


def betaIUpdate(ratingsPerItem, alpha, betaU, betaI, lamb):
    # Update equation for betaI
    newBetaI = {}
    for i in ratingsPerItem:
        total = 0.0
        for (u, r) in ratingsPerItem[i]:
            total += r - (alpha + betaU[u])
        newBetaI[i] = total / (lamb + len(ratingsPerItem[i]))
    return newBetaI


# In[12]:


def msePlusReg(ratingsTrain, alpha, betaU, betaI, lamb):
    # Compute the MSE and the mse+regularization term
    mse = 0
    for (u, i, r) in ratingsTrain:
        pred = alpha + betaU[u] + betaI[i]
        mse += (r - pred) ** 2
    mse /= len(ratingsTrain)

    regularizer = sum(bu**2 for bu in betaU.values()) + sum(bi**2 for bi in betaI.values())
    mseReg = mse + lamb * regularizer
    return mse, mseReg


# In[13]:


def validMSE(ratingsValid, alpha, betaU, betaI):
    # Compute the MSE on the validation set
    mse = 0
    for (u, i, r) in ratingsValid:
        bu = betaU.get(u, 0.0)
        bi = betaI.get(i, 0.0)
        pred = alpha + bu + bi
        mse += (r - pred) ** 2
    return mse / len(ratingsValid) if ratingsValid else 0



# In[ ]:


def goodModel(ratingsTrain, ratingsPerUser, ratingsPerItem, alpha, betaU, betaI):
    # Improve upon your model from the previous question (e.g. by running multiple iterations)
    lamb = 1.0
    maxIter = 50
    tol = 1e-4
    prev_mseReg = float('inf')

    # Initialize missing entries to zero just in case
    for (u, i, r) in ratingsTrain:
        if u not in betaU:
            betaU[u] = 0
        if i not in betaI:
            betaI[i] = 0

    for it in range(maxIter):
        alpha = alphaUpdate(ratingsTrain, alpha, betaU, betaI, lamb)
        betaU = betaUUpdate(ratingsPerUser, alpha, betaU, betaI, lamb)
        betaI = betaIUpdate(ratingsPerItem, alpha, betaU, betaI, lamb)
        mse, mseReg = msePlusReg(ratingsTrain, alpha, betaU, betaI, lamb)

        if abs(prev_mseReg - mseReg) < tol:
            break
        prev_mseReg = mseReg

    return alpha, betaU, betaI

# In[ ]:


def writePredictionsRating(alpha, betaU, betaI):
    # Write your predictions to a file that you can submit
    predictions = open("predictions_Rating.csv", 'w')
    for l in open("pairs_Rating.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u,b = l.strip().split(',')
        bu = 0
        bi = 0
        if u in betaU:
            bu = betaU[u]
        if b in betaI:
            bi = betaI[b]
        _ = predictions.write(u + ',' + b + ',' + str(alpha + bu + bi) + '\n')

    predictions.close()


# In[ ]:


##################################################
# Read prediction                                #
##################################################


# In[ ]:


def generateValidation(allRatings, ratingsValid):
    # Using ratingsValid, generate two sets:
    # readValid: set of (u,b) pairs in the validation set
    # notRead: set of (u,b') pairs, containing one negative (not read) for each row (u) in readValid  
    # Both should have the same size as ratingsValid
    readValid = set()
    notRead = set()

    # Build a mapping of books read by each user in the training data
    booksReadByUser = defaultdict(set)
    allBooks = set()
    for (u, b, r) in allRatings:
        booksReadByUser[u].add(b)
        allBooks.add(b)

    allBooks = list(allBooks)
    for (u, b, r) in ratingsValid:
        readValid.add((u, b))
        # sample one random book not read by this user
        for _ in range(100):  # avoid infinite loop
            neg = random.choice(allBooks)
            if neg not in booksReadByUser[u]:
                notRead.add((u, neg))
                break

    return readValid, notRead


# In[ ]:


def baseLineStrategy(mostPopular, totalRead):
    return1 = set()

    # Compute the set of items for which we should return "True"
    # This is the same strategy implemented in the baseline code for Assignment 1
    return1 = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > totalRead/2: break

    return return1


# In[ ]:


def improvedStrategy(mostPopular, totalRead):
    return1 = set()

    # Same as above function, just find an item set that'll have higher accuracy
    # A stronger baseline: use a lower popularity cutoff (more items predicted as read)
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        # predict True for top ~70% of reads instead of 50%
        if count > totalRead * 0.7:
            break

    return return1


# In[ ]:


def evaluateStrategy(return1, readValid, notRead):

    # Compute the accuracy of a strategy which just returns "true" for a set of items (those in return1)
    # readValid: instances with positive label
    # notRead: instances with negative label
    # Compute accuracy of a strategy that predicts True for items in return1
    correct = 0
    total = 0

    for x in readValid:
        total += 1
        if x[1] in return1:
            correct += 1

    for x in notRead:
        total += 1
        if x[1] not in return1:
            correct += 1

    acc = correct / total if total > 0 else 0
    return acc


# In[ ]:


def jaccardThresh(u,b,ratingsPerItem,ratingsPerUser):

    # Compute the similarity of the query item (b) compared to the most similar item in the user's history
    # Return true if the similarity is high or the item is popular
    maxSim = 0
    userItems = ratingsPerUser.get(u, [])
    itemUsers = ratingsPerItem.get(b, [])

    users_b = set([x for x in itemUsers])
    for b2 in userItems:
        users_b2 = set([x for x in ratingsPerItem.get(b2, [])])
        sim = Jaccard(users_b, users_b2)
        if sim > maxSim:
            maxSim = sim
    
    if maxSim > 0.013 or len(ratingsPerItem[b]) > 40: # Keep these thresholds as-is
        return 1
    return 0


# In[ ]:


def writePredictionsRead(ratingsPerItem, ratingsPerUser):
    predictions = open("predictions_Read.csv", 'w')
    for l in open("pairs_Read.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u,b = l.strip().split(',')
        pred = jaccardThresh(u,b,ratingsPerItem,ratingsPerUser)
        _ = predictions.write(u + ',' + b + ',' + str(pred) + '\n')

    predictions.close()


# In[ ]:


##################################################
# Category prediction                            #
##################################################


# In[ ]:


def featureCat(datum, words, wordId, wordSet):
    feat = [0]*len(words)

    # Compute features counting instance of each word in "words"
    # after converting to lower case and removing punctuation
    review_text = datum['review_text'].lower()
    translator = str.maketrans('', '', string.punctuation)
    review_text = review_text.translate(translator)
    for w in review_text.split():
        if w in wordSet:
            feat[wordId[w]] += 1
    
    feat.append(1) # offset (put at the end)
    return feat


# In[ ]:


def betterFeatures(data):

    # Produce better features than those from the above question
    # Return matrix (each row is the feature vector for one entry in the dataset)
    wordCount = defaultdict(int)
    translator = str.maketrans('', '', string.punctuation)
    for d in data:
        review_text = d['review_text'].lower()
        review_text = review_text.translate(translator)
        for w in review_text.split():
            wordCount[w] += 1

    # Use top 1000 words instead of 500
    topWords = [x[0] for x in sorted(wordCount.items(), key=lambda x: -x[1])[:1000]]
    wordId = {w: i for i, w in enumerate(topWords)}
    wordSet = set(topWords)

    X = []
    for d in data:
        feat = [0]*len(topWords)
        review_text = d['review_text'].lower().translate(translator)
        for w in review_text.split():
            if w in wordSet:
                feat[wordId[w]] += 1
        feat.append(1)
        X.append(feat)
    return X


# In[ ]:


def runOnTest(data_test, mod):
    Xtest = [featureCat(d) for d in data_test]
    pred_test = mod.predict(Xtest)


# In[ ]:


def writePredictionsCategory(pred_test):
    predictions = open("predictions_Category.csv", 'w')
    pos = 0

    for l in open("pairs_Category.csv"):
        if l.startswith("userID"):
            predictions.write(l)
            continue
        u,b = l.strip().split(',')
        _ = predictions.write(u + ',' + b + ',' + str(pred_test[pos]) + '\n')
        pos += 1

    predictions.close()


# In[ ]:




