# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_score
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD


import scipy
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

numFolds = 4
random_seed = 1

seeds = [0,42,100,1337]
plots = [221, 222, 223, 224]


def digits():
    digits = datasets.load_digits()

    n_samples = len(digits.images)

    num_train_samples = int(n_samples * 0.8)

    trainData = digits.data[:num_train_samples]
    trainTarget = digits.target[:num_train_samples]

    testData = digits.data[num_train_samples:]
    testTarget = digits.target[num_train_samples:]

    #perform_test(trainData, trainTarget, testData, testTarget)
    return (digits.data, digits.target, trainData, trainTarget, testData, testTarget)

def SDSS():
    # The SDSS dataset
    # code for loading SDSS dataset comes from https://www.kaggle.com/jaysn1/sdss-object-detection
    dataset=pd.read_csv('SDSS.csv',skiprows=0)



    #Dropping unimportant fields
    dataset=dataset.drop(columns=['objid','specobjid','run','rerun','camcol','field'])

    #print dataset.head()

    from sklearn.preprocessing import LabelEncoder
    #dataset=dataset.apply(LabelEncoder().fit_transform)
    dataset['class'] = LabelEncoder().fit_transform(dataset['class'])
    #print dataset.head()

    X=dataset.drop(columns=['class'])
    y=dataset.iloc[:,7].values


    n_samples = len(y)
    #print "Num samples: ", n_samples
    #print len(X)

    X = X.values
    from sklearn.preprocessing import MinMaxScaler
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(X)
    X = scaling.transform(X)
    
    num_train_samples = int(n_samples * 0.8)

    trainData = X[:num_train_samples]
    trainTarget = y[:num_train_samples]

    train_count = {}
    test_count = {}
    for ty in trainTarget:
        if ty in train_count:
            train_count[ty]+=1
        else:
            train_count[ty]=0

    

    testData = X[num_train_samples:]
    testTarget = y[num_train_samples:]

    for ty in testTarget:
        if ty in test_count:
            test_count[ty]+=1
        else:
            test_count[ty] = 0


    #perform_test(trainData, trainTarget, testData, testTarget)
    return (X, y, trainData, trainTarget, testData, testTarget)

def time_model(trainData, trainTarget, testData, testTarget, model):
    beforeTrain = time.time()
    model.fit(trainData, trainTarget)
    afterTrain = time.time()

    beforeTest = time.time()
    score = model.score(testData, testTarget)
    afterTest = time.time()

    print ("train time: " + str(afterTrain - beforeTrain) + ". test time: " + str( (afterTest - beforeTest) / len(testData)) + ". Score=" + str(score))


def nn_test(trainData, trainTarget, testData, testTarget, hidden=(15,), actFunc='relu', solver='adam'):
    clf = MLPClassifier(hidden_layer_sizes=hidden, random_state=1, max_iter=500, warm_start=True, activation=actFunc, solver=solver)
    test_score = clf.fit(trainData, trainTarget).score(testData, testTarget)
    train_score = clf.score(trainData, trainTarget)
    return (train_score, test_score)
    

def testNNs(trainData, trainTarget, testData, testTarget):
    possible_hidden = [(10,10)]
    
    kf = KFold(n_splits=numFolds)
    for hidden in possible_hidden:
        print("Hidden=" + str(hidden))
        train_score = 0
        test_score = 0
        for fold, (train, test) in enumerate(kf.split(trainData, trainTarget)):
            curTrain, curTest = nn_test(trainData, trainTarget, testData, testTarget, hidden=hidden)
            train_score += curTrain
            test_score += curTest
        train_score /= numFolds
        test_score /= numFolds
        print("Train Accuracy= " + str(train_score) + ". Test Score=" + str(test_score))
        if test_score > bestScore:
            bestScore = test_score
            bestHidden = hidden

    print('The best CV score was hidden=' + str(bestHidden) + " with CV err = " + str(bestScore))

    clf = MLPClassifier(hidden_layer_sizes=bestHidden, random_state=1, max_iter=500, warm_start=True, activation='relu', solver='adam')
    time_model(trainData, trainTarget, testData, testTarget, clf)

def perform_test(trainData, trainTarget, testData, testTarget):
    bestNN = testNNs(trainData, trainTarget, testData, testTarget)

def run_kmeans(data, name):
    ks = []
    inertias = []
    previous = None
    highest_sil = -2.0
    highest_sil_k = None
    for i in range(1, 30):
        k = i + 1
        kmeans = KMeans(n_clusters = k, random_state=0, n_jobs=2)
        kmeans.fit(data)
        inertia = kmeans.inertia_
        ks = ks + [k]
        inertias = inertias + [inertia]
        labels = kmeans.predict(data)
        silhouette_avg = silhouette_score(data, labels)
        if silhouette_avg > highest_sil:
            highest_sil = silhouette_avg
            highest_sil_k = k
        print "%s,%s,%s" % (k, inertia, silhouette_avg)
        if previous is not None:
            absdiff = previous - inertia
            perdiff = (previous / inertia) - 1
            #print "Diff abs=%s, percentage=%s" % (absdiff, perdiff)
        previous = inertia


    print "Highest silhouette value was %s for k=%s" % (highest_sil, highest_sil_k)
    plt.plot(ks, inertias)
    plt.title("K-Means of " + name)
    plt.ylabel("Inertia")
    plt.xlabel('Number of clusters')
    plt.tight_layout()
    plt.savefig(name + "-kmeans", bbox_inches = "tight")
    plt.close()
    return highest_sil_k

def run_em(data, name):
    ks = []
    scores = []
    previous = None
    highest_sil = -2.0
    highest_sil_k = None
    print "K,log-likelihood,Silhouette Score"
    for i in range(1, 20):
        k = i + 1
        gmm = GaussianMixture(n_components = k, random_state=0)
        gmm.fit(data)
        score = gmm.score(data)
        ks = ks + [k]
        scores = scores + [score]
        labels = gmm.predict(data)
        silhouette_avg = silhouette_score(data, labels)
        if silhouette_avg > highest_sil:
            highest_sil = silhouette_avg
            highest_sil_k = k
        print "%s,%s,%s" % (k, score, silhouette_avg)
        if previous is not None:
            absdiff = previous - score
            perdiff = (previous / score) - 1
            #print "Diff abs=%s, percentage=%s" % (absdiff, perdiff)
        previous = score

    print "Highest silhouette value was %s for k=%s" % (highest_sil, highest_sil_k)
    plt.plot(ks, scores)
    plt.title("EM of " + name)
    plt.ylabel('log likelihood')
    plt.xlabel('number of mixture components')
    plt.tight_layout()
    plt.savefig(name + "-gmm", bbox_inches = "tight")
    plt.close()
    return highest_sil_k

#Run the clustering algorithms on the data sets and describe what you see.
def run_clustering(data, name):
    best_kmeans = run_kmeans(data, name)
    best_gmm = run_em(data, name)
    return (best_kmeans, best_gmm)

# Run clustering on the dataset, and see how the labels compare with the labeled training data.
# Do for this clusters of n=2 and n=N
def run_cluster_table(data, target, name):
    ns = [2, len(np.unique(target))]

    for n in ns:
        kmeans = KMeans(n_clusters = n, random_state=0, n_jobs=2)
        kmeans.fit(data)
        kmeansout = kmeans.predict(data)
        mat = confusion_matrix(target, kmeansout)
        with open(name + " kmeans k=" + str(n) + ".csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerows(mat)
        
        gmm = GaussianMixture(n_components = n, random_state=0)
        gmm.fit(data)
        gmmout = gmm.predict(data)
        mat = confusion_matrix(target, gmmout)
        with open(name + " gmm k=" + str(n) + ".csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerows(mat)
        print ""

def run_nn(trainData, trainTarget, testData, testTarget):
    model = MLPClassifier(hidden_layer_sizes=(500,500), random_state=1, max_iter=500, warm_start=True, activation='relu', solver='adam')
    beforeTrain = time.time()
    model.fit(trainData, trainTarget)
    afterTrain = time.time()

    beforeTest = time.time()
    score = model.score(testData, testTarget)
    afterTest = time.time()

    print ("train time: " + str(afterTrain - beforeTrain) + ". test time: " + str( (afterTest - beforeTest) / len(testData)) + ". Score=" + str(score))
    
def run_pca(data, target, targets, name):
    pca_n = get_variance(data, target, targets, name)
    visualize_pca(data, target, targets, name)
    return pca_n

def get_variance(data, target, targets, name):
    pca = PCA(n_components=len(data[0]), random_state=0)
    X_r = pca.fit(data).transform(data)

    n = np.argmax(pca.explained_variance_ratio_.cumsum() > 0.99) + 1
    print "\nPCA Variance Analysis on " + name + " get reaches 99% variance on n=" + str(n)
    print "Original number of features is " + str(len(data[0]))
    print pca.explained_variance_[0:n]
    print pca.explained_variance_ratio_[0:n]
    print pca.explained_variance_ratio_.cumsum()[0:n]

    pca = PCA(n_components=n, random_state=0)
    X_r = pca.fit(data).transform(data)
    recon = pca.inverse_transform(X_r)
    mse = ((data - recon) ** 2).mean()
    print 'PCA reconstruction MSE: %s' % (mse)

    return n

def visualize_pca(data, target, targets, name):
    pca = PCA(n_components=2, random_state=0)
    X_r = pca.fit(data).transform(data)
    for i, target_name in zip(targets, targets):
        plt.scatter(X_r[target == i, 0], X_r[target == i, 1], alpha=.8, lw=2,
                label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title("PCA of " + name)
    plt.savefig(name + " pca")
    plt.close()

def get_kurtosis(data, target, targets, name, n):
    from sklearn.preprocessing import StandardScaler
    #ss = StandardScaler(copy=True)
    ica = FastICA(n_components=n, random_state=0, max_iter=10000)
    X_r = ica.fit(data).transform(data)
    
    print 'kurtosis of %s with %s components.' % (name, n)

    vals = []
    for i in range(n):
        vals = vals +  [scipy.stats.kurtosis(X_r[:,i], fisher=False)]

    vals.sort()
    print vals
    
    return X_r


def visualize_ica(data, target, targets, name):
    ica = FastICA(n_components=2, random_state=0, max_iter=100000)
    X_r = ica.fit(data).transform(data)
    
    for i, target_name in zip(targets, targets):
        plt.scatter(X_r[target == i, 0], X_r[target == i, 1], alpha=.8, lw=2,
                label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title("ICA of " + name)
    plt.savefig(name + " ica")
    plt.close()

def recon_ica(data, targeet, targets, name):
    components = 42
    if len(data[0]) < 42:
        components = 7
    ica = FastICA(n_components=components, random_state=0, max_iter=100000)

    X_r = ica.fit(data).transform(data)
    recon = ica.inverse_transform(X_r)
    mse = ((data - recon) ** 2).mean()
    print 'ICA reconstruction with %s components is %s' % (components, mse)

def run_ica(data, target, targets, name):
    get_kurtosis(data, target, targets, name, 2)
    get_kurtosis(data, target, targets, name, min(42, len(data[0])))
    visualize_ica(data, target, targets, name)
    recon_ica(data, target, targets, name)

def run_rand(data, target, targets, name):
    plt.subplots(figsize=(18,10))
    for i in range(len(seeds)):
        transformer = GaussianRandomProjection(n_components=2, random_state=seeds[i])
        transformer.fit(data)
        randTrain = transformer.transform(data)
        plt.subplot(plots[i])
        for i, target_name in zip(targets, targets):
            plt.scatter(randTrain[target == i, 0], randTrain[target == i, 1], alpha=.8, lw=2,
                label=target_name)

        
        plt.legend(loc='best', shadow=False, scatterpoints=1)
        plt.title("Randomized Projection of " + name)
    plt.savefig(name + " random")
    plt.close()

"""
def run_selectbest(data, target, targets, name):
    best = SelectKBest(chi2, k=2)
    X_r = best.fit(data, target).transform(data)
    
    for i, target_name in zip(targets, targets):
        plt.scatter(X_r[target == i, 0], X_r[target == i, 1], alpha=.8, lw=2,
                label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title("select2best of " + name)
    plt.savefig(name + " 2best")
    plt.close()


def run_selectbest(data, target, targets, name):
    best = LatentDirichletAllocation(n_topics=2)
    X_r = best.fit(data).transform(data)
    
    for i, target_name in zip(targets, targets):
        plt.scatter(X_r[target == i, 0], X_r[target == i, 1], alpha=.8, lw=2,
                label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title("LDA of " + name)
    plt.savefig(name + " lda")
    plt.close()
"""

def reconstruct_svd(data, target, targets, name):
    svd = TruncatedSVD(n_components=len(data[0]) - 1, random_state=0)
    X_r = svd.fit(data).transform(data)
    n = np.argmax(svd.explained_variance_ratio_.cumsum() > 0.99) + 1
    print "\nSVD Variance Analysis on " + name + " get reaches 99% variance on n=" + str(n)
    print "Original number of features is " + str(len(data[0]))
    print svd.explained_variance_[0:n]
    print svd.explained_variance_ratio_[0:n]
    print svd.explained_variance_ratio_.cumsum()[0:n]

    print 'Doing SVD reconstruction with %s features' % (n)
    svd = TruncatedSVD(n_components=n, random_state=0)
    X_r = svd.fit(data).transform(data)
    recon = svd.inverse_transform(X_r)
    mse = ((data - recon) ** 2).mean()
    print 'SVD reconstruction MSE: %s' % (mse)


def visualize_svd(data, target, targets, name):
    svd = TruncatedSVD(n_components=2, random_state=0)
    X_r = svd.fit(data).transform(data)
    for i, target_name in zip(targets, targets):
        plt.scatter(X_r[target == i, 0], X_r[target == i, 1], alpha=.8, lw=2,
                label=target_name)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title("Kernel PCA of " + name)
    plt.savefig(name + " kpca")
    plt.close()


#Apply the dimensionality reduction algorithms to the two datasets and describe what you see.
def run_dimred(data, target, targets, name):
 
    # need to run:
    # PCA
    pca_n = run_pca(data, target, targets, name)
    # ICA
    run_ica(data, target, targets, name)
    # Randomized Projections
    
    #run_rand(data, target, targets, name)
    # ETC
    reconstruct_svd(data, target, target, name)
    #visualize_svd(data, target, targets, name)
    return pca_n, pca_n, pca_n, pca_n

#Reproduce your clustering experiments, but on the data after you've run dimensionality reduction on it.
def run_cluster_on_pca(data, target, targets, name, pca_n):
    pca = PCA(n_components=pca_n, random_state=0)
    X_r = pca.fit(data).transform(data)
    run_clustering(X_r, "Digits reduced by PCA")
    run_cluster_table(X_r, target, "Digits reduced by PCA")
    


#Apply the dimensionality reduction algorithms to one of your datasets from assignment #1 (if you've reused the datasets from assignment #1 to do experiments 1-3 above then you've already done this) and rerun your neural network learner on the newly projected data.
def run_dimredNN(trainData, trainTarget, testData, testTarget, pca_n, ica_n, rand_n, bestk_n):
    if True:
        pca = PCA(n_components=pca_n, random_state=0)
        pca.fit(trainData)
        pcaTrain = pca.transform(trainData)
        pcaTest = pca.transform(testData)
        print "Running NN with pca n=%s feature reduced dimensions." % (pca_n)
        run_nn(pcaTrain, trainTarget, pcaTest, testTarget)

    
    if True:
        ica = FastICA(n_components=ica_n, random_state=0, max_iter=100000)
        ica.fit(trainData)
        icaTrain = ica.transform(trainData)
        icaTest = ica.transform(testData)
        print "Running NN with ica n=%s feature reduced dimensions." % (ica_n)
        run_nn(icaTrain, trainTarget, icaTest, testTarget)
    

    if True:
        for seed in seeds:
            transformer = GaussianRandomProjection(n_components=rand_n, random_state=seed)
            transformer.fit(trainData)
            randTrain = transformer.transform(trainData)
            randTest = transformer.transform(testData)
            print "Running NN with randomized projections with seed %s." % (seed)
            run_nn(randTrain, trainTarget, randTest, testTarget)

    if True:
        svd = TruncatedSVD(n_components=pca_n, random_state=0)
        X_r = svd.fit(trainData)
        selectTrain = svd.transform(trainData)
        selectTest = svd.transform(testData)
        print "Running NN with SVD n=%s feature reduced dimensions." % (bestk_n)
        run_nn(selectTrain, trainTarget, selectTest, testTarget)


#Apply the clustering algorithms to the same dataset to which you just applied the dimensionality reduction algorithms (you've probably already done this), treating the clusters as if they were new features. In other words, treat the clustering algorithms as if they were dimensionality reduction algorithms. Again, rerun your neural network learner on the newly projected data.
def run_clusteredNN(trainData, trainTarget, testData, testTarget, kmean_k, gmm_k):
    if True:
        kmeans = KMeans(n_clusters = kmean_k, random_state=0, n_jobs=2)
        kmeans.fit(trainData)
        train_labels = kmeans.predict(trainData)
        test_labels = kmeans.predict(testData)
        train_labels = train_labels.reshape((len(train_labels), 1))
        test_labels = test_labels.reshape((len(test_labels), 1))
        kmeansTrain = np.hstack((trainData, train_labels))
        kmeansTest = np.hstack((testData, test_labels))
        print "Running NN with k-means k=%s feature cluster added." % (kmean_k)
        run_nn(kmeansTrain, trainTarget, kmeansTest, testTarget)

    if True:
        gmm = GaussianMixture(n_components = gmm_k, random_state=0)
        gmm.fit(trainData)
        train_labels = gmm.predict(trainData)
        train_labels = train_labels.reshape((len(train_labels), 1))
        test_labels = gmm.predict(testData)
        test_labels = test_labels.reshape((len(test_labels), 1))

        gmmTrain = np.hstack((trainData, train_labels))
        gmmTest = np.hstack((testData, test_labels))
        print "Running NN with gmm k=%s feature cluster added." % (gmm_k)
        run_nn(gmmTrain, trainTarget, gmmTest, testTarget)   

# ru9ns all the above tests.
def run_all():
    _, _, digitsTrainData, digitsTrainTarget, digitsTestData, digitsTestTarget = digits()
    _, _, sdssTrainData, sdssTrainTarget, sdssTestData, sdssTestTarget = SDSS()
    
    print "---------- Starting clustering on digits ----------"
    best_digits_kmeans, best_digits_gmm = run_clustering(digitsTrainData, "DIGITS")
    run_cluster_table(digitsTrainData, digitsTrainTarget, "DIGITS")
    run_clusteredNN(digitsTrainData, digitsTrainTarget, digitsTestData, digitsTestTarget, best_digits_kmeans, best_digits_gmm)

    print "\n\n---------- Starting clustering on SDSS ----------"
    
    best_sdss_kmeans, best_sdss_gmm = run_clustering(sdssTrainData, "SDSS")
    run_cluster_table(sdssTrainData, sdssTrainTarget, "SDSS")
    run_clusteredNN(digitsTrainData, digitsTrainTarget, digitsTestData, digitsTestTarget, best_sdss_kmeans, best_sdss_gmm)
    
    print "\n\n---------- Starting dimensionality reduction on digits ----------"
    pca_n, ica_n, rand_n, bestk_n = run_dimred(digitsTrainData, digitsTrainTarget, [0,1,2,3,4,5,6,7,8,9], 'digits')

    
    print "\n\n---------- Starting dimensionality reduction on SDSS ----------"
    run_dimred(sdssTrainData, sdssTrainTarget, [0,1,2], "sdss")
    
    print "\n\n---------- Starting NN on dimensionality reduction ----------"
    run_dimredNN(digitsTrainData, digitsTrainTarget, digitsTestData, digitsTestTarget, pca_n, pca_n, pca_n, pca_n)

    print "\n\n---------- Starting clustering on dimensionality reduction ----------"
    run_cluster_on_pca(digitsTrainData, digitsTrainTarget, [0,1,2,3,4,5,6,7,8,9], 'digits', pca_n)


run_all()