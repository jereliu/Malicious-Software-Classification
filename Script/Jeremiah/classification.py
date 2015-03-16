## This file provides starter code for extracting features from the xml files and
## for doing some learning.
##
## The basic set-up: 
## ----------------
## main() will run code to extract features, learn, and make predictions.
## 
## extract_feats() is called by main(), and it will iterate through the 
## train/test directories and parse each xml file into an xml.etree.ElementTree, 
## which is a standard python object used to represent an xml file in memory.
## (More information about xml.etree.ElementTree objects can be found here:
## http://docs.python.org/2/library/xml.etree.elementtree.html
## and here: http://eli.thegreenplace.net/2012/03/15/processing-xml-in-python-with-elementtree/)
## It will then use a series of "feature-functions" that you will write/modify
## in order to extract dictionaries of features from each ElementTree object.
## Finally, it will produce an N x D sparse design matrix containing the union
## of the features contained in the dictionaries produced by your "feature-functions."
## This matrix can then be plugged into your learning algorithm.
##
## The learning and prediction parts of main() are largely left to you, though
## it does contain code that randomly picks class-specific weights and predicts
## the class with the weights that give the highest score. If your prediction
## algorithm involves class-specific weights, you should, of course, learn 
## these class-specific weights in a more intelligent way.
##
## Feature-functions:
## --------------------
## "feature-functions" are functions that take an ElementTree object representing
## an xml file (which contains, among other things, the sequence of system calls a
## piece of potential malware has made), and returns a dictionary mapping feature names to 
## their respective numeric values. 
## For instance, a simple feature-function might map a system call history to the
## dictionary {'first_call-load_image': 1}. This is a boolean feature indicating
## whether the first system call made by the executable was 'load_image'. 
## Real-valued or count-based features can of course also be defined in this way. 
## Because this feature-function will be run over ElementTree objects for each 
## software execution history instance, we will have the (different)
## feature values of this feature for each history, and these values will make up 
## one of the columns in our final design matrix.
## Of course, multiple features can be defined within a single dictionary, and in
## the end all the dictionaries returned by feature functions (for a particular
## training example) will be unioned, so we can collect all the feature values 
## associated with that particular instance.
##
## Two example feature-functions, first_last_system_call_feats() and 
## system_call_count_feats(), are defined below.
## The first of these functions indicates what the first and last system-calls 
## made by an executable are, and the second records the total number of system
## calls made by an executable.
##
## What you need to do:
## --------------------
## 1. Write new feature-functions (or modify the example feature-functions) to
## extract useful features for this prediction task.
## 2. Implement an algorithm to learn from the design matrix produced, and to
## make predictions on unseen data. Naive code for these two steps is provided
## below, and marked by TODOs.
##
## Computational Caveat
## --------------------
## Because the biggest of any of the xml files is only around 35MB, the code below 
## will parse an entire xml file and store it in memory, compute features, and
## then get rid of it before parsing the next one. Storing the biggest of the files 
## in memory should require at most 200MB or so, which should be no problem for
## reasonably modern laptops. If this is too much, however, you can lower the
## memory requirement by using ElementTree.iterparse(), which does parsing in
## a streaming way. See http://eli.thegreenplace.net/2012/03/15/processing-xml-in-python-with-elementtree/
## for an example. 

import os
import sys
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

#bread and butter
import nltk
import numpy as np
from scipy import stats
from scipy import sparse

#toyz
from sklearn import cross_validation as cv
from sklearn import ensemble as es
os.chdir("/Users/Jeremiah/GitHub/CS-181-Practical-2/Script/Jeremiah")
import util
import pandas as pd

#plot
from ggplot import *


################################################
#### 1. Auxilliary Funcs #######################
################################################
def print_full(x):
    #print all rows of a panda dataframe
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


def Accuracy(clf_preds, t_train):
    clf_missId = ((clf_preds - t_train) != 0)
    clf_miss = t_train[(clf_preds - t_train) != 0]
    
    rate = 1 - np.mean(clf_missId) #error rate
    return rate, clf_miss

def name_for_feature(command, t_train, X_train, feature_dict):
    try:
        col_id = feature_dict[command]
    except KeyError:
        print ("command not exist")
        raise KeyError
        
    t_train = t_train.reshape(1, np.max(t_train.shape))
    ftData = X_train.T[col_id]

    nmIdx = t_train[np.array(ftData>0)]
    out = [util.malware_classes[i] for i in nmIdx]
    out = stats.itemfreq(out)
    
    return out


def pruneFeatures(minFreq, X_train_dense, global_feat_dict):
    print "pruning training features..."
    sys.stdout.flush()

    #featureFreq = [sum(feature != 0) for feature in X_train_dense.T]
    #featureFreq_pd = pd.Series(featureFreq, global_feat_dict.keys())
    featureFreq = []
    X_train_denseT = X_train_dense.T
    i = 0
    i_max = len(X_train_denseT)

    for feature in X_train_denseT:
        featureFreq.append(sum(feature != 0))
        i += 1
        if i % 5000 == 0: 
            print "Obtaining featureFreq: " + \
                    str(round(float(i)*100/i_max, 3)) + "%"

    print "obtaining prunId..."
    sys.stdout.flush()

    prunId = []
    i = 0
    for item in featureFreq: 
        if item > minFreq: prunId.append(i)
        i += 1
        if i % 5000 == 0: 
            print "Obtaining prunId: " + \
                    str(round(float(i)*100/i_max, 3)) + "%"
            sys.stdout.flush()
        
    print "obtaining prunId...Done!"
    sys.stdout.flush()

    print "Pruning Design Matrix... "; sys.stdout.flush()
    X_train_prune = X_train_dense.T[prunId].T #update X
    print "Pruning Design Matrix...Done!"

    global_feat_dict_prune = Counter() #update global_feat_dict
    
    dictList = global_feat_dict.items()
    dictList_prune = np.array(dictList)[np.array(prunId)]
    
    for i in range(len(dictList_prune)):
        global_feat_dict_prune[dictList_prune[i][0]] = dictList_prune[i][1]
        if i % 5000 == 0: 
            print "Pruning feature dict: " + \
                str(round(float(i)*100/len(dictList_prune), 3)) + "%"
            sys.stdout.flush()

    print "done pruning training features!"
    return X_train_prune, global_feat_dict_prune, featureFreq, prunId

    
################################################

################################
#### 2. IO functions
################################

def extract_feats(ffs, direc="train", global_feat_dict=None):
    """
    arguments:
      ffs are a list of feature-functions.
      direc is a directory containing xml files (expected to be train or test).
      global_feat_dict is a dictionary mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that 
      the columns of the test matrix align correctly.

    returns: 
      a sparse design matrix, a dict mapping features to column-numbers,
      a vector of target classes, and a list of system-call-history ids in order 
      of their rows in the design matrix.
      
      Note: the vector of target classes returned will contain the true indices of the
      target classes on the training data, but will contain only -1's on the test
      data
    """
    
    fds = [] # list of feature dicts
    classes = []
    ids = [] 
    
    i = 0
    N = str(len(os.listdir(direc)))
    
    for datafile in os.listdir(direc):
        if datafile == "DS.Store": continue
        # extract id and true class (if available) from filename
        id_str,clazz = datafile.split('.')[:2]
        ids.append(id_str)
        # add target class if this is training data
        try:
            classes.append(util.malware_classes.index(clazz))
        except ValueError:
            # we should only fail to find the label in our list of malware classes
            # if this is test data, which always has an "X" label
            assert clazz == "X"
            classes.append(-1)
        rowfd = {}
        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        # accumulate features
        [rowfd.update(ff(tree)) for ff in ffs]
        fds.append(rowfd)
        
        i += 1
        if i%100 == 0: 
            print "Progress: " + str(i) + "/" + N
            sys.stdout.flush()
        
    X,feat_dict = make_design_mat(fds,global_feat_dict)
    return X, feat_dict, np.array(classes), ids


def make_design_mat(fds, global_feat_dict = None):
    """
    arguments:
      fds is a list of feature dicts (one for each row).
      global_feat_dict is a dictionary mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that 
      the columns of the test matrix align correctly.
       
    returns: 
        a sparse NxD design matrix, where N == len(fds) and D is the number of
        the union of features defined in any of the fds 
    """
    if global_feat_dict is None:
        all_feats = set()
        [all_feats.update(fd.keys()) for fd in fds]
        feat_dict = dict([(feat, i) for i, feat in enumerate(sorted(all_feats))])
    else:
        feat_dict = global_feat_dict
        
    cols = []
    rows = []
    data = []        
    for i in xrange(len(fds)):
        temp_cols = []
        temp_data = []
        for feat,val in fds[i].iteritems():
            try:
                # update temp_cols iff update temp_data
                temp_cols.append(feat_dict[feat])
                temp_data.append(val)
            except KeyError as ex:
                if global_feat_dict is not None:
                    pass  # new feature in test data; nbd
                else:
                    raise ex

        # all fd's features in the same row
        k = len(temp_cols)
        cols.extend(temp_cols)
        data.extend(temp_data)
        rows.extend([i]*k)

    assert len(cols) == len(rows) and len(rows) == len(data)
   

    X = sparse.csr_matrix(
    (np.array(data),(np.array(rows), np.array(cols))), 
    shape=(len(fds), len(feat_dict))
    )
    
    return X, feat_dict
    
################################################

################################
#### 3. Feature Vectors
################################

## Here are two example feature-functions. They each take an xml.etree.ElementTree object, 
# (i.e., the result of parsing an xml file) and returns a dictionary mapping 
# feature-names to numeric values.
## TODO: 
##    1. DLL Type and Address.
##    2. enum_values, explorer
##    3. create_namedpipe
##    4. create_thread_remote


def get_all_keys(tree):
    key_list = Counter()
    tag_exclusion_set = ["load_dll"]
    
    in_all_section = False

    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            #if el.tag in tag_set
            for key in el.keys():
                if key not in tag_exclusion_set:
                    key_list_name = \
                        (el.tag + "_" + key + "-" + el.get(key)).lower()
                    key_list[key_list_name] += 1
    return key_list


def dll_type(tree, name_only = False):
    #type and address of loaded DLLs
    dll_list = {"name": [], "extn": [], "addr": []} 

    for el in tree.iter():
        # obtain DLL target in element
        if el.tag == "load_dll":
            dll_name = el.get('filename')
            #split filename and file_address
            try:
                key_bag = dll_name.split("\\")
                dll_name = "dll_name-" + key_bag[len(key_bag) - 1]
                dll_addr = "dll_addr-" + "//".join(key_bag[:(len(key_bag) - 1)])
            except AttributeError:
                dll_name = "dll_name-DLL_FILE_NOT_SPECIFIED"
                dll_addr = "dll_addr-" + "//".join(key_bag)
            
            if len(dll_name.split(".")) == 2:
                key_bag = dll_name.split(".")
                dll_name = key_bag[0]
                dll_extn = "dll_extn-" + key_bag[1]
            else:
                dll_extn = "dll_extn-"
            
            dll_name = dll_name.lower()
            dll_extn = dll_extn.lower()
            dll_addr = dll_addr.lower()
            #TODO: convert to lower case
            dll_list["name"].append(dll_name)
            dll_list["extn"].append(dll_extn)
            dll_list["addr"].append(dll_addr)

    dll_list["name"] = stats.itemfreq(dll_list["name"])
    dll_list["extn"] = stats.itemfreq(dll_list["extn"])    
    dll_list["addr"] = stats.itemfreq(dll_list["addr"])    
    dll_list_join = concatenate([dll_list["name"], \
                    dll_list["extn"], dll_list["addr"]])

    dll_name_counter = Counter()
    for item in dll_list_join: 
        dll_name_counter[item[0]] = int(item[1])

    return dll_name_counter


def call_freq(tree, name_only = False):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    returns:
      a dictionary mapping 'first_call-x' to 1 if x was the first system call
      made, and 'last_call-y' to 1 if y was the last system call made. 
      (in other words, it returns a dictionary indicating what the first and 
      last system calls made by an executable were.)
    """
    callz = []
    in_all_section = False
    first = True # is this the first system call
    last_call = None # keep track of last call we've seen
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            callz.append(el.tag)

    # finally, count the frequencies
    freqList = stats.itemfreq(callz)   
    
    if name_only == True:        
        c = set(callz)
    else: 
        c = Counter()
        for item in freqList: c["sys_call-" +item[0]] = int(item[1])

    return c


def call_eigen(tree, call_dict, name_only = False):
    #TODO: Implement this!
    c = []
    return c


def first_last_system_call_feats(tree):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    returns:
      a dictionary mapping 'first_call-x' to 1 if x was the first system call
      made, and 'last_call-y' to 1 if y was the last system call made. 
      (in other words, it returns a dictionary indicating what the first and 
      last system calls made by an executable were.)
    """
    c = Counter()
    in_all_section = False
    first = True # is this the first system call
    last_call = None # keep track of last call we've seen
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            if first:
                c["first_call-"+el.tag] = 1
                first = False
            last_call = el.tag  # update last call seen
            
    # finally, mark last call seen
    c["last_call-"+last_call] = 1
    return c

def system_call_count_feats(tree):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    returns:
      a dictionary mapping 'num_system_calls' to the number of system_calls
      made by an executable (summed over all processes)
    """
    c = Counter()
    in_all_section = False
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            c['num_system_calls'] += 1
    return c


################################
#### 4. Variables of Interest
################################


def call_type(direc):
    names = set() # list of feature dicts
    for datafile in os.listdir(direc):
        if datafile == "DS.Store": continue
        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        # accumulate features
        newnames = call_freq(tree, name_only = True)
        names = names.union(newnames)
    out_names = names
    return out_names

def call_freq_emp(direc):
    names = dict() # list of feature dicts
    for datafile in os.listdir(direc):
        if datafile == "DS.Store": continue
        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        # accumulate features
        newnames = call_freq(tree)
        for k in newnames.iterkeys():
            if k in names: names[k] = names[k] + newnames[k]
            else: names[k] = newnames[k]
            
    out_names = names
    return out_names
    
def call_freq_byType(direc):
    #initiate class dependent dictionary
    name_byType = dict() # list of feature dicts
    for TypeName in util.malware_classes:
        name_byType[TypeName] = dict()
 
    for datafile in os.listdir(direc):
        if datafile == "DS.Store": continue
        # extract id and true class (if available) from filename
        id_str,clazz = datafile.split('.')[:2]
        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        # accumulate features: first create new feature
        newnames = call_freq(tree)
        # 
        curNames = name_byType[clazz]
        for k in newnames.iterkeys():
            if k in curNames: 
                curNames[k] = curNames[k] + newnames[k]
            else: curNames[k] = newnames[k]
        name_byType[clazz] = curNames
        
    out_names = name_byType
    return out_names
    
def dll_type_2(tree, name_only = False):
    dll_list = {"name": [], "addr": []} 

    for el in tree.iter():
        # obtain DLL target in element
        if el.tag == "load_dll":
            dll_name = el.get('filename')
            #split filename and file_address
            key_bag = dll_name.split("\\")
            
            dll_name = key_bag[len(key_bag) - 1]
            dll_addr = "//".join(key_bag[:(len(key_bag) - 1)])
            #TODO: convert to lower case

            dll_list["name"].append(dll_name)
            dll_list["addr"].append(dll_addr)

    dll_list["name"] = stats.itemfreq(dll_list["name"])
    dll_list["addr"] = stats.itemfreq(dll_list["addr"])    
    dll_list_join = concatenate([dll_list["name"], dll_list["addr"]])

    dll_name_counter = Counter()
    for item in dll_list_join: 
        dll_name_counter[item[0]] = int(item[1])

    return dll_name_counter


## The following function does the feature extraction, learning, and prediction
def main():
    train_dir = "../../../Data/train"
    test_dir = "../../../Data/test"
    outputfile = "../../Output/Jeremiah.csv"  # feel free to change this or take it as an argument
 
    ################################
    #### Empirical Summary 
    ################################

    #Get types & fre of commands
    
    #raw count
    lesNames =  call_freq_emp(train_dir)
    lesNames_freq = pd.Series(lesNames.values(), lesNames.keys())
    lesNames_freq = lesNames_freq/sum(lesNames_freq)
    lesNames_freq.sort()
    lesNames_freq

    #raw count by Type
    lesNames_byType =  call_freq_byType(train_dir)
    
    lesNames_byType_freq = dict()
    for TypeName in util.malware_classes:
        lesNames_byType_freq[TypeName] = []
    
    for keyName in lesNames_byType.keys():
        namez = lesNames_byType[keyName]
        namez_freq = pd.Series(namez.values(), namez.keys())
        namez_freq = namez_freq/sum(namez_freq)
        namez_freq.sort(ascending = False)
        lesNames_byType_freq[keyName] = namez_freq[namez_freq > 0.01]
        print(keyName + " Finished!")
        sys.stdout.flush()
    
    lesNames_byType_freq #most frequent commands in each class
    
    
    ##Bar plot
    t_label = np.array(util.malware_classes)[np.array(t_train)] 
    
    
    bar_df = stats.itemfreq(t_label).T
    bar_df = stats.itemfreq(t_train).T
    
    
    bar_df = pd.DataFrame(data= bar_df).T
    bar_df.columns = ['name', 'count']
    bar_df[['count']] = bar_df[['count']].astype(int)
    
    ggplot(aes(x = "name", weight = "count"), bar_df) + \
           xlab("count") + geom_bar() + \
           ggtitle("Frequency Count for Malware Types")
    
    
    
    ################################
    #### Feature Extraction and Prunning 
    ################################

    # TODO put the names of the feature functions you've defined above in this list
    ffs = [first_last_system_call_feats, system_call_count_feats, \
            call_freq, dll_type]#, get_all_keys]

    # extract features
    print "extracting training features..."
    X_train,global_feat_dict,t_train,train_ids = extract_feats(ffs, train_dir)
    X_train_dense = X_train.todense()
    del X_train
    print "done extracting training features"
    print
    sys.stdout.flush()
    
    #prunning
    X_train_prune, global_feat_dict_prune, featureFreq, prunId = \
    pruneFeatures(minFreq = 1, X_train_dense = X_train_dense, \
                    global_feat_dict = global_feat_dict)

    ################################
    #### CV-based training 
    ################################
    n = X_train_dense.shape[0]
    nForest = 1000
    n_cv = 5
    
    print str(n_cv) + " fold learning initiated..."
    eRate_cv = []
    kf_cv = cv.KFold(n, n_folds = n_cv)
    clf_cv = es.RandomForestClassifier(n_estimators = nForest)
    i = 0
    
    for train_index, test_index in kf_cv:
        i += 1
        #create CV dataset and fit
        F_train, F_test = X_train_prune[train_index], X_train_prune[test_index]
        y_train, y_test = t_train[train_index], t_train[test_index]
        clf_fit = clf_cv.fit(F_train, y_train)
        #prediction
        clf_pred = clf_fit.predict(F_test)
        accuracy = Accuracy(clf_pred, y_test)[0]
        eRate_cv.append(accuracy)
        print("Fold " + str(i) + " Classification Accuracy = " + str(accuracy))
        sys.stdout.flush()
    print "done learning"
    print
    
    np.mean(eRate_cv)

    ################################
    #feature importance assessment: 
    ################################
    # train here, and learn your classification parameters
    print "learning..."
    nForest = 1000
    clf = es.RandomForestClassifier(n_estimators = nForest, \
                verbose = 1, n_jobs = -1)
    clf_fit = clf.fit(X_train_dense, t_train)
    print "done learning"
    print

    #TODO: Figure out param Name that Feature Importance corresponds to
    ftImp = pd.DataFrame(sorted(global_feat_dict.keys()), \
                            columns = ["Name"])
    ftImp["FeatureImp"] = clf_fit.feature_importances_
    ftImp_s = ftImp.sort(columns = "FeatureImp", ascending = False)

    print_full(ftImp_s)
    ftImp_s.loc[ ftImp_s['FeatureImp']> 0.000, :]

    ####################################
    # in sample prediction and mis-classification rate
    ####################################
    
    print "making in-sample predictions..."
    clf_preds = clf_fit.predict(X_train_dense)
    clf_missId = ((clf_preds - t_train) != 0)
    clf_miss = t_train[(clf_preds - t_train) != 0]
    
    rate = 1 - np.mean(clf_missId) #error rate
    
    clf_miss = [util.malware_classes[i] for i in clf_miss]
    stats.itemfreq(clf_miss)
    print "done making in-sample predictions"
    
    # get rid of training data and load test data
    del X_train
    del t_train
    del train_ids
    print "extracting test features..."
    X_test,_,t_ignore,test_ids = extract_feats(ffs, test_dir, global_feat_dict=global_feat_dict)
    X_test_dense = X_test.todense()
    X_test_prune = X_test_dense.T[prunId].T
    
    print "done extracting test features"
    print
    
    # TODO make predictions on text data and write them out
    print "making predictions..."
    clf_preds = clf_fit.predict(X_test_prune)
    print "done making predictions"
    print
    
    print "writing predictions..."
    util.write_predictions(clf_preds, test_ids, outputfile)
    print "done!"

if __name__ == "__main__":
    main()
    