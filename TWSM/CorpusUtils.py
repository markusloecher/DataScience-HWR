# -*- coding: utf-8 -*-


import os
#os.chdir('C:\\Users\\loecherm\\Dropbox\\Markus\\Teaching\\SS2019\\AnalyticsLab\\Lessons\\')
#os.getcwd()

import os
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer 
import string # for punctuation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


###########################################################
###########################################################
###########################################################


# a function to POS-tag each word for lemmatization


def get_wordnet_pos(word):
    '''Map POS tag to first character lemmatize() accepts'''
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return(tag_dict.get(tag, wordnet.NOUN))  # NOUN is the default


# mystops is a user defined list of stopwords
mystops = ['mr', 'said', 'sir', 'upon', 'mrs', 
                                    'replied', 'man', 'one', 'little', 
                                    'say', 'dont', 'old', 'gentleman',
                                    'time', 'two', 'never', 'see', 'door',
                                    'dear', 'well', 'now', 'will', 'dear',
                                    'know', 'head', 'come', 'much', 'hand',
                                    'o', 'inquired', 'room', 'think',
                                    'way', 'away', 'great', 'good', 'gentleman', 
                                    'lady', 'long', 'first', 'made', 
                                    'back', 'another', 'can', 'take', 'must',
                                    'just', 'ever', 'face', 'nothing', 'without',
                                    'ever', 'shall', 'took', 'look', 'friend',
                                    'oh', 'yes', 'many','last', 'might', 'go',
                                    'may', 'looking', 'rather', 'got', 'place',
                                    'mind', 'right', 'house', 'three','every', 'day',
                                    'put', 'thats', 'quite',
                                    'call', 'could', 'even', 'eye', 'get', 'give', 'let', 'make', 'open',
           'reply', 'turn', 'would']
            


def makeCleanCorpus(abspath = os.path.abspath('.') + '\\all-chapters\\',
                    removePunct=True,
                    removeNums=True,
                    lower=True,
                    stops=[],
                    removeStopw=True,
                    lemmatize=False,
                    textlist=None):
    '''
    The makeCleanCorpus function will look for text files in the directory 
    specified by abspath. Change this to suit you.
    
    Other Arguments:
        removePunct: Should punctuation be removed?
        removeNums:  Should numbers be removed"
        lower:       Should words be converted to lower-case?
        removeStopw: Should stop-words be removed?
                     Apart for the standard English stop-words, the 
                     variable "mystops" in the code below allows you to add 
                     your own stop-words
        lemmatize:   Should words be lemmatized? The default is "False"
                     Changing this to true will (really) slow this function down!
        textlist:    None by default. This function is intended for teaching 
                     purposes. 
                     Pass a list of sentences here to demonstrate the idea 
                     of a DTM. None of the other options will work (right now) 
                     if textlist is not None.
                     
        
    If "textlist" is None (the default) this function looks for text files 
    in the directory specified by the "abspath" directory and returns a dictionary 
    called "clean_dict" in which the keys are the file names and the 
    values are the cleaned up text.
    
    If "textlist" is a list of English sentences, this function will return 
    a DTM for the list.
    
    '''
    
    
    def cleanUp(files):
        '''
            A nested function in the makeCleanCorpus() function.
            This function cleans up the corpus using the arguments 
            passed to the makeCleanCorpus() function.
            This function returns a dictionary of filenames and cleaned up text.
        '''
        
        clean_files = {}
        
        for filename, text in files.items():
            print('Cleaning:', filename)
            #print(text[0:700])
            #print("=" * 80)
            StopWords = stopwords.words("english") + mystops
            
                       
            if lemmatize:
                lemmatizer = WordNetLemmatizer()
            # remove punctuation
            # use the three argumnt form of text.maketrans()
            if removePunct:
                text = text.translate(text.maketrans('', '', string.punctuation))
            
            # remove numbers
            if removeNums:
                text = ''.join([x for x in text if not x.isdigit()])
            
            # convert to lower-case
            if lower:
                text = text.lower()
                       
            # lemmatize words
            if lemmatize:
                word_list = nltk.word_tokenize(text)    
                text = ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in word_list])
                
            # remove stopwords
            if removeStopw:
                text = ' '.join([word for word in text.split() if word not in StopWords])
             
            #print(text[0:700])
            #print("=" * 80)
            clean_files[filename] = text
            
        return(clean_files)
    # end of nested function cleanUp
    ############
    
    if textlist is None:  # reading all text files from a directory
    
        files = {}  # a dictionary to hold all the filenames and their content
        # clean_dict = {}  # returns a dict with file-names as keys and text as values
                
        for filename in os.listdir(abspath):
            #print(filename)
            with open(abspath+filename, "r", encoding='latin-1') as file:
                files[filename] = file.read()
        
            #print("Done reading file...\n\n")
            #print('--'*25, '\n')          
                
        # call the function that cleans up the corpus
        files = cleanUp(files)            
                
        print('\nDone!')
        print('='*20)
        return(files)
            
    else:  # a list of actual text has been passed to the argument textlist

        files = {} # a dict to hold the text indexed by numbers       
        index = range(1, len(textlist)+1)        
        files = dict(zip(index, textlist))
        files = cleanUp(files)
        return(files)

# end of function makeCleanCorpus

def makeDTM(corpus, tfidf=False):
    '''
        This function will make a Document Term Matrix (DTM) from a corpus 
        passed to it. If tfidf is True, then the DTM returned will have 
        TfIdf values for the terms and not the frequencies.
    '''
    
    if (not tfidf):
        cvec = CountVectorizer()
        smat = cvec.fit_transform(corpus.values())  
        dtm = pd.DataFrame(smat.toarray(), 
                           columns=cvec.get_feature_names(), 
                           index=corpus.keys())
        return(dtm)
    else:
        tvec = TfidfVectorizer()
        tmat = tvec.fit_transform(corpus.values())
        dtm = pd.DataFrame(tmat.toarray(), 
                           columns=tvec.get_feature_names(), 
                           index=corpus.keys())
        return(dtm)

# end of function makeDTM
        
#################################

# make a clean corpus and then return the cleaned up corpus   
# for a list of sentences


################################################################
################################################################

# this works, but is still experimental

# controlling sparsity

# LOGIC: replace each non-zero frequency in the DTM with a 1
# Then, the sum of col is the number of times the term occurs in the corpus.
# The sparsity is this col sum divided by the number of docs 

# percent is a number between 1 and 100
def controlSparsity(percent):
    
    dtm.iloc[:,1]
    
    dtm.iloc[1]  # some values > 1
    occ_dtm = dtm.copy(deep=True)
    occ_dtm.iloc[1]
    occ_dtm[occ_dtm != 0] = 1
    occ_dtm.iloc[1]  # all non-zero val are 1
    dtm.iloc[1]  # has not changed - that's what a deep copy does
    
    # now use the occurence matrix occ_dtm to calculate sparsesness
    
    occ_col_sums = occ_dtm.sum()  # the no of docs in which each term appears
    
    occ_col_sums[occ_col_sums==5]
    
    # calculate the sparsity
    num_docs = len(dtm.index)
    num_docs
    sparsity = occ_col_sums/num_docs
    sparsity
    len(sparsity)
    
    # got the sparsity
    # now remove terms with sparsity less than specified level
    sparsity_cutoff = percent/100
    
    
    sp = sparsity[sparsity>sparsity_cutoff]
    len(sp)
    sp
    
    
    dtm.index
    dtm.columns
    
    sp.index
    
    # for sparsity > sparsity_cutoff, keep these indices:
    keep_indices = dtm.columns.difference(dtm.columns.difference(sp.index))
    keep_indices
    
    # and here is the sparse DTM!
    dtm.sparse95 = dtm.loc[:, keep_indices]
    # the above throws a warning (?) the first time you run it, but it works :)
    
    
    return(dtm.sparse95)


#############################################
    




