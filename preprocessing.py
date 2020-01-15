# -*- coding: utf-8 -*-
"""
Created on Wednesday January 15 22:26
@author: eskandari
"""

from __future__ import unicode_literals
from hazm import *
from nltk.stem.wordnet import WordNetLemmatizer
import pickle
#hazm
import re
import html
import pandas as pd
import numpy as np
import pickle



def cleanhtml(raw_html):
  #decode html entities, removes html tags, email address , twitter and telegram id  links and common patterns 
  
  
  #decode html 
  cleantext = html.unescape(raw_html)
  
  #remove anchor tags and theit inner content
  cleanr = re.compile('<\s*a[^>]*>.*<\s*/\s*a\s*>')
  cleantext = re.sub(cleanr, ' ', cleantext)

  #remove tags  
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, ' ', cleantext)
  



    
  #email address
  cleanr = re.compile('/\A[\w+\-.]+@[a-z0-9\-]+(\.[a-z]+)*\.[a-z]+/i')
  cleantext = re.sub(cleanr, ' ', cleantext) 

  #twitter and telegram id 
  cleanr = re.compile('(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)')
  cleantext = re.sub(cleanr, ' ', cleantext) 

  # remove links
  cleanr = re.compile('(http[s]?://)?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
  cleantext = re.sub(cleanr, ' ', cleantext)

  # remove punctuations
  cleanr = re.compile('[*?؟!].*')
  cleantext = re.sub(cleanr, ' ', cleantext)
    
  # remove multiple spaces
  cleanr = re.compile('\s+')
  cleantext = re.sub(cleanr, ' ', cleantext)  
  
  return cleantext






def cleanText(docs , stopwords_path):
    
    #normalize, clean , tokenize documents and remove stop wors 
    
    normalizer = Normalizer()
    lemmatizer = Lemmatizer()

    
    for idx in range(len(docs)):
        #remove non asci, non persoan and non arbic characters
       # print(docs[idx])
        docs[idx]  = re.sub(r'[^\x00-\x7f\u0600-\u0605 ؐ-ؚ\u061Cـ ۖ-\u06DD ۟-ۤ ۧ ۨ ۪-ۭ ً-ٕ ٟ ٖ-ٞ ٰ ، ؍ ٫ ٬ ؛ ؞ ؟ ۔ ٭ ٪ ؉ ؊ ؈ ؎ ؏۞ ۩ ؆ ؇ ؋ ٠۰ ١۱ ٢۲ ٣۳ ٤۴ ٥۵ ٦۶ ٧۷ ٨۸ ٩۹ ءٴ۽ آ أ ٲ ٱ ؤ إ ٳ ئ ا ٵ ٮ ب ٻ پ ڀة-ث ٹ ٺ ټ ٽ ٿ ج ڃ ڄ چ ڿ ڇ ح خ ځ ڂ څ د ذ ڈ-ڐ ۮ ر ز ڑ-ڙ ۯ س ش ښ-ڜ ۺ ص ض ڝ ڞۻ ط ظ ڟ ع غ ڠ ۼ ف ڡ-ڦ ٯ ق ڧ ڨ ك ک-ڴ ػ ؼ ل ڵ-ڸ م۾ ن ں-ڽ ڹ ه ھ ہ-ۃ ۿ ەۀ وۥ ٶۄ-ۇ ٷ ۈ-ۋ ۏ ى يۦ ٸ ی-ێ ې ۑ ؽ-ؿ ؠ ے ۓ \u061D]',r'', docs[idx] )
    
        #normalize text
        docs[idx] =  normalizer.normalize(docs[idx])
        #clean html 
        docs[idx] = cleanhtml(docs[idx].lower())  # Convert to lowercase.    
        
    for idx in range(len(docs)):    
        docs[idx] = word_tokenize(docs[idx])  # Split into words.

    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isnumeric()] for doc in docs]


    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 1] for doc in docs]

    #stemming 
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]
    docs = [[lemma[ 0 if lemma.find('#') <0 else lemma.find('#') : ]  for lemma in doc] for doc in docs] # for verbs
                                    
    # remove suffixes and prefixes missed by lemmatizer  
    docs = [[token.replace('\u200cای' , '').replace('\u200cهایی', '').replace('\u200cهای', '').replace('\u200cها', '').replace('می\u200c', '') for token in doc] for doc in docs]

    


    #remove stop words
      
    # Read the stop words CSV into a pandas data frame 
    if not stopwords_path=='' :
        df = pd.read_csv(stopwords_path, delimiter=',')
        
        stop_words = df["stopwords"].values
     
        filtered_docs = []
        
        for idx in range(len(docs)):  
            filtered_docs.append( [w for w in docs[idx] if not w in stop_words ])  #if not w in docs[idx]
    
        
    else:
        filtered_docs =  docs
    

#    allTokens = []
#    
#    for tokens in filtered_docs :
#        allTokens.extend( [ token for token in tokens ] )
#    import nltk
#    all_word_dist = nltk.FreqDist(allTokens)
#    most_common= all_word_dist.most_common(300)
#    print('FFFFFFFFFFFFFFFffff')
#    print( most_common)
#    print('FFFFFFFFFFFFFFFffff')
    
    return filtered_docs
def get_most_frequent_words(docs): 
    docs = cleanText (docs , '')
    allTokens = []
    for tokens in docs :
        allTokens.extend( [ token for token in tokens ] )
    import nltk
    all_word_dist = nltk.FreqDist(allTokens)
    most_common= all_word_dist.most_common(300)
    return most_common