# -*- coding: utf-8 -*-
"""""""""""""""""""""""""""""""""""""""
Emotion Detection Using Depechemood in Python

This script uses spacy tool along with DepecheMood v2.0 as an external library to provide
emotion measures of a particular text evoked in the reader of that text

@author: Mojtaba Heidarysafa
"""""""""""""""""""""""""""""""""""""""

# using depechemood...
# =============================================================================
import spacy
import pandas as pd
nlp = spacy.load("en")
path_to_depechemood_freq = 'C:/Users/Moji/Downloads/DepecheMood_v2.0/DepecheMood/DepecheMood_freq.txt'
def prepare_depechemood(path_to_depechemood_freq):
    '''
    this function reads external file of DepecheMood_freq.txt as a pandas' dataframe and 
    adds 2 extra columns for the word and part of speech tag of that word
    return: (pandas' dataframe) depeche_df
    '''
    depeche_df = pd.read_csv(path_to_depechemood_freq ,sep = '\s+')
    depeche_df['lemma']=[x[:-2] for x in depeche_df['Lemma#PoS']]
    depeche_df['pos']=[x[-1] for x in depeche_df['Lemma#PoS']]
    return depeche_df
            





def convert_pos(pos):
    '''
    function to convert part of speech tags of spacy to be compatible with depechemood pos
    parameters: pos a list of lists [ index, parsed document part of speech byspacy]
    return: pos_ converted a converted list of the same shape
    '''
    pos_converted=[]
    for i,pos_ in pos:
        
        if pos_ == 'VERB':
            print(i)
            pos_ = 'v'
        elif pos_ == 'ADV':
            pos_ = 'r'
        elif pos_ == 'NOUN':
            pos_ = 'n'
        elif pos_ == 'ADJ':
            pos_ = 'a'
        pos_converted.append((str(i),pos_))
    return pos_converted
def detect_feeling(doc, depeche):
    '''
    main function to detect emotion of a text
    parameters: doc(parsed result of spacy nlp)
    depeche(modified dataframe)
    returns: tuple of 3 dataframes
    
    '''
    pos = [[i, i.pos_] for i in doc]
    pos2 = convert_pos(pos)
    df3 = pd.DataFrame()
    for word,tag in pos2:
        df2 = pd.DataFrame()
        if word in depeche['lemma'].values:
            #print(type(depoche[depoche['lemma']==word]))
            df2 = df2.append(depeche[depeche['lemma']==word])
            if tag in df2['pos'].values:
                df3 = df3.append(df2[df2['pos']==tag])
                
    feelings = df3.iloc[:,1:-2].sum(axis=0)/len(pos)
    feelings_norm = (feelings - feelings.min()) / (feelings.max() - feelings.min())
    return feelings, feelings_norm,df3.iloc[:,1:-1]

####### Example test  #######      
test_text =    '''
A magnitude 7.7 earthquake struck Tuesday about 80 miles from Jamaica, shaking people in the Caribbean and as far away as Miami.

A tsunami of 0.4 feet was recorded in the Cayman Islands at George Town, but no tsunami was observed at Port Royal, Jamaica, or Puerto Plata, Dominican Republic.
There were several aftershocks, including one the US Geological Survey said had a magnitude of 6.1.
"Based on all available data, there is no significant tsunami threat from this (6.1) earthquake. However, there is a very small possibility of tsunami waves along coasts located nearest the epicenter," the National Weather Service's Pacific Tsunami Warning Center said.
The quakes come three weeks after a magnitude 6.4 earthquake struck Puerto Rico.
''' 
document = nlp(test_text)
depeche_dataframe = prepare_depechemood(path_to_depechemood_freq)
emotions2, normalized_emotions2,depoch_word_selected_df2 = detect_feeling(document, depeche_dataframe)
