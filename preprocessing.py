from numpy.core.defchararray import index
import pandas as pd
import numpy as np
import re
import emoji
import nltk
import stopwords
nltk.download('words')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
import itertools


#separates qoutation marks from text by inserting a space. e.g.: "Hello World" -> " Hello World "
def separateQoutation(text):
    insertStr = " "
    qoutationStr = "\""
    if(text.find(qoutationStr) != -1):
        idx = text.index(qoutationStr)
        seperatedtext = text[:idx] + insertStr + text[idx] + insertStr + text[idx+1:] 
        return seperatedtext
    else:
        return text
    
#separates dost from text by inserting a space. e.g.: Hello World. ->  Hello World . 
def separateDot(text):
    insertStr = " "
    dotStr = "."
    
    if(text.find(dotStr) != -1):
        idx = text.index(dotStr)
        seperatedtext = text[:idx] + insertStr + text[idx] + insertStr + text[idx+1:] 
        return seperatedtext
    else:
        return text
    
#basic text clean up, removes unnecessary characters. e.g.: <Hello World^ -> Hello World
def removeNoise(text):
    text = re.sub("(<.*?>)","",text)

    text = re.sub("(\\W|\\d)"," ",text)

    text = text.strip()

    return text

#sets all characters to lower case e.g.: Hello World -> hello world. 
def toLowerCase(text):
    text = text.lower()

    return text

#replaces emojis with their name. Returns string with emojis replaced.
def replaceEmoji(text):

    text = emoji.demojize(text)

    return text

#replaces all words staring with "@". Returns string without usernames.
def removeUsername(text):
    
    text = re.sub("(@)\w+","",text)

    return text

#replaces all words starting with "http" with "Weblink" e.g.: https:\\hello.world.com -> Weblink. Returns string with replaced weblinks.
def replaceLinks(text):
    
    text = re.sub("(http)\S+","WebLink",text)
    
    return text

#removes stop words from text, according to the Natural Language Toolkit libary. Returns string without stopwords.
def removeStopWords(text):    
    text_tokens = tokenize(text)

    stopwordsdict = stopwords.get_stopwords('en')

    tokens_no_sw = [word for word in text_tokens if not word in stopwordsdict]

    filtered_text = untokenize(tokens_no_sw)

    return filtered_text

#splits contractions into their respective words e.g.: I'll -> I will. Returns string with split contractions.
def splitContractions(text):

    contractions = { 
    "ain't": "am not / are not / is not / has not / have not",
    "aren't": "are not / am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": "he shall have / he will have",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is / how does",
    "I'd": "I had / I would",
    "I'd've": "I would have",
    "I'll": "I shall / I will",
    "I'll've": "I shall have / I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it shall / it will",
    "it'll've": "it shall have / it will have",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she shall / she will",
    "she'll've": "she shall have / she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what're": "what are",
    "what's": "what has / what is",
    "what've": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "who'll've": "who shall have / who will have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why has / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you shall / you will",
    "you'll've": "you shall have / you will have",
    "you're": "you are",
    "you've": "you have"
    }

    pattern = re.compile(r'(?<!\w)(' + '|'.join(re.escape(key) for key in contractions.keys()) + r')(?!\w)')
    result = pattern.sub(lambda x: contractions[x.group()], text)


    return text

#lemmatizes words according to the Natural Language Toolkit libraries for wordstems e.g.: learning -> learn. Returns lemmatized string.
def lemmatize(text):
    
    lemmatizer = nltk.stem.WordNetLemmatizer()

    text_tokens = tokenize(text) 

    tokens_lemmatized = [lemmatizer.lemmatize(word) for word in text_tokens]
    
    lemmatized_text = untokenize(tokens_lemmatized)

    return lemmatized_text


#tokenizes words in string. Returns list of word tokens.
def tokenize(text):

    mytokenizer = TweetTokenizer()

    text_tokens = mytokenizer.tokenize(text)

    return text_tokens

#joins tokens and returns string.
def untokenize(tokens):

    text = (" ").join(tokens)

    return text

#compares labels for each row where labels in columns 'columnname1' and 'columnname2' disagree and replaces both columns containing labels with new column calles 'label'. Prioritiztes 'columnname2' during disagreement. Returns number of unique labels after joining.
def joinLabels(columnname1, columnname2):
    joinedLabel = np.where(df[columnname1] == 'UNKNOWN',df[columnname2],df[columnname2])
    df['label'] = joinedLabel
    df.drop(columns=[columnname1,columnname2],axis=1,inplace=True)
    return df['label'].value_counts()

#drops rows with missing data. Returns number of unique values in column after removing missing data.
def dropMissingData(columnname, Value):
    indexNames = df[df[columnname] == Value].index
    df.drop(indexNames,inplace=True)
    
    return df[columnname].value_counts()

#filters text for unreal words according to Natual Language Processing libary. Returns filtered text.
def filterWords(text):
    listText = text.split()
    words = set(nltk.corpus.words.words())
    filteredText = [word for word in listText if(word in words)]
    returnText = (" ").join(filteredText)
    return returnText

#applies preprocessing steps to dataframe, with option of selecting only groups of preprocessing steps. Returns preprocessed dataframe.
def process(standardize,cleanup,removal,df):
    returndf = df
    
    if(standardize == True): 
        returndf['text'] = returndf['text'].apply(lambda x:replaceEmoji(x))
        returndf['text'] = returndf['text'].apply(lambda x:toLowerCase(x))
        returndf['text'] = returndf['text'].apply(lambda x:splitContractions(x))       
        returndf['text'] = returndf['text'].apply(lambda x:replaceLinks(x))

    print('standardize finished')        

    if(cleanup == True):    
        returndf['text'] = returndf['text'].apply(lambda x:removeNoise(x)) 
        returndf['text'] = returndf['text'].apply(lambda x:separateDot(x)) 
        returndf['text'] = returndf['text'].apply(lambda x:separateQoutation(x)) 
    
    print('cleanup finished')


    if(removal == True):  
        returndf['text'] = returndf['text'].apply(lambda x:removeUsername(x)) 
        returndf['text'] = returndf['text'].apply(lambda x:removeStopWords(x)) 
        returndf['text'] = returndf['text'].apply(lambda x:filterWords(x)) 


    returndf['text'] = returndf['text'].apply(lambda x:lemmatize(x))

    return returndf

#applies countvectorization to a dataframe and returns the vectorized dataframe. Minoccurence sets the number of occurences a word has to appear in the dataframe to be included in the vectorization.
def vectorize(minoccurence,dframe):
    df.dropna(axis=0,inplace=True)


    cv = CountVectorizer(binary=False, lowercase=False, stop_words=None, min_df=minoccurence, ngram_range=(1,1))


    df2 = pd.DataFrame(cv.fit_transform(dframe['text']).toarray(), columns=cv.get_feature_names())
    df2.insert(0,'label',dframe['label'])
    df2.insert(1,'retweet_count',dframe['retweet_count'])
    df2.insert(2,'user_verified',dframe['user_verified'])
    df2.insert(3,'user_friends_count',dframe['user_friends_count'])
    df2.insert(4,'user_followers_count',dframe['user_followers_count'])
    df2.insert(5,'user_favourites_count',dframe['user_favourites_count'])
    df2.insert(6,'tweet_source',dframe['tweet_source'])
    df2.insert(7,'geo_coordinates_available',dframe['geo_coordinates_available'])
    df2.insert(8,'num_hashtags',dframe['num_hashtags'])
    df2.insert(9,'num_mentions',dframe['num_mentions'])
    df2.insert(10,'num_urls',dframe['num_urls'])
    df2.insert(11,'num_media',dframe['num_media'])
    dfencoded = df2
    return dfencoded

  

df = pd.read_csv()

df.drop(columns=['Unnamed: 0','fake_news_category_2'], inplace=True)
processedDF = process(True,True,False,df)
vectorizedDF = vectorize(2,processedDF)  

vectorizedDF.to_csv()
