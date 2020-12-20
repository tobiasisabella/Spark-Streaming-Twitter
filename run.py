#Twitter sentiment analysis using Python and Spark

#An example of how to use pyspark in a Python IDE
import findspark
findspark.init()
import pyspark
sc = pyspark.SparkContext(appName="myAppName")
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

#Imports
from pyspark.streaming import StreamingContext
from operator import add
import requests_oauthlib
import requests
import time
import string
import ast
import json
from nltk.classify import NaiveBayesClassifier
from nltk.sentiment import SentimentAnalyzer
from nltk.corpus import stopwords
from nltk.sentiment.util import *

#Update frequency
INTERVALO_BATCH = 5

#Creating the StreamingContext
ssc = StreamingContext(sc, INTERVALO_BATCH)

# Reading the text file and creating an RDD in memory with Spark
#This file contains feelings related to a book. Number 0 is negative feelings and number 1 is positive feelings. It is an example.
arquivo = sc.textFile("dataset_analysis.csv")

#Removing the header
header = arquivo.take(1)[0]
dataset = arquivo.filter(lambda line: line != header)

#This function separates the columns in each row, creates a tuple and removes the punctuation.
def get_row(line):
  row = line.split(',')
  sentimento = row[1]
  tweet = row[3].strip()
  translator = str.maketrans({key: None for key in string.punctuation})
  tweet = tweet.translate(translator)
  tweet = tweet.split(' ')
  tweet_lower = []
  for word in tweet:
    tweet_lower.append(word.lower())
  return (tweet_lower, sentimento)

#Apply the function to each row in the dataset
dataset_treino = dataset.map(lambda line: get_row(line))

#Create a SentimentAnalyzer object
sentiment_analyzer = SentimentAnalyzer()

# Get the list of stopwords in English
nltk.download("stopwords")
stopwords_all = []
for word in stopwords.words('english'):
  stopwords_all.append(word)
  stopwords_all.append(word + '_NEG')
  
#Get 10,000 tweets from the training dataset and return all non-stopwords
dataset_treino_amostra = dataset_treino.take(10000)
all_words_neg = sentiment_analyzer.all_words([mark_negation(doc) for doc in dataset_treino_amostra])
all_words_neg_nostops = [x for x in all_words_neg if x not in stopwords_all]

#Create a unigram and extract features
unigram_feats = sentiment_analyzer.unigram_word_feats(all_words_neg_nostops, top_n = 200)
sentiment_analyzer.add_feat_extractor(extract_unigram_feats, unigrams = unigram_feats)
training_set = sentiment_analyzer.apply_features(dataset_treino_amostra)

# Train the model
trainer = NaiveBayesClassifier.train
classifier = sentiment_analyzer.train(trainer, training_set)

#Tests the classifier in some sentences
test_sentence1 = [(['this', 'program', 'is', 'bad'], '')]
test_sentence2 = [(['tough', 'day', 'at', 'work', 'today'], '')]
test_sentence3 = [(['good', 'wonderful', 'amazing', 'awesome'], '')]
test_set = sentiment_analyzer.apply_features(test_sentence1)
test_set2 = sentiment_analyzer.apply_features(test_sentence2)
test_set3 = sentiment_analyzer.apply_features(test_sentence3)

#Twitter authentication
consumer_key = "XXXXXXXXXXXXXXXXXXXXXXX"
consumer_secret = "XXXXXXXXXXXXXXXXXXXXXXXXXXX"
access_token = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
access_token_secret = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

#Specifies the search term
search_term = 'Example'
sample_url = 'https://stream.twitter.com/1.1/statuses/sample.json'
filter_url = 'https://stream.twitter.com/1.1/statuses/filter.json?track='+search_term

#Creating the authentication object for Twitter
auth = requests_oauthlib.OAuth1(consumer_key, consumer_secret, access_token, access_token_secret)

#Configuring the Stream
rdd = ssc.sparkContext.parallelize([0])
stream = ssc.queueStream([], default = rdd)

#Total tweets per update
NUM_TWEETS = 50

#This function connects to Twitter and returns a specific number of Tweets (NUM_TWEETS)
def tfunc(t, rdd):
  return rdd.flatMap(lambda x: stream_twitter_data())
def stream_twitter_data():
  response = requests.get(filter_url, auth = auth, stream = True)
  print(filter_url, response)
  count = 0
  for line in response.iter_lines():
    try:
      if count > NUM_TWEETS:
        break
      post = json.loads(line.decode('utf-8'))
      contents = [post['text']]
      count += 1
      yield str(contents)
    except:
      result = False      
stream = stream.transform(tfunc)
coord_stream = stream.map(lambda line: ast.literal_eval(line))

#This function classifies tweets, applying the features of the template created earlier
def classifica_tweet(tweet):
  sentence = [(tweet, '')]
  test_set = sentiment_analyzer.apply_features(sentence)
  print(tweet, classifier.classify(test_set[0][0]))
  return(tweet, classifier.classify(test_set[0][0]))
  
#This function returns the Twitter text
def get_tweet_text(rdd):
  for line in rdd:
    tweet = line.strip()
    translator = str.maketrans({key: None for key in string.punctuation})
    tweet = tweet.translate(translator)
    tweet = tweet.split(' ')
    tweet_lower = []
    for word in tweet:
      tweet_lower.append(word.lower())
    return(classifica_tweet(tweet_lower))

# Create an empty list for the results  
resultados = []

#This function saves the result of batches of Tweets along with the timestamp
def output_rdd(rdd):
  global resultados
  pairs = rdd.map(lambda x: (get_tweet_text(x)[1],1))
  counts = pairs.reduceByKey(add)
  output = []
  for count in counts.collect():
    output.append(count)
  result = [time.strftime("%I:%M:%S"), output]
  resultados.append(result)
  print(result)

# The foreachRDD () function applies a function to each RDD to streaming data 
coord_stream.foreachRDD(lambda t, rdd: output_rdd(rdd))

#Start streaming
ssc.start()

#Determines the number of batches to be read
cont = True
while cont:
  if len(resultados) > 5:
    cont = False
    
#Save the results
rdd_save = '/path/'+time.strftime("%I%M%S")
resultados_rdd = sc.parallelize(resultados)
resultados_rdd.saveAsTextFile(rdd_save)

#View the results
resultados_rdd.collect()

# End streaming
ssc.stop()
