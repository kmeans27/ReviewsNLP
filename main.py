#import all the needed libraries
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from nltk.stem import WordNetLemmatizer


#open the file and save the data as a pandas dataframe
reviews = pd.read_csv("fire_hd_reviews_pre_cleaned.csv", on_bad_lines='skip', nrows=100)
#check if the shape is correct
print("Dataframe shape:", reviews.shape)


#clean the data - remove upper case letters, symbols and stopwords
corpus = []
#ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
for w in range (0, len(reviews)):
    title = re.sub('[^a-zA-Z]', ' ', reviews['Reviews'][w])
    title = title.lower()
    title = title.split()
    #title = [ps.stem(word) for word in title if not word in stopwords.words('english')]
    title = [lemmatizer.lemmatize(word) for word in title if not word in stopwords.words('english')]
    title = ' '.join(title)
    corpus.append(title)
print(corpus[0])


#create the Bag of Words - BOW corpus
bow = CountVectorizer().fit(corpus)
vocabulary = bow.get_feature_names()
vect = bow.transform(corpus)
#print(vocabulary)


#view the Bag of Words corpus as a pandas dataframe
df_count_vocabulary = pd.DataFrame(data=vect.toarray(), columns = vocabulary)
#print(df_count_vocabulary)


#create the Term Frequency - Inverse Document Frequency from the cleaned corpus
vectorizer_test = TfidfVectorizer(min_df=1)
model_test = vectorizer_test.fit_transform(corpus)
data = pd.DataFrame(model_test.toarray(), columns=vectorizer_test.get_feature_names())
#print(data)


#semantic analysis
#Latent Semantic Analysis
vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True)
model = vectorizer.fit_transform(corpus)
LSA_model = TruncatedSVD(n_components=20, algorithm="randomized", n_iter=10)
lsa = LSA_model.fit_transform(model)
# print("Review 0 :")
# for i, topic in enumerate(lsa[0]):
#     print("Topic ", i, " : ", topic*100)


#Latent Dirichlet Allocation
lda_model = LatentDirichletAllocation(n_components=20, learning_method="online", random_state=50,)
lda_top = lda_model.fit_transform(model)

# print("Review 1: ")
# for i, topic in enumerate(lda_top[0]):
#     print("Topic ", i, ": ", topic*100, "%")
# if __name__ == '__main__':
#     pass
