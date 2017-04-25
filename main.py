import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

df = pd.read_csv('dataSets/train_set.csv', sep='\t')

my_additional_stop_words=['example1','example2']  #our custom stop words
stop_words = ENGLISH_STOP_WORDS.union(my_additional_stop_words)

#create image for every category
for category in df.Category.unique():
    print category
    text = df.loc[df['Category'] == category]
    textString=text.Content.to_string()
    textString = ' '.join([word for word in textString.split() if word not in stop_words])
    wordcloud = WordCloud().generate(textString)
    print wordcloud
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
