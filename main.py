import pandas as pd
import numpy as np
import os
import Image as img
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# read trainset

df = pd.read_csv('dataSets/train_set.csv', sep='\t')

my_additional_stop_words=['said','th','month','much','thing','say','says'] # also: new, it's ?
s?
stop_words = ENGLISH_STOP_WORDS.union(my_additional_stop_words)

# create dir for wordclouds if it doesn't exist

wordcloud_dir = "wordClouds/"

if not os.path.exists(wordcloud_dir):
    os.makedirs(wordcloud_dir)

print "\nGenerating wordClouds... \n"  

# create wordcloud for every category

for category in df.Category.unique():
    print category
    text = df.loc[df['Category'] == category]
    textString = text.Content.to_string()

    wordcloud = WordCloud(stopwords=stop_words,
                          background_color='white',
                          width=1200,
                          height=1000,
                         ).generate(textString)

    wordcloud.to_file(wordcloud_dir + category + ".png")
    
    # show in terminal ?

    fig = plt.gcf()
    fig.canvas.set_window_title(category)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

