import pandas as pd
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv('dataSets/train_set.csv', sep='\t')


f = open('out.txt', 'w')
print >> f, 'Filename:', df[["Content"]]  # or f.write('...\n')
f.close()


print df[["Content"]]

#print df[["Category"]].iloc[5:10]

A = np.array(df[["Content"]])
text = ""
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        text += str(A[i,j]) + ","

#print text
#Generate a word cloud image
wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
