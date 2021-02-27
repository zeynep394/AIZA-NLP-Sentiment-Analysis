# AIZA-NLP-Sentiment-Analysis
Sentiment Analysis Using Natural Language Processing

## Natural Language Processsing Kullanarak Sentiment Analizi yapma

 Bu ödevde bizden istenilen Bacchanal Buffet diye bilinen Restoranın müşteri yorum verilerini alarak yorumları analiz etmemiz. Bu yorumlara göre restoranın gelişebilmesi için NLP'ye dayalı analizler yapacağız.
 
 İzleyeceğimiz adımlar:
 
 - Veriye ulaşma
 - Veri Analizi
 - Dil Tanıma
 - Gereksiz Karakterlerden Kurtulma
 - Stemma ve Lemma uygulama
 - Veriyi x ve y olmak üzere iki verisetine böl ve train ve test değerlerini ayarlama
 - Vectorizer işlemlerini uygulama
 - Classifier algoritmalarını kullanarak sınıflandırma
### Veriye Ulaşma

### Veri Analizi

Veri setinde bulunan stars sütünu, bize müşterilerin restoran için verdiği puan sayısını gösterir (5-en iyi, 1-en kötü). Kaç farklı puanlama sayısı olduğunu bulmak için pandas kütüphanesinin [.unique()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.unique.html) fonksiyonundan yararlanıyoruz;
```python
df.stars.unique()

>>array([5, 4, 1, 3, 2], dtype=int64)
```
Puanların dağılımını görmek içinse .value_counts() fonksiyonunu kullanabiliriz.

```python
df.stars.value_counts()

>>5    4247
  4    2636
  3    1561
  1    1056
  2     917
  Name: stars, dtype: int64
  ```
**Grafik Çizme Üzerine Küçük Notlar**

  ```python
  plt.figure(figsize=(8,8))
  df['stars'].value_counts().plot.pie(autopct='%1.1f%%', startangle=60)
  plt.title('Yıldızların Dağılım Grafiği')
```
Bize puanların dağılımını gösteren daire grafiğini verir.
figsize: çizilecek daire grafiğinin genişliği ve boyunu belirler
 `df['stars'].value_counts().plot.pie(autopct='%1.1f%%', startangle=60)`: daire grafiğinin neye göre çizileceğini belirler. Bu örnek için daire dilimi df dataframeinin stars sütunundaki değerlerin dağılımını gösterecektir. 
 
 
```python
sns.countplot(df['stars'])

df['year'] = pd.DatetimeIndex(df['date']).year
df['year']
df['month'] = pd.DatetimeIndex(df['date']).month
df['month']

sns.barplot(x=df.year , y=df.stars);

sns.barplot(x=df.month , y=df.stars);

stars=pd.DataFrame(df['stars'].value_counts(normalize=True).round(decimals=2))
stars
```
Sırasıyla  yıllara ve aylara göre müşteriler tarafından verilen puanların dağılım grafikleri sns yardımıyla çiziliyor.


**Dataframe Üzerinden Devam Edelim**

Müşterilerin puanları üzerinde sentiment analizi yapabilmek için puanları 3 -> nötr, 4 ve 5 -> pozitif, 1 ve 2 -> negatif olacak şekilde 3 gruba ayırıyoruz ve bunları yeni oluşturulan 'sentiment' sütununa atıyoruz:

```python
df.loc[df['stars'] == 3, 'sentiment'] = 'neutral' 
df.loc[df['stars'] < 3, 'sentiment'] = 'negative' 
df.loc[df['stars'] > 3, 'sentiment'] = 'positive'
```

Veri setinde bize sunulan metin, puan ve sentiment harici diğer değerler işimize yaramayacak olan veriler. Bunun için df dataframeindeki text, sentiment ve stars sütunlarını alarak yelp isimli başka bir dataframe oluşturuyoruz:

```python
yelp= df[['stars', 'sentiment','text']]
```

### Dil Tanıma 

```python
from langdetect import detect
yelp=yelp[yelp['text'].apply(detect)=='en']
```
Verimiz global bir veri setinden alındığı için bazı müşteri yorumları ingilizce değil. Bu tarz yorumlarla başa çıkmanın iki yolu var; birincisi, langdetect modülünün sunduğu detect fonksiyonuyla yorumların dillerini algılayarak ingilizce dilinde olmayan yorumları dataframeden çıkarma. ikincisi, detect fonksiyonuyla yorumların dilini belirleyip ingilizce olmayan yorumları tespit ederek textBlob yardımıyla yorumları ingilizceye tercüme etme. Biz bu ödev için birinci şıkkı kullanarak langdetect kütüphanesinden detect fonksiyonunu indirdiğimizde sadece "en" ingilizce kodlu yorumları seçerek yelp dataframeini düzenliyoruz.


### Removing Characters

Daha temiz bir metin verisiyle çalışabilmek için yorumlar içinde geçen rakamlar, boşluklar, noktalama işaretleri gibi gereksiz karakterleri verimizden çıkarıyoruz.
```python
yelp['text']=yelp['text'].str.lower()
yelp['text']=yelp['text'].str.replace('[^\w\s]','')
yelp['text']=yelp['text'].str.replace('\d+','') #removing numerals 
yelp['text']=yelp['text'].str.replace('\n',' ').str.replace('\r','')
```

### Polarity-Subjectivity
Sentiment analizine başlamadan önce bu analizi kolaylaştırmak için polarity yöntemini kullanacağız. Bu yöntemle elimizdeki yorumların negatife mi yakın pozitife mi yakın olduklarını 1 ile -1 arasında değişen sayısal değerler ile ölçeceğiz. 

```python
sentiment_series = yelp['polarity'].tolist()
#column=['polaarity', 'subjectivity']
yelp[['polaarity', 'subjectivity']] = pd.DataFrame(sentiment_series,index=yelp.index)
yelp.drop('polarity', inplace=True, axis=1)
```
![Alt text](https://github.com/zeynep394/AIZA-NLP-Sentiment-Analysis/blob/main/pol-sub.png)

### Word Cloud

Yorumlara bakarak pozitif yorum yapan müşterilerin beğendikleri ortak özellikleri tanımlayabilmek için en çok kullanılan kelimeleri analiz edeceğiz bu analizi görselleştirmenin bir yolu da wordcloud kullanmak. Word cloud farklı şekillerin kullanılmasına imkan verdiği için en çok tercih edilen ve görsel zevklere en iyi hitap eden görselleştirme şekli.

```python
from wordcloud import WordCloud

wc = WordCloud( background_color="white", colormap="Dark2",
               max_font_size=150, random_state=42)
```             
Wordcloud'u import edip görünüşünü tanımladıktan sonra wordcloud'vereceğimiz datayı hazırlamaya geçelim;

```python
text_5_star=yelp[yelp['stars']>3]
text_5_star
text_5_star_review = " ".join(review for review in text_5_star.text)

text_1_star=yelp[yelp['stars']<3]
text_1_star
text_1_star_review = " ".join(review for review in text_1_star.text)
```

```python
wine_mask = np.array(Image.open("cloud.png"))
wc = WordCloud(background_color="white", max_words=1000, mask=wine_mask,
                contour_width=3, contour_color='firebrick')

# Generate a wordcloud
wc.generate(text_5_star_review)

# store to file
wc.to_file("cloud2.png")

# show
plt.figure(figsize=[20,10])
plt.imshow(wc)
plt.axis("off")
plt.title('5 Star Review')
plt.show()


# Generate a wordcloud
wc.generate(text_1_star_review)

# store to file
wc.to_file("cloud3.png")

# show
plt.figure(figsize=[20,10])
plt.imshow(wc)
plt.axis("off")
plt.title('1 Star Review')
plt.show()
```
### Classification

```python
yelp['text']=yelp['text'].str.replace('[^a-zA-Z]',' ')

yelp['clean_text'] = yelp['text'].apply(lambda x: nltk.word_tokenize(x) ) 

from nltk.stem import WordNetLemmatizer
lemmetizer_output= WordNetLemmatizer()
  
yelp['stem_text'] = yelp['clean_text'].apply(lambda x: [lemmetizer_output.lemmatize(j) for j in x if not j in set(STOP_WORDS)] )

yelp['cleaned_text'] = yelp['stem_text'].apply(lambda x: ' '.join(x) )
```
```python
x=yelp.cleaned_text #text sütunu x oluyor
y=yelp.sentiment #dataframein stars sütunu y oluyor 

print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,random_state=42)

vect = CountVectorizer(lowercase=True, stop_words='english')
x_train_dtm= vect.fit_transform(x_train) #yazıyı alıp vektöre çeviriyor.
#yazıyı textten vektöre tarnsfrom ettiğimiz için fit_transform kullanıyoruz
print(x_train_dtm)

x_test_dtm=vect.transform(x_test)

tf=pd.DataFrame(x_train_dtm.toarray(),columns=vect.get_feature_names())

vect = CountVectorizer(ngram_range=(1,2))
x_train_dtm = vect.fit_transform(x_train)
x_train_dtm.shape
vect = CountVectorizer()

x_train_dtm = vect.fit_transform(x_train)
x_test_dtm = vect.transform(x_test)
```
```python
models=[gbc,d,k,g,b,r,log,xgbc]
def tokenize_test(models,vect):
    row_list=[]
    for i in range(len(models)):
        x_train_dtm = vect.fit_transform(x_train)
        x_test_dtm = vect.transform(x_test)
        nb = models[i]
        nb.fit(x_train_dtm, y_train)
        y_pred_class = nb.predict(x_test_dtm)
        
        data={'accuracy_score':metrics.accuracy_score(y_test, y_pred_class)}
        
        row_list.append(data)
        #print ('Accuracy: ', metrics.accuracy_score(y_test, y_pred_class))
    df = pd.DataFrame(row_list, index =['gbc', 'd', 'k', 'g','b','r','log','xgbc']) 
    return df
    
vect = CountVectorizer(ngram_range=(1,2))
tokenize_test(models,vect)
```

not: classification algoritmalarını, text verisisni vektörize ettikten sonra textlerin sentiment analizini yapmak için kullanıyoruz. Böylece yeni birisi yorum yaptığı zaman bu yorumun pozitif mi nötr mü negatif mi olduğunu anlayabiliyoruz.  

- Getting rid of the punctuation characters, because they behave as the noise inside the text data because they hold no specific semantic meaning.
- Getting rid of the numbers and numerals, because we want to perform a qualitative analysis (positive or negative) instead of any sort of quantitative analysis involving numbers.
- Removing all the words with less than or equal to three characters, because such words can either be stopwords, or they can be the words acting as slang terms.
- Removing stopwords like and, or, of, the, is, am, had, etc. Stopwords are so common, and hold so little semantic information, that their removal is favorable because it not only reduces the dimensionality of the vector-space model, but also increases accuracy in classification in most cases.
- Converting the entire text into lowercase, because a computer is going to interpret “Awesome” and “awesome” as two different terms, because of the difference in the encodings of lowercase and uppercase ‘A’ and ‘a’.
- Applying Porter Stemmer. Stemming is one important phase, because it reduces the words or terms to their roots. For example, the term “faster” gets converted into “fast”, “recommended” to “recommend”, “eating” to “eat”, etc. This helps in retaining the semantic meaning of the sentence while simplifying the repetition.
- Grouping terms by documents.
- Extracting keywords from the grouped terms, because we don’t want every other word to be qualified as a feature. We just consider the most important terms in our dataset, and only keep the qualifying keyword terms.
Converting extracted keywords into sparse vectors, or what I call “vectorization”.

**Notes to Self**

Önce nltk kütüphanesini indirdik `import nltk` bu kütüphane yardımıyla:

- TreebankWordTokenizer, sent_tokenize gibi nltk modülleriyle verilen paragrafları ve cümleleri tokenize ettik.
 
- nltk yardımıyla stopwords'leri indirdik. (Dilediğiniz dilin stopwordslerini indirebilirsiniz.)
    ```python
    nltk.download('stopwords')
    stop_word_list=nltk.corpus.stopwords.words('turkish')
    ```
    
- PorterStemmer, LancasterStemmer, RegexpStemmer, SnowballStemmer gibi stemmer modülleriyle tokenize ile ayırdığımız kelimelerin köklerini bulduruyoruz.

- WordNetLemmatizer modülüyle kelimelerin eklerini ayırıyoruz. (worging-->work gibi)

- DefaultTagger  modülüyle verileri etiketleyebiliyoruz.
    ```python
    from nltk.tag import DefaultTagger
    tag = DefaultTagger('Telefon no')
    tag.tag(['05545055522','05050800081'])
    ```
- spell veya textBlob gibi modüllerle hatalı yazılan kelimeleri otomatik olarak düzeltebiliriz.
    ```python
    from autocorrect import spell
    spell("Tghe")
    spell("from what i have heard this is a very sirious satuation")
    ```
    veya
    ```python
    from textblob import TextBlob
    b=TextBlob("from what i have heard this is a very sirius satuation")
    print(b.detect_language())
    print(b.correct())
    ```

### NLP Projesinde yapılması gereken adımlar
* Bütün cümleler küçük harfe çevirilecek
* Noktlama işaretleri kaldır
* Rakamları kaldır
* Satır sonu, \n veya \r gibi karakterleri kaldır
* Stop words kaldır (gereksiz kelimeler yardıcı fiiller am are gibi) 
* Tokenize etmek (kelimeleri ayır)
* Lema ve stemma uygula *ekleri kaldırıp kökleri bulma işlemi*
* Vectorizer ile yazıları rakama atıyoruz.
* Sentiment analizine başlıyoruz.
