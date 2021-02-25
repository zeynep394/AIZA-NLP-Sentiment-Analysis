# AIZA-NLP-Sentiment-Analysis
Sentiment Analysis Using Natural Language Processing

**Notes to Self**

Önce nltk kütüphanesini indirdik `import nltk` bu kütüphane yardımıyla:

- TreebankWordTokenizer, sent_tokenize gibi nltk modülleriyle verilen paragrafları ve cümleleri tokenize ettik.
- 
- nltk yardımıyla stopwords'leri indirdik. (Dilediğiniz dilin stopwordslerini indirebilirsiniz.)
    ```
    nltk.download('stopwords')
    stop_word_list=nltk.corpus.stopwords.words('turkish')
    ```
    
- PorterStemmer, LancasterStemmer, RegexpStemmer, SnowballStemmer gibi stemmer modülleriyle tokenize ile ayırdığımız kelimelerin köklerini bulduruyoruz.

- WordNetLemmatizer modülüyle kelimelerin eklerini ayırıyoruz. (worging-->work gibi)

- DefaultTagger  modülüyle verileri etiketleyebiliyoruz.
     ```
    from nltk.tag import DefaultTagger
    tag = DefaultTagger('Telefon no')
    tag.tag(['05545055522','05050800081'])
    ```
- spell veya textBlob gibi modüllerle hatalı yazılan kelimeleri otomatik olarak düzeltebiliriz.
    ```
    from autocorrect import spell
    spell("Tghe")
    spell("from what i have heard this is a very sirious satuation")
    ```
    veya
    ```
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
