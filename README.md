# AIZA-NLP-Sentiment-Analysis
Sentiment Analysis Using Natural Language Processing

**Notes to Self**
- Önce nltk kütüphanesini indirdik `import nltk` bu kütüphane yardımıyla:
-TreebankWordTokenizer, sent_tokenize gibi nltk modülleriyle verilen paragrafları ve cümleleri tokenize ettik.
-nltk yardımıyla stopwords'leri indirdik. (Dilediğiniz dilin stopwordslerini indirebilirsiniz.)
    ```
    nltk.download('stopwords')
    stop_word_list=nltk.corpus.stopwords.words('turkish')
    ```
-PorterStemmer, LancasterStemmer, RegexpStemmer, SnowballStemmer gibi stemmer modülleriyle tokenize ile ayırdığımız kelimelerin köklerini bulduruyoruz.
-WordNetLemmatizer modülüyle kelimelerin eklerini ayırıyoruz. (worging-->work gibi)
-DefaultTagger  modülüyle verileri etiketleyebiliyoruz.
     ```
    from nltk.tag import DefaultTagger
    tag = DefaultTagger('Telefon no')
    tag.tag(['05545055522','05050800081'])
    ```
    -
