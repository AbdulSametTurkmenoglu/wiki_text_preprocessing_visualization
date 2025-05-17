# Wiki Text Preprocessing & Visualization

This project demonstrates how to preprocess Wikipedia-based text data and visualize the most frequent words using both **Bar Plot** and **WordCloud** techniques.

##  Dataset

The code expects a CSV file located at:
``datasets/wiki_data.csv``

Make sure your CSV contains a `text` column with raw textual content.

##  Features

- Lowercasing
- Removal of:
  - Punctuation
  - Numbers
  - Newline characters
  - Stopwords
  - Least frequent 1000 words
- Word Lemmatization
- Term Frequency (TF) visualization:
  - Bar Plot for words with frequency > 2000
  - WordCloud of most common words
- Automatic saving of plots as PNG files in the `outputs/` folder

##  Requirements

Install the following Python packages before running the script:

```bash
pip install pandas matplotlib wordcloud nltk textblob
```

```python
import nltk
nltk.download('stopwords')
```
```bash
python your_script_name.py
```

##  Output

- outputs/barplot.png: Bar plot of high-frequency terms

![Bar plot](outputs/barplot)

- outputs/wordcloud.png: Word cloud of the most common words
![Word cloud](outputs/wordcloud)









