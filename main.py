import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from warnings import filterwarnings
import os  # to handle directories

# Ignore warnings
filterwarnings('ignore')

# Pandas display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# Create output directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Load dataset
df = pd.read_csv('datasets/wiki_data.csv')
df = df[:2000]  # Use first 2000 rows

# Function to clean text
def clean_text(text):
    text = text.str.lower()  # Lowercase
    text = text.str.replace(r'[^\w\s]', '')  # Remove punctuation
    text = text.str.replace("\n", '')  # Remove newlines
    text = text.str.replace('\d', '')  # Remove digits
    return text

df['text'] = clean_text(df['text'])  # Clean text

# Function to remove stopwords
def remove_stopwords(text):
    stop_words = stopwords.words('english')
    text = text.apply(lambda x: " ".join(x for x in str(x).split() if x not in stop_words))
    return text

df['text'] = remove_stopwords(df['text'])  # Remove stopwords

# Remove least common 1000 words
sil = pd.Series(' '.join(df['text']).split()).value_counts()[-1000:]
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sil))

# Lemmatization
df['text'] = df['text'].apply(lambda x: ' '.join([Word(word).lemmatize() for word in x.split()]))

# Final preprocessing + optional visualizations with save option
def wiki_preprocess(text, Barplot=False, Wordcloud=False, save_dir="outputs"):
    text = text.str.lower()
    text = text.str.replace(r'[^\w\s]', '')
    text = text.str.replace("\n", '')
    text = text.str.replace('\d', '')

    stop_words = stopwords.words('english')
    text = text.apply(lambda x: " ".join(x for x in str(x).split() if x not in stop_words))

    # Remove rare words
    sil = pd.Series(' '.join(text).split()).value_counts()[-1000:]
    text = text.apply(lambda x: " ".join(x for x in x.split() if x not in sil))

    # Lemmatize words
    text = text.apply(lambda x: ' '.join([Word(word).lemmatize() for word in x.split()]))

    if Barplot:
        tf = text.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
        tf.columns = ["words", "tf"]
        tf = tf.sort_values("tf", ascending=False)
        bar_data = tf[tf["tf"] > 2000]

        # Plot and save bar chart
        plt.figure(figsize=(10, 6))
        bar_data.plot.bar(x="words", y="tf", legend=False)
        plt.title("High Frequency Words (TF > 2000)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "barplot.png"))  # Save plot
        plt.show()

    if Wordcloud:
        all_text = " ".join(i for i in text)
        wordcloud = WordCloud(max_words=2000, background_color='white').generate(all_text)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title("WordCloud")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "wordcloud.png"))  # Save image
        plt.show()

    return text

# Apply preprocessing with visualizations and save outputs
df['text'] = wiki_preprocess(df['text'], Barplot=True, Wordcloud=True)
