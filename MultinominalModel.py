from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
import nltk
import pandas as pd

lemmatizer = WordNetLemmatizer()

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('all-corpora')

stop_words = set(stopwords.words('english')) 

df = pd.read_csv("amazon_reviews.csv")

# Preprocess text data
def preprocess(review):
    review = review.lower()
    tokens = word_tokenize(review)
    lemmas = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(lemmas)


# Format csv data into array of [review, rating]
review_ratings = []
for i in range(len(df)):
    review_text = str(df.loc[i]["reviewText"])
    rating = int(df.loc[i]["overall"])
    review_ratings.append([review_text, rating])

# Create corpus of preprocessed text
corpus = []
for i in range(len(review_ratings)):
    review = review_ratings[i][0]
    rating = review_ratings[i][1]
    preprocessed_text = preprocess(review)
    corpus.append(preprocessed_text)


# Convert to vector representation
vectorizer = TfidfVectorizer(max_features=10000)
X = vectorizer.fit_transform(corpus).toarray()
y = [r[1] for r in review_ratings]

# Generate synthetic samples as 5 star rating reviews are overbalanced
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Create model and fit
model = MultinomialNB()
model.fit(X_train, y_train)

y_predict = model.predict(X_test)
print("Accuracy", accuracy_score(y_test, y_predict))

def predict_rating(review):
    preprocessed_text = preprocess(review)
    vectorized = vectorizer.transform([preprocessed_text]).toarray()
    return model.predict(vectorized)