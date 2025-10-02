import streamlit as st
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

# -------------------------
# Feature extraction
# -------------------------
def has_ip_address(url):
    ip_pattern = re.compile(
        r'(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)'
    )
    return 1 if ip_pattern.search(url) else 0

def count_digits(s):
    return sum(c.isdigit() for c in s)

def count_tokens(s):
    tokens = re.split(r'[^a-zA-Z0-9]', s)
    tokens = [t for t in tokens if t]
    return len(tokens)

def domain_length(url):
    try:
        return len(urlparse(url).netloc)
    except:
        return 0

def path_length(url):
    try:
        return len(urlparse(url).path)
    except:
        return 0

def suspicious_words_count(url):
    suspicious = [
        'login', 'signin', 'secure', 'account', 'update',
        'free', 'click', 'verify', 'bank', 'confirm',
        'paypal', 'wallet', 'ebay', 'dropbox'
    ]
    s = url.lower()
    return sum(1 for w in suspicious if w in s)

# -------------------------
# Pipeline helpers
# -------------------------
class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key='url'):
        self.key = key
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.key].astype(str).values

class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, key='url'):
        self.key = key
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        urls = X[self.key].astype(str).values
        feats = []
        for u in urls:
            arr = [
                len(u),
                domain_length(u),
                path_length(u),
                has_ip_address(u),
                count_digits(u),
                count_tokens(u),
                suspicious_words_count(u)
            ]
            feats.append(arr)
        return np.array(feats)

class PipelineWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, transformer, inner):
        self.transformer = transformer
        self.inner = inner
    def fit(self, X, y=None):
        arr = self.transformer.transform(X)
        self.inner.fit(arr, y)
        return self
    def transform(self, X):
        arr = self.transformer.transform(X)
        return self.inner.transform(arr)

# -------------------------
# Streamlit UI
# -------------------------
st.title("🛡️ Malicious URL Detector (ML)")

# Load dataset
csv_path = st.text_input("Dataset path", value="data/malicious_urls_sample.csv")
df = pd.read_csv(csv_path)
df['label'] = df['type'].map(lambda x: 0 if str(x).lower()=='benign' else 1)
df = df.dropna(subset=['url','label']).reset_index(drop=True)

# Train model
X = df[['url']]
y = df['label'].values

tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,6), max_features=2000)
union = FeatureUnion([
    ('tfidf', PipelineWrapper(TextSelector('url'), tfidf)),
    ('url_feats', PipelineWrapper(URLFeatureExtractor('url'), StandardScaler()))
])
X_trans = union.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_trans, y, test_size=0.2, random_state=42)
clf = SVC(kernel='linear', probability=True, random_state=42)
clf.fit(X_train, y_train)

# Predict URL
st.subheader("Check a URL")
url_input = st.text_input("Enter URL to check", value="https://www.google.com")
if st.button("Predict URL"):
    df2 = pd.DataFrame([{'url': url_input}])
    X_t = union.transform(df2)
    pred = clf.predict(X_t)[0]
    prob = clf.predict_proba(X_t)[0][1]
    label = "Malicious" if pred==1 else "Benign"
    if pred==1:
        st.error(f"⚠️ Prediction: {label}, probability = {prob:.3f}")
    else:
        st.success(f"✅ Prediction: {label}, probability = {prob:.3f}")
