
from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf(X_train, X_test):
 
    tfidf = TfidfVectorizer(max_df=0.85, stop_words='english', max_features=5000)

 
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf, tfidf
