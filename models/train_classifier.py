# import libraries
import sys
# Libraries to manipulate the data
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine

# Libraries to do the natural language processing
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import pos_tag

# Libraries to create the machine leaning model
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier

# Libraries to evaluate the model
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score, classification_report


def load_data(database_filepath):
    '''
    Description: Load the data from the database
    
    INPUT:
        the database name

    OUTPUT:
        a X, y array (features and labels)
    '''
    # load data from database
    engine = create_engine(f'sqlite:///...data/{database_filepath}.db')
    df = pd.read_sql('DisasterResponse', engine)

    # Splitting the dataframe in Features (X) and Labels (y, variables we want to predict)
    X = df.message
    y = df.iloc[:,3:].values

    return X, y

def tokenize(text):
        '''
    This functions transform the raw text data in tokens (words) to the correct format for we use them to machine learning.
    
    INPUT:
    Text = string
    
    OUTPUT:
    Tokens = Strings
    '''
    # Removing URLs:
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = re.sub(url, 'urlplaceholder', text )
    
    # Normalize case and remove punctuation
    txt = re.sub('\W', ' ', text.lower())
    
    # Tokens
    tokens = word_tokenize(txt)
    
    # Lemmatize, remove stop words and white spaces
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    tokens = [lemmatizer.lemmatize(word).strip() for word in tokens if word not in stop_words]
    
    return tokens


def build_model():
    '''
    Description:
    This function trains the model

    INPUT:
    No inputs

    OUTPUT:
    The machine learning model
    '''
    # Splitting the data in train and test
    X, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(X, y)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', RandomForestClassifier())
])
    # Parameters to be tested
    parameters = {
                'clf__min_samples_split':[8,20],
                'clf__n_estimators':[200,400]              
            }
    # Training the model
    cv = GridSearchCV(pipeline, parameters)

    cv.fit(x_train, y_train)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Description:
    This function is used to predict features in the test dataset,
    and will display the results.
    
    INPUT:
    Machine learning model, Features for test,
    labels test, and category names

    OUTPUT:
    Thre predict result
    '''
    # Predicting
    y_pred = model.predict(X_test)

    # Display the results
    print(classification_report(Y_test, y_pred, target_names=category_names, zero_division=1))


def save_model(model, model_filepath):
    '''
    Description:
    This function saves the model

    INPUT:
    the model and the path to save the model

    OUTPUT:
    No outputs
    '''

    # Exporting the model
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()