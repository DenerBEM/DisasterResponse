# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    '''
    Description: This function loads two datasets and makes 
    an inner merge using the column 'id' as key.
    INPUT:
    Two files paths in csv format.
    OUTPUT:
    The dataframe merged.
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')

    return df


def clean_data(df):
    '''
    Description: This function cleans the dataframe, remove text
    from the labels, split the labels and expand them in many columns,
    rename the columns, drop unnecessaries columns,
    remove missing values and duplicated values.
    INPUT:
    The dataframe from the function load
    OUTPUT:
    A clean dataframe
    '''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0, :]

    #extract a list of new column names for categories.
    category_colnames = row.str.split(' - ', expand=True)[0]

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].\
            str.split(' - ', expand=True)[1]
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Replace categories column in df with new category columns.
    df.drop(columns='categories', inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # Drop missing values
    df.dropna(inplace=True)

    # Convert float to int
    for col in df.iloc[:, 4:]:
        df[col] = df[col].astype(int)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # Dropping unnecessaries columns
    df.drop(columns=['genre', 'child_alone'], inplace=True)

    # Transforming values equals number 2, to number 1
    df.loc[df.related == 2, 'related'] = 1

    return df


def save_data(df, database_filename):
    ''' 
    Description: This function saves the dataframe in a database
    INPUT:
    Dataframe, the name of the database
    OUTPUT:
    No outputs
    '''
    #Saving the dataframe
    db = database_filename
    engine = create_engine(f'sqlite:///{db}.db')
    df.to_sql('DisasterResponse', engine, index=False,
              if_exists='replace')


def main():
    '''
    Description: This function executes all the other functions
    in this script.
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath =
        sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE:
              {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages andcategories '\
'datasets as the first and second argument respectively,as '\
'well as the filepath of the database to save the cleaned data '\
'to as the third argument. \n\nExample: python process_data.py '\
'disaster_messages.csv disaster_categories.csv '\
'DisasterResponse.db')


if __name__ == '__main__':
    main()
