import pandas as pd

def load_books_data(csv_path):
    books = pd.read_csv(csv_path,encoding='latin-1',low_memory=False)
    books.drop_duplicates(subset='Book-Title',inplace=True)
    books.fillna('',inplace=True)
    books['content'] = (
        books['Book-Title'].astype(str) + ' ' +
        books['Book-Author'].astype(str) + ' ' + 
        books['Publisher'].astype(str))
    return books
