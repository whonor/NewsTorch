import json
import pickle

# This script transforms a JSON file into a Pickle file.
# The JSON file is expected to be in the same directory as this script.
# This code will generate a Pickle file with the reversed mapping: ID_news-%s.pkl for LKPNR model.
def transform_json_to_pickle(json_file_path, pickle_file_path):
    # Define the paths for the JSON and Pickle files
    # json_file_path = 'news_ID-small.json'
    # pickle_file_path = 'ID_news-small.pkl'

    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    # Reverse the ID and news_id mapping
    # Original: {news_id: ID} -> Reversed: {ID: news_id}
    reversed_data = {int(value): key for key, value in data.items()}

    with open(pickle_file_path, 'wb') as pickle_file:
        pickle.dump(reversed_data, pickle_file)

    print(f"JSON data transformed to Pickle with reversed mapping saved at {pickle_file_path}")

if __name__ == '__main__':
    transform_json_to_pickle(json_file_path='../../cache/news_ID-large.json',
                             pickle_file_path='../../KGraph_LKPNR/ID_news-large.pkl')



