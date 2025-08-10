import json
import pickle

# This script transforms a JSON file into a Pickle file.
# The JSON file is expected to be in the same directory as this script.
def transform_json_to_pickle(json_file_path, pickle_file_path):
    # Define the paths for the JSON and Pickle files
    # json_file_path = 'news_ID-small.json'
    # pickle_file_path = 'ID_news-small.pkl'

    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

    with open(pickle_file_path, 'wb') as pickle_file:
        pickle.dump(data, pickle_file)

    print(f"JSON data transformed to Pickle saved at {pickle_file_path}")


