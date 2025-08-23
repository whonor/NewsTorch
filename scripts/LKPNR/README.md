# Steps:
Prepare the data for LKPNR model after downloading the 'KGraph_LKPNR'.

1. Generate the merged dataset.
- generate_merged_file.py

2. Build the LLM-based word embedding.
- get_item.py

3. Save the adjacent entity dictionary, leaving only adjacent entities with emb vectors
- count_link_count.py

4. Saving entities appearing in news&embedding entities in merged datasets.
- get_node_emb.py
- get_all_entities_emb_dict.py

# TransformJSON2pkl.py

This script transforms a JSON file into a Pickle file, with a key modification from the original version:

## Changes Made

1. **Reversed Mapping**: The original JSON mapping was `{news_id: ID}`, but we've reversed it to `{ID: news_id}` to match the expected format for downstream processing.

2. **Type Conversion**: The ID values are explicitly converted to integers to ensure consistent data types.

## Usage

The script reads a JSON file where each entry maps a news identifier to an integer ID, and outputs a pickle file where each entry maps an integer ID to a news identifier.

Example:
- Input JSON: `{"N12345": 1, "N67890": 2}`
- Output Pickle: `{1: "N12345", 2: "N67890"}`

## Running the Script

```bash
python TransformJSON2pkl.py
```

The script will transform `../../cache/news_ID-large.json` to `../../KGraph_LKPNR/ID_news-large.pkl` with the reversed mapping.