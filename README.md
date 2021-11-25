# Sentence Boundary Detection
This code contains a deep learning method for sentence boundary detection. The problem is framed as a supervised binary classification problem.
The idea is to create a dataset from raw text. Positive and negative samples are extracted from the text data.

# Setup
Install depedencies
`pip install -r requirements.txt`

# Data

# Training and evaluation
run `python3 main.py --model_type 'lstm' --context_encoding 'word'`

# Inference on unseen data
run `python3 predict.py`

