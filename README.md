# Sentence Boundary Detection

This code contains a deep learning method for sentence boundary detection. The problem is framed as a supervised binary classification problem.
We go briefly through the main steps of the project, from the data collection and the creation of the dataset, to the models and the inference on unseen data.

## Setup
Create an environment and install depedencies with:
`pip install -r requirements.txt`

## Data

**Instruction**  unzip the file `fr-en.zip` in the .data folder.

For the dataset we use the "European Parliament Proceedings Parallel Corpus" 1996-2011 (https://www.statmt.org/europarl/). This corpus was initially designed to benchmark machine translation methods, but has also been used for sentence boundary tasks. We only use the english corpus. The dataset is segmented in sentences (one sentence per line - it contains linebreaks "\n"). To create a dataset we first use a regex to detect all positive samples (i.e a point followed by a linebreak), and all negative samples (a point or other relevant punctuation not followed by a linebreak). For both cases, we want to represent the context of the end of a sentence (EOS). To do so, we will extract 3 words before the EOS, and 3 words after the EOS. This sequence of 6 tokens will be a single datapoint in our dataset.

**An example would be**
***sentence*** = "it should start by doing this within the Union. \n It should decide, as an emergency measure" --> ***context**** = [within, the, Union, It, should, decide] --> **BOW features** = =[2, 128, 78, 8, 9, 3]

This idea of representing the context of the end of a sentence (EOS) was insipired by the paper [1].
Other papers such as [2] encode the context of a EOS at the character level, taking a few characters before and after as the EOS context representation.

To make this problem feasible in a short time on a CPU, we select the 50 millions first characters of the Europarl english corpus and extract all EOS candidates which gives us more than 350,000 datapoints total. We split this in train, test (20%) and validation (20%) sets. 

The preprocessing is minimal, we simply remove the digits and lowercase before tokenizing the input text. After this we create a vocabulary and use a Bag of Words (BOW) vectorization. If for some reason a sequence is smaller than 6, we pad the sequence with a 'PAD_' token.

All the code related to the creation and pre-processing of the dataset is in the script `dataset_context.py`

## Models
We use recurrent neural networks, specifically LSTMs. Since the size of all the input sequences is rather small (6), a 1-layer LSTM network can be trained fairly quickly on CPU, even though the number of samples is large. We also experiment with a bidirectional LSTM but it did not improve the performance. Before the LSTM, an embedding layer maps the BOW representations to a continuous vector space. 
The models are implemented in `torch` in the script `models.py`

## Training and evaluation
run `python3 main.py --model_type 'lstm' --context_encoding 'word'`

the `main.py` script contains the code to create the dataset, train and evaluate an LSTM model on a subset of the Europarl english corpus. We can also use a dataset, with context encoded at the character level by changing the --context_embedding argument to 'char'. 

The training and evaluation code in pytorch is the `trainer.py` script. We train our LSTM model for 10 epochs with a learning rate of 0.001 and a batch size of 64. For evaluation, since the classification problem is unbalanced, we report precision, recall and f1 score in addition to the accuracy metric. Overall the performance metrics on the test set are:

* accuracy = 0.76
* precision = 0.92
* recall = 0.80
* f1 score = 0.85

## Inference on unseen data
run `python3 predict.py`

The `predict.py` contains the code to run inference with the trained LSTM model on unseen text data. For the example we use a small subset of the Europarl that was not seen during the training. For inference, we first extract a list of potential EOS candidates (We use a regex to extract relevant punctuation, without detecting the linebreaks this time). We use our train vocabulary and encode these candidates into BOW features that are then passed to the trained LSTM to run inference. A trained model is available at `inference_model.pth`. If the predicted probability is greater than 0.5, we predict an EOS. 


## References
[1] SATZ - An Adaptive Sentence Segmentation System, David D. Palmer, 1995

[2] Deep-EOS: General-Purpose Neural Networks for Sentence Boundary Detection, Stefan Schweter, Sajawel Ahmed

[3] Adaptive Multilingual Sentence Boundary Disambiguation, David D. Palmer, Marti A. Hearst

[4] European Parliament Proceedings Parallel Corpus, 1996-2011