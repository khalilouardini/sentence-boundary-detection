import numpy as np
from sklearn.model_selection import train_test_split
from dataset_char import prepare_data
from models import LSTM, Bi_LSTM
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from trainer import train_model, inference
import argparse
import pickle
from dataset_context import encode_features

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file",
                        help="dataset path", default='data/fr-en/europarl-v7.fr-en.en')
    parser.add_argument("--model_type", default='lstm',
                        help="Which recurrent model ", choices=['lstm', 'bi_lstm'])
    parser.add_argument("--context_encoding", default='word',
                    help="Whether to encode at the word or character level ", choices=['word', 'char'])
    # Parameters
    args = parser.parse_args()
    training_file = args.input_file
    model_type = args.model_type
    context_encoding = args.context_encoding

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #training_file = 'data/SETIMES2.en-tr.tr.sentences.train'
    
    with open(training_file, mode='r', encoding='utf-8') as f:
        training_corpus = f.read()

    if context_encoding == 'char':
        X, y, vocab_size = prepare_data(training_file)
    else:
        X, y, dataset, vocab, word2idx = encode_features(training_corpus[:50000000])

    # Train/Test split
    batch_size = 64
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)
    print(x_train.shape, x_valid.shape, x_test.shape, y_train.shape, y_valid.shape, y_test.shape)

    # Datasets
    train_dataset = TensorDataset(torch.Tensor(x_train),
                       torch.tensor(y_train)
                       )

    valid_dataset = TensorDataset(torch.Tensor(x_valid),
                        torch.tensor(y_valid)
                        )

    test_dataset = TensorDataset(torch.Tensor(x_test),
                    torch.tensor(y_test)
                    )

    train_loader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True)

    valid_loader = DataLoader(dataset=valid_dataset,
                                batch_size=batch_size,
                                shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=True)                       

    # Hyperparameters
    embedding_dim = 32
    lstm_size = 64
    dropout_rate = 0.15
    epochs = 1
    learning_rate = 1e-3
    history = {'train_loss': [], 'valid_loss': [],
        'train_acc': [], 'valid_acc': []
        }

    # intialize model
    if model_type == 'lstm':
        if context_encoding == 'char':
            lstm = LSTM(vocab_size, embedding_dim, lstm_size, dropout_rate)
        else:
            lstm = LSTM(len(vocab), embedding_dim, lstm_size, dropout_rate)
    #bi_lstm = Bi_LSTM(vocab_size, embedding_dim, lstm_size, dropout_rate)

    # Objective function
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    lr_scheduler = ReduceLROnPlateau(optimizer)

    #Train
    train_model(lstm, train_loader, valid_loader, history, epochs, optimizer, criterion, lr_scheduler, True, device)
    print("Training Done")

    # Save model and vocabulary
    torch.save(lstm, 'inference_model.pth')
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(word2idx, f)

    # Evaluation
    acc, precision, recall, f1 = inference(lstm, test_loader, device)
    print("Evaluation Done")



