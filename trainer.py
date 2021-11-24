import torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model(model, train_loader, valid_loader, history, epochs, optimizer, criterion, lr_scheduler, clipping, device):
    """
    :param: model: Pytorch model to be trained
    :param: train_loader: train dataloader (torch.utils.data.Dataloader object)
    :param: valid_loader: validation dataloader (torch.utils.data.Dataloader object)
    :param: history: dictionnary where learning curves are stored
    :param: epochs: number of epochs
    :param: optimizer: torch.optim object
    :param: criterion: objective function
    :param: lr_scheduler: learning rate scheduler
    :param: clipping: Whether to use gradient clipping (for RNN)
    :param: CPU or GPU device
    :return: None
    """
    model.train() #Ensure the network is in "train" mode with dropouts active
    for e in range(epochs):
        running_loss, running_accuracy = 0, 0
        for x, labels in train_loader:
            x = x.int().to(device)
            labels = labels.float().to(device)
            # Forward Pass
            optimizer.zero_grad()
            output = model(x).flatten()
            loss = criterion(output, labels)
            # Backward Pass
            loss.backward() 
            if clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            running_loss += loss.item()
            running_accuracy += compute_accuracy(output, labels)
        else:
            validation_loss, validation_acc = eval_model(model, valid_loader, device, criterion)
            
            model.train()

            lr_scheduler.step(validation_loss)

            train_loss = running_loss / len(train_loader)
            train_accuracy = running_accuracy / len(train_loader)
            
            history['train_loss'].append(train_loss), history['train_acc'].append(train_accuracy)
            history['valid_loss'].append(validation_loss), history['valid_acc'].append(validation_acc)

            if (e+1)%1 == 0:
                print("Epoch {}: Training loss: {} | Training accuracy: {}".format(e+1, train_loss, train_accuracy))
                print("Epoch {}: Validation loss: {} | Validation accuracy: {}".format(e+1, validation_loss, validation_acc))
                print()                


def compute_accuracy(probs, labels):
    """
    :param: probs: inferred probabiliy
    :param: labels: groundtruth binary labels
    :return: accuracy metric
    """
    preds = (probs > 0.5).int()
    accuracy = (preds == labels).float()
    accuracy = torch.mean(accuracy)
    return accuracy

@torch.no_grad()
def eval_model(model, dataloader, device, criterion):
    """
    :param: model: Pytorch model to evaluate
    :param: dataloader: torch.utils.data.Dataloader object
    :return: Validation loss and Accuracy
    """
    with torch.no_grad():
        model.eval()
        eval_loss, eval_acc = 0, 0
        for x, labels in dataloader:
            x = x.int().to(device)
            labels = labels.float().to(device)
            output = model(x).flatten()
            loss = criterion(output, labels)
            eval_loss += loss.item()
            eval_acc += compute_accuracy(output, labels)
        
    return eval_loss / len(dataloader), eval_acc / len(dataloader)

@torch.no_grad()
def inference(model, dataloader, device):
    """
    :param: model: Pytorch model to evaluate
    :param: dataloader: torch.utils.data.Dataloader object
    :param: labels: groundtruth labels
    :return: Evaluation metric (accuracy, precision, recall, f1)
    """
    with torch.no_grad():
        test_probs = []
        test_labels = []
        for x, labels in dataloader:
            x = x.int().to(device)
            test_probs.append(model(x).flatten())
            test_labels.append(labels)
        probs = torch.cat(test_probs, axis=0).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        labels = torch.cat(test_labels, axis=0).cpu().numpy()
        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        print("Macro average accuracy:", acc)
        print("Macro average precision:", precision)
        print("Macro average recall:", recall)
        print("Macro average f1:", f1)
        return acc, precision, recall, f1
