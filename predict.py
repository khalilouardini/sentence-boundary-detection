import torch
import re
from string import punctuation
import pickle
from dataset_context import pad_data, indexer 
import numpy as np

def find_eos_candidates(text): 
    # potential end of sentence candidates
    PUNCT = '[\(\)\u0093\u0094`“”\"›〈⟨〈<‹»«‘’–\'``'']*'
    potential_eos = [m for m in re.finditer(r'([\.:?!;])(\s+' + PUNCT + '|' + PUNCT + '\s+|[\s\n]+)', text)]
    eos_candidates = []
    for eos_position in potential_eos:
        try:
            extract = text[eos_position.start() - 100: eos_position.start() + 100].replace("\n", " ")
            extract = extract.replace(". ", ".")
            match = eos_position.group(0).replace("\n", "")
            before = ''.join([c.lower() for c in extract.split(match)[0] if c not in punctuation])
            before = [c.lower() for c in before.split(" ")[-3:] if c!='']
            after = ''.join([c.lower() for c in extract.split(match)[1] if c not in punctuation])
            after = [c.lower() for c in after.split(" ")[:3] if c!= '']
            context = before + after
            eos_candidates.append((eos_position.start(), context))
        except:
            continue

    return eos_candidates

def predict_eos(model_path, vocab_path, input_file):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load torch model
    model = torch.load(model_path)
    model.eval()

    # Load vocab
    with open(vocab_path, 'rb') as handle:
        word2idx = pickle.load(handle)

    # Load text
    with open(input_file, mode='r', encoding='utf-8') as f:
        text = f.read()
    text = text[50000000:60000000]

    eos_candidates = find_eos_candidates(text[:2000])
    max_len = 6
    output = []
    last = 0
    for eos_candidate in eos_candidates:
        eos_pos = eos_candidate[0]
        x = np.array(pad_data(indexer(eos_candidate[1], word2idx), max_len))
        x = torch.Tensor(x).int().to(device)
        x = torch.unsqueeze(x, dim=0)
        prob = model(x).flatten()
        if prob > 0.5:
            output.append(text[last: eos_pos])
            last = eos_pos
    
    # Final output
    for sentence in output:
        print(sentence + " <eos\>" + "\n")

    print("Done")

if __name__ == '__main__':
    
    input_file = 'data/fr-en/europarl-v7.fr-en.en'
    model_path = 'inference_model.pth'
    vocab_path = 'vocab.pkl'

    predict_eos(model_path, vocab_path, input_file)
