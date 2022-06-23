import random
import torch
from torch.utils.data import Dataset
import argparse
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

"""
The input-output pairs (x, y) of the NameDataset are of the following form:

  x: Where was Khatchig Mouradian born?⁇Lebanon⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Lebanon⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  x: Where was Jacob Henry Studer born?⁇Columbus⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
  y: □□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□⁇Columbus⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

Using the PAD_CHAR characters in y before the ⁇[place] keeps the trainer from
optimizing the model to predict the question, "Where was...".

Note that the NameDataset should take the pretraining_dataset defined in run.py
as an input. This is to allow the vocab specification of the NameDataset to be
the same as that of the pretraining dataset.

You don't need to implement anything in NameDataset.
"""

class NameDataset(Dataset):
    def __init__(self, pretraining_dataset, data):
        self.MASK_CHAR = u"\u2047" # the doublequestionmark character, for mask
        self.PAD_CHAR = u"\u25A1" # the empty square character, for pad
        self.itos = pretraining_dataset.itos 
        self.stoi = pretraining_dataset.stoi 
        self.block_size = pretraining_dataset.block_size
        self.data = list(data.encode('utf-8').decode('ascii', errors='ignore').split('\n'))

    def __len__(self):
        # returns the length of the dataset
        return len(self.data) - 1

    def __getitem__(self, idx):
        inp, oup = self.data[idx].split('\t')
        x = inp + self.MASK_CHAR + oup + self.MASK_CHAR
        x = x + self.PAD_CHAR*(self.block_size - len(x))
        y = self.PAD_CHAR*(len(inp)-1) + x[len(inp):]
        
        x = x[:-1]
        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)
        return x, y

class CharCorruptionDataset(Dataset):
    def __init__(self, data, block_size):
        self.MASK_CHAR = u"\u2047" # the doublequestionmark character, for mask
        self.PAD_CHAR = u"\u25A1" # the empty square character, for pad

        chars = list(sorted(list(set(data))))
        assert self.MASK_CHAR not in chars 
        assert self.PAD_CHAR not in chars
        chars.insert(0, self.MASK_CHAR)
        chars.insert(0, self.PAD_CHAR)

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }

        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data.split('\n')

    def __len__(self):
        # returns the length of the dataset
        return len(self.data)-1

    def __getitem__(self, idx):
        # TODO [part e]: see spec above
        document = self.data[idx]

        # if len(document) <= 0:
        #     raise IndexError
        truncate_length = random.randint(4, int(self.block_size*7/8))

        truncate_length = min(len(document), truncate_length)
        truncate_doc = document[:truncate_length]

        masked_length = random.randint(truncate_length//8, truncate_length*3//8)
        prefix_length = random.randint(1, truncate_length-masked_length-1)

        prefix = truncate_doc[:prefix_length]
        masked_content = truncate_doc[prefix_length:prefix_length+masked_length]
        suffix = truncate_doc[prefix_length+masked_length:]

        # [prefix] MASK_CHAR [suffix] MASK_CHAR [masked_content] [pads]
        masked_string = prefix + self.MASK_CHAR + suffix + self.MASK_CHAR + masked_content
        masked_string += self.PAD_CHAR*max(self.block_size-truncate_length-2, 0)

        dix = [self.stoi[s] for s in masked_string]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)

        return x, y





"""
Code under here is strictly for your debugging purposes; feel free to modify
as desired.
"""
if __name__ == '__main__':
    # argp = argparse.ArgumentParser()
    # argp.add_argument('dataset_type', help="Type of dataset to sample from."
    #         "Options: namedata, charcorruption.",
    #         choices=["namedata", "charcorruption"])
    # args = argp.parse_args()

    # if args.dataset_type == 'namedata':
    #     # Even if it hasn't been implemented, we use it to define the vocab
    #     corruption_dataset = CharCorruptionDataset(open('wiki.txt', encoding='utf-8').read(), 128)
    #     # Make the name dataset
    #     name_dataset = NameDataset(corruption_dataset,
    #         open('birth_places_train.tsv').read())
    #     for _, example in zip(range(4), name_dataset):
    #         x, y = example
    #         print('x:', ''.join([name_dataset.itos[int(c)] for c in x]))
    #         print('y:', ''.join([name_dataset.itos[int(c)] for c in y]))
    #     pass
    # elif args.dataset_type == 'charcorruption':
    block_size = 128
    text = open('../wiki.txt', encoding='utf-8').read()
    pretrain_dataset = CharCorruptionDataset(text, block_size)
    loader = DataLoader(pretrain_dataset, batch_size=128, num_workers=0)
    print(pretrain_dataset.__len__())
    for i, k in enumerate(loader):
        print(i)
    # pbar = tqdm(enumerate(loader), total=len(loader))
    # for it, (x, y) in pbar:
    #     print(x, y)
    # else:
    #     raise ValueError("Unknown dataset type in command line args: {}"
    #             .format(args.dataset_type))

