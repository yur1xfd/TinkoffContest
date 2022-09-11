import argparse
import re
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import random
from tqdm import tqdm
import sys


class TextPreprocessing:
    def __init__(self, text):
        self.text = text.lower()
        self.text_spl = self.text.split()
        self.new_para = []
        for i in self.text_spl:
            self.new_para.append(re.sub('[^а-яА-ЯёЁ]', " ", i))

        self.text = " ".join(self.new_para)


def remove_dup(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def preprocess(text):
    t1 = TextPreprocessing(text)
    data = t1.text
    word_list = data.split()
    word_list = remove_dup(word_list)
    word_dict = {}
    for i in range(len(word_list)):
        if word_list[i] not in word_dict:
            word_dict[word_list[i]] = i
    return word_dict


def get_dict(path):
    word_dict = {}
    if path is None:
        print('input the text:')
        text = input()
        word_dict = preprocess(text)
        int_text = [word_dict[w] for w in word_dict]
        return word_dict, int_text

    for filename in os.listdir(path):
        with open(path + '/' + filename, 'r', encoding='WINDOWS-1251') as file:
            print(filename+' processing....')
            text = file.read()
        temp = preprocess(text)
        leng = len(word_dict)
        for w in temp:
            temp[w] += leng
        word_dict = word_dict | temp
    int_text = [word_dict[w] for w in word_dict]
    return word_dict, int_text


class TextGenRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, device):
        super(TextGenRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.device = device
        self.embedding = nn.Embedding(self.input_size, self.hidden_size, device=self.device)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True, device=self.device)
        self.linear = nn.Linear(self.hidden_size, self.output_size, device=self.device)

    def forward(self, batch):
        batch = batch.long().to(self.device)
        out = self.embedding(batch)
        out = out.to(self.device)
        out, _ = self.rnn(out)
        out = self.linear(out)

        return out


class ModelText:
    def __init__(self, int_text, word_dict, device):

        self.seq_len = 32  # predicting next word form the previous 32 words
        self.batch_size = 16  # total of 16 , 32 seq in a batch
        self.word2int_dict = word_dict
        self.int2word_dict = {i: j for j, i in word_dict.items()}
        self.int_text = int_text[:-(len(int_text) % (self.seq_len * self.batch_size))]
        self.input_size = int_text[-1]
        self.hidden_size = 256
        self.output_size = self.input_size
        self.n_layers = 1
        self.epochs = 25
        self.print_every = 1
        self.lr = 0.001
        self.device = device
        self.rnn = TextGenRNN(self.input_size,
                              self.hidden_size,
                              self.output_size,
                              self.n_layers,
                              self.device)
        self.optimizer = optim.Adam(self.rnn.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.loss_ = []
        self.iterations = []

    def get_batches(self):

        i = 0
        counter = 0
        no_time = len(self.int_text) // (self.seq_len * self.batch_size)
        while i != int(no_time) - 1:

            x_list = []
            y_seq = []

            for _ in range(self.batch_size):
                if not self.int_text[counter:self.seq_len + counter]:
                    print(
                        "\nError: The dataset is too small and/or the values  seq_len and batch_size are incorrect in"
                        "ModelText",
                        file=sys.stderr)
                    exit()
                x_list.append(self.int_text[counter:self.seq_len + counter])
                y_seq.append(self.int_text[counter + 1:self.seq_len + counter + 1])
                counter += self.seq_len

            i += 1
            yield x_list, y_seq

    def fit(self):
        for epoch in tqdm(range(self.epochs)):
            for label, actual in self.get_batches():
                label = torch.Tensor(label).float().to(self.device)
                actual = torch.Tensor(actual).long().to(self.device)

                y_pred = self.rnn(label)
                y_pred = y_pred.transpose(1, 2)
                loss = self.criterion(y_pred, actual)

                loss.backward()

                with torch.no_grad():
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            self.iterations.append(epoch)
            self.loss_.append(loss)

            if epoch % self.print_every == 0:
                print(f"Loss after {epoch} iteration : {loss}")

    def generate(self, init_str, predict_len):
        if init_str is None:
            init_str = self.int2word_dict[random.randint(0, len(self.word2int_dict) - 1)]
        if predict_len is None:
            predict_len = 20
        generated_list = [init_str]
        for _ in range(predict_len):
            encoded_input = torch.Tensor([self.word2int_dict[w] for w in init_str.lower().split(' ')]).long().unsqueeze(
                0)
            gen_text = self.rnn(encoded_input)
            _, max_ele = torch.topk(gen_text[0], k=1)
            generated_list.append(self.int2word_dict[max_ele[0].item()])

            init_str = self.int2word_dict[max_ele[0].item()]

        return " ".join(generated_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='path to dataset')
    parser.add_argument('--model', type=str, default='model.pkl', help='path to model .pkl')

    args = parser.parse_args()

    if torch.cuda.is_available:
        device = torch.device("cuda:0")
    else:
        device = "cpu"
    word_dict, int_text = get_dict(args.input_dir)
    g1 = ModelText(int_text, word_dict, device)
    g1.fit()
    filename = args.model
    pickle.dump(g1, open(filename, 'wb'))


if __name__ == '__main__':
    main()
