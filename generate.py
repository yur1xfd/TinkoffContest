import argparse
import pickle
from train import ModelText
from train import TextGenRNN


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='model.pkl', help="path to models")
    parser.add_argument("--length", help="length of generated text")
    parser.add_argument("--prefix", nargs='?', help="the beginning of the text")
    args = parser.parse_args()
    model = pickle.load(open(args.model, 'rb'))
    lenn = None
    if args.length is not None:
        lenn = int(args.length)
    print(model.generate(init_str=args.prefix, predict_len=lenn))


if __name__ == '__main__':
    main()
