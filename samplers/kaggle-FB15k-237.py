# sample fb15k-237 datasets

from collections import defaultdict
from random import choice
import pandas as pd
import tqdm

train = "../FB15k-237/train.txt"
valid = "../FB15k-237/valid.txt"
test = "../FB15k-237/test.txt"



def read_file(file, mask=''):
    records = defaultdict(list)
    answer = defaultdict(list)

    def masking(h, r, t):
        if len(mask) == 0:
            records['head'].append(h)
            records['relation'].append(r)
            records['tail'].append(t)
        else:
            mask_type = choice(mask)
            if 'h' == mask_type:
                records['head'].append('[mask]')
                answer['ans'].append(h)
                records['relation'].append(r)
                records['tail'].append(t)
            if 'r' == mask_type:
                records['head'].append(h)
                records['relation'].append('[mask]')
                answer['ans'].append(r)
                records['tail'].append(t)
            if 't' == mask_type:
                records['head'].append(h)
                records['relation'].append(r)
                records['tail'].append('[mask]')
                answer['ans'].append(t)

    with open(file, 'rt') as f:
        for l in tqdm.tqdm(f.readlines()):
            h, r, t = l.split()
            for _r in r.split('.'):
                masking(h, _r, t)

    return records, answer

if __name__ == "__main__":
    data, ans = read_file(train, '')
    df = pd.DataFrame(data)
    df.to_csv('kaggle/train.csv', index=False)

    data, ans = read_file(valid, 'th')
    df = pd.DataFrame(data)
    df.to_csv('kaggle/public.csv', index=False)

    df = pd.DataFrame(ans)
    df.to_csv('kaggle/public_ans.csv', index=False)

    data, ans = read_file(test, 'th')
    df = pd.DataFrame(data)
    df.to_csv('kaggle/private.csv', index=False)

    df = pd.DataFrame(ans)
    df.to_csv('kaggle/private_ans.csv', index=False)
