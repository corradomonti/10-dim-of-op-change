import pandas as pd
from pyspark import SparkContext, SparkConf, SQLContext
from tqdm import tqdm

from glob import glob
import json
from operator import itemgetter
from pathlib import Path

NEGATIVE_SUBS = 'NEG'

labelled_sub_df = pd.read_csv("../data/top-2000-sfw-subreddits-labelled.csv")
labelled_sub_df.subreddit = labelled_sub_df.subreddit.map(lambda s: s.lower()[len("/r/"):])

top_2000_subs = set(labelled_sub_df.subreddit.values)
sociopol_subs = set(labelled_sub_df[labelled_sub_df.sociopolitical == 'X'].subreddit.values)
negative_subs = top_2000_subs - sociopol_subs - {'changemyview'}

# Reddit collection from https://pushshift.io/
PATHS = glob("/data/big/reddit/submissions/*/*.bz2")

t2path = {f.split("/RS_")[1][:-4]: f for f in sorted(PATHS)}

K_train = 50 # S.t. K * 120 * len(sociopol_subs) is not too large
K_test = 50

def resilient_json(s):
    try:
        return json.loads(s)
    except: # pylint: disable=W0702
        return {}

def main():
    Path("../data/sociopol/train/").mkdir(parents=True, exist_ok=True)
    Path("../data/sociopol/test/").mkdir(parents=True, exist_ok=True)
    
    sc = SparkContext()
    for t, monthly_posts_path in t2path.items():
        print(f"Building data set for {t}...")
        posts_rdd = sc.textFile(monthly_posts_path).map(resilient_json)
        self_posts = posts_rdd.filter(lambda p: p.get('is_self', False))
        sub2text = self_posts.map(lambda p: (p.get('subreddit', '').lower(), p.get('selftext', '')))
        pos_sub2text = sub2text.filter(lambda x: x[0] in sociopol_subs)
        neg_sub2text = sub2text.filter(lambda x: x[0] in negative_subs)
        pos_sub2text.persist()
        
        for sub in tqdm(list(sociopol_subs) + [NEGATIVE_SUBS]):
            if sub == NEGATIVE_SUBS:
                rdd = neg_sub2text.map(itemgetter(1))
            else:
                rdd = pos_sub2text.filter(lambda x: x[0] == sub).map(itemgetter(1)) # pylint: disable=W0640
            
            sample_size = (K_train + K_test)
            if sub == NEGATIVE_SUBS:
                sample_size *= len(sociopol_subs)
            
            sample = rdd.filter(lambda t: len(t) >= 800).takeSample(
                withReplacement=False, num=sample_size, seed=123)

            if len(sample) == (K_train + K_test):
                num_train = K_train
            else:
                num_train = int(len(sample) * K_train / (K_train + K_test))

            if num_train > 0:
                sample = [x.replace('\n', ' ').replace('\r', ' ') for x in sample]
                train, test = sample[:num_train], sample[num_train:]
                
                filename = f"{t}-{'negative' if sub == NEGATIVE_SUBS else 'positive'}.txt"
                for dirname, dataset in [('train', train), ('test', test)]:
                    with open(f"../data/sociopol/{dirname}/{filename}", 'a') as f:
                        if sub == NEGATIVE_SUBS:
                            for line in dataset:
                                f.write(line + '\n')
                        else:
                            for line in dataset:
                                f.write(sub + '\t' + line + '\n')
                            
        
        pos_sub2text.unpersist()
                        
if __name__ == '__main__':
    main()
