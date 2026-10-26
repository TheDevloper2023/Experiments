"""
I am lazy to make a whole ass CFG system so everything will be just this.
"""

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
BATCH_SIZE = 64
MAX_LEN = 256
EMOTIONS = [
    'admiration','amusement','anger','annoyance','approval','caring','confusion',
    'curiosity','desire','disappointment','disapproval','disgust','embarrassment',
    'excitement','fear','gratitude','grief','joy','love','nervousness','optimism',
    'pride','realization','relief','remorse','sadness','surprise','neutral'
]

NUM_EMOTIONS = len(EMOTIONS)



# Model conf
input_dim = 384
hdn_dims = input_dim // 2 # todo

# Train conf
epochs = 100
lr = 1e-4