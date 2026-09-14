import numpy as np
import sys
sys.path.insert(0, '.')
from mms_tts import preprocess_input, vocab, MAX_LENGTH

input_ids, attention_mask = preprocess_input(
    'Mister quilter is the apostle of the middle classes and we are glad to welcome his gospel.',
    vocab, MAX_LENGTH)
print('input_ids', input_ids.shape, input_ids.dtype)
print('attention_mask', attention_mask.shape, attention_mask.dtype)
len_ = input_ids.astype(np.int64).sum(axis=1)[0]
print('seq len (non-zero tokens * 2):', len_)
np.save('/tmp/input_ids.npy', input_ids.astype(np.int64))
np.save('/tmp/attention_mask.npy', attention_mask.astype(np.int64))
print('saved to /tmp/input_ids.npy and /tmp/attention_mask.npy')