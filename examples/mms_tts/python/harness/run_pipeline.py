import sys
import argparse
import numpy as np
import torch
sys.path.insert(0, '/home/ubuntu/rknn_model_zoo/examples/mms_tts/python')
from mms_tts import vocab, MAX_LENGTH, preprocess_input, middle_process, init_model, run_encoder, run_decoder, release_model
import soundfile as sf

TEXT = "Mister quilter is the apostle of the middle classes and we are glad to welcome his gospel."
SR = 16000


def main():
    p = argparse.ArgumentParser(description='Full TTS pipeline, saves WAV, prints per-token durations.')
    p.add_argument('--encoder', required=True, help='encoder .rknn or .onnx')
    p.add_argument('--decoder', required=True, help='decoder .rknn or .onnx')
    p.add_argument('--out-wav', required=True)
    p.add_argument('--target', default='rk3588')
    p.add_argument('--text', default=TEXT)
    args = p.parse_args()

    input_ids, attn_mask = preprocess_input(args.text, vocab, MAX_LENGTH)

    enc = init_model(args.encoder, args.target, None)
    dec = init_model(args.decoder, args.target, None)

    log_dur, pad_mask, prior_means, prior_log_vars = run_encoder(enc, input_ids, attn_mask)
    log_dur = np.asarray(log_dur)
    print('log_duration: shape=%s dtype=%s range=[%.6f, %.6f] uniq(4dp)=%d'
          % (log_dur.shape, log_dur.dtype, log_dur.min(), log_dur.max(),
             len(np.unique(np.round(log_dur.ravel(), 4)))), flush=True)

    attn, out_mask, pred_len = middle_process(log_dur, pad_mask, MAX_LENGTH)
    per_token = np.asarray(torch.as_tensor(attn).sum(dim=(0, 1, 2)).numpy())  # (Tin,)
    nonzero = np.nonzero(per_token)[0]
    print('pred_len=%d  nonzero-duration tokens=%d' % (pred_len, len(nonzero)), flush=True)
    print('per-token durations (nonzero):', flush=True)
    print('  idx = %s' % nonzero.tolist(), flush=True)
    print('  dur = %s' % [int(per_token[i]) for i in nonzero], flush=True)

    waveform = run_decoder(dec, attn, out_mask, prior_means, prior_log_vars)
    np_wav = np.array(waveform[0][:pred_len * 256])
    sf.write(args.out_wav, np_wav, SR)
    print('saved %s (len=%d samples, %.2fs)' % (args.out_wav, len(np_wav), len(np_wav) / SR), flush=True)

    release_model(enc)
    release_model(dec)


if __name__ == '__main__':
    main()