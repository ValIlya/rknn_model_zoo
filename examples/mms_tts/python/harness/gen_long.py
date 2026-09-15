import argparse
import sys
import numpy as np
sys.path.insert(0, '/home/ubuntu/rknn_model_zoo/examples/mms_tts/python')
from mms_tts import vocab, MAX_LENGTH, preprocess_input, middle_process, init_model, run_encoder, run_decoder, release_model
import soundfile as sf

SR = 16000


def chunk_text(text, max_chars=90):
    segs, cur = [], ''
    for ch in text:
        cur += ch
        if len(cur) >= max_chars and ch in '.!?;:,':
            segs.append(cur.strip())
            cur = ''
        elif len(cur) >= max_chars:
            sp = cur.rfind(' ')
            if sp >= max_chars // 2:
                segs.append(cur[:sp].strip())
                cur = cur[sp:]
    if cur.strip():
        segs.append(cur.strip())
    return segs


def main():
    p = argparse.ArgumentParser(description='Chunked full TTS pipeline for LONG text (encoder cap 200 tokens/segment).')
    p.add_argument('--encoder', required=True, help='encoder .rknn or .onnx')
    p.add_argument('--decoder', required=True, help='decoder .rknn or .onnx')
    p.add_argument('--out-wav', required=True)
    p.add_argument('--text-file', required=True)
    p.add_argument('--target', default='rk3588')
    p.add_argument('--max-chars', type=int, default=88)
    args = p.parse_args()

    text = open(args.text_file).read()
    segs = chunk_text(text, args.max_chars)
    print('text chars=%d -> %d segments' % (len(text), len(segs)), flush=True)

    enc = init_model(args.encoder, args.target, None)
    dec = init_model(args.decoder, args.target, None)

    wavs = []
    for i, seg in enumerate(segs):
        input_ids, attn_mask = preprocess_input(seg, vocab, MAX_LENGTH)
        log_dur, pad_mask, prior_means, prior_log_vars = run_encoder(enc, input_ids, attn_mask)
        log_dur = np.asarray(log_dur)
        attn, out_mask, pred_len = middle_process(log_dur, pad_mask, MAX_LENGTH)
        per_token = np.asarray(np.asarray(attn).sum(axis=(0, 1, 2)))
        nz = np.nonzero(per_token)[0]
        wf = run_decoder(dec, attn, out_mask, prior_means, prior_log_vars)
        np_wav = np.array(wf[0][:pred_len * 256])
        wavs.append(np_wav)
        print('  seg %2d: chars=%3d pred_len=%4d  %.2fs  dur_range=[%d, %d]  log_dur=[%.4f, %.4f]'
              % (i, len(seg), pred_len, len(np_wav) / SR,
                 int(per_token[nz].min()) if len(nz) else 0,
                 int(per_token[nz].max()) if len(nz) else 0,
                 float(log_dur[np.nonzero(attn_mask)[0], 0, np.nonzero(attn_mask)[1]].min()) if np.nonzero(attn_mask)[0].size else 0,
                 float(log_dur[np.nonzero(attn_mask)[0], 0, np.nonzero(attn_mask)[1]].max()) if np.nonzero(attn_mask)[0].size else 0),
              flush=True)

    full = np.concatenate(wavs)
    sf.write(args.out_wav, full.astype(np.float32).ravel(), SR)
    print('saved %s (total %d samples, %.2fs)' % (args.out_wav, len(full), len(full) / SR), flush=True)

    release_model(enc)
    release_model(dec)


if __name__ == '__main__':
    main()