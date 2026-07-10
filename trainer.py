#!/usr/bin/env python3
"""trainer.py — TOUCH-FLIP-OUT PRO calibrator (BTC only).

  python3 trainer.py --months 2026-05
  python3 trainer.py --months 2026-04,2026-05     # more months = better verification

Strategy needs no heavy models. This script:
 1. downloads BTC (+ETH,SOL for the side-picker) 1s klines from Binance Vision
 2. trains the light no-leak side-picker (predicts first-15s direction from pre-window data)
 3. backtests TOUCH-FLIP-OUT PRO on the frozen last 25%: hysteresis sweep, cost sweep,
    full class decomposition (holds/flips/scratches + win rates)
 4. saves tfo_config.pkl -> live.py

Rules it calibrates: enter at window open; price closes BAND beyond open against you ->
flip once; crosses back through the band -> scratch out; else hold to resolution.

pip install numpy pandas scikit-learn lightgbm requests
"""
import argparse, calendar, os, pickle, sys, zipfile
import numpy as np, pandas as pd, requests

DATA = './kdata1s'
TICKERS = ['btc', 'eth', 'sol']

def dl(sym, month):
    dest = f'{DATA}/{sym}-1s-{month}.zip'
    if os.path.exists(dest) and os.path.getsize(dest) > 5_000_000: return dest
    url = f'https://data.binance.vision/data/spot/monthly/klines/{sym}/1s/{sym}-1s-{month}.zip'
    print('download', url, flush=True)
    r = requests.get(url, stream=True, timeout=600)
    if r.status_code != 200: sys.exit(f'FAILED {r.status_code}: {url}')
    with open(dest, 'wb') as f:
        for ch in r.iter_content(1 << 20): f.write(ch)
    return dest

def load(months):
    start = int(pd.Timestamp(f'{months[0]}-01', tz='UTC').timestamp())
    y, m = map(int, months[-1].split('-'))
    N = int(pd.Timestamp(f'{y}-{m:02d}-{calendar.monthrange(y,m)[1]} 23:59:59', tz='UTC').timestamp()) + 1 - start
    A = {}
    for t in TICKERS:
        S = t.upper() + 'USDT'
        a = {k: np.full(N, np.nan, np.float32) for k in ('open', 'close')}
        a.update({k: np.zeros(N, np.float32) for k in ('vol', 'buyvol')})
        for month in months:
            z = dl(S, month)
            with zipfile.ZipFile(z) as zf, zf.open(zf.namelist()[0]) as f:
                for ch in pd.read_csv(f, header=None, usecols=[0, 1, 4, 5, 9],
                                      names=['ot', 'open', 'close', 'vol', 'tb'], chunksize=3_000_000):
                    if not str(ch['ot'].iloc[0]).lstrip('-').isdigit(): ch = ch.iloc[1:]
                    ch = ch.astype({'ot': np.int64, 'open': float, 'close': float, 'vol': float, 'tb': float})
                    ts = ch['ot'].values
                    ts = ts // 1_000_000 if ts[0] > 10**14 else ts // 1000
                    sec = ts - start
                    ok = (sec >= 0) & (sec < N)
                    a['open'][sec[ok]] = ch['open'].values[ok]
                    a['close'][sec[ok]] = ch['close'].values[ok]
                    a['vol'][sec[ok]] = ch['vol'].values[ok]
                    a['buyvol'][sec[ok]] = ch['tb'].values[ok]
        c = a['close']
        for i in range(1, N):
            if not np.isfinite(c[i]): c[i] = c[i - 1]
        A[t] = a
        print(S, 'loaded', flush=True)
    return A, N

def feats_pre(A, t0):
    """No-leak features strictly before the window opens."""
    c = A['btc']['close']; a = A['btc']
    if t0 < 3600: return None
    op = c[t0]
    if not np.isfinite(op) or op <= 0: return None
    def rr(sb):
        pp = c[t0 - sb]
        return op / pp - 1.0 if np.isfinite(pp) and pp > 0 else 0.0
    v1 = a['vol'][t0-15:t0].sum(); b1 = a['buyvol'][t0-15:t0].sum()
    v4 = a['vol'][t0-60:t0].sum(); b4 = a['buyvol'][t0-60:t0].sum()
    f = dict(r15=rr(15), r30=rr(30), r60=rr(60), r5m=rr(300), r15m=rr(900),
             fl15=(2*b1-v1)/v1 if v1 > 0 else 0.0,
             fl60=(2*b4-v4)/v4 if v4 > 0 else 0.0,
             hour=(t0//3600) % 24)
    for s in ('eth', 'sol'):
        cs = A[s]['close']
        f[f'{s}_r15'] = cs[t0]/cs[t0-15]-1.0 if np.isfinite(cs[t0-15]) and cs[t0-15] > 0 else 0.0
    return f

def sim_tfo(cb, t0, up, band_bp, cost):
    """One window of touch-flip-out. Returns (class, pnl_per_share)."""
    op = cb[t0]
    if not np.isfinite(op): return None
    d = op * band_bp * 1e-4
    state = 1 if up else -1
    flips = 0
    for j in range(4, 300, 5):
        x = cb[t0 + j]
        if not np.isfinite(x): continue
        if state == 1 and x <= op - d:
            if flips == 0: state = -1; flips = 1
            else: return ('scratch', -cost * 2)
        elif state == -1 and x >= op + d:
            if flips == 0: state = 1; flips = 1
            else: return ('scratch', -cost * 2)
    cl = cb[t0 + 299]
    if not np.isfinite(cl) or cl == op: return None
    won = (cl > op) == (state == 1)
    pay = (0.49 - flips * cost) if won else (-0.51 - flips * cost)
    return (('hold' if flips == 0 else 'flip') + ('_w' if won else '_l'), pay)

def main(months):
    import lightgbm as lgb
    os.makedirs(DATA, exist_ok=True)
    A, N = load(months)
    cb = A['btc']['close']
    # windows + side-picker
    t0s, rows, y15 = [], [], []
    for w in range(12, N // 300):
        t0 = w * 300
        f = feats_pre(A, t0)
        if f is None: continue
        r = cb[t0+14] / cb[t0] - 1.0 if np.isfinite(cb[t0+14]) else 0.0
        t0s.append(t0); rows.append(f); y15.append(1 if r > 0 else 0)
    df = pd.DataFrame(rows); y15 = np.array(y15); n = len(df)
    k1, k3 = int(n * 0.60), int(n * 0.75)
    m = lgb.train(dict(objective='binary', verbosity=-1, learning_rate=0.04,
                       num_leaves=31, min_data_in_leaf=80, lambda_l2=5.0, seed=0),
                  lgb.Dataset(df.values[:k1], y15[:k1]), 1000,
                  valid_sets=[lgb.Dataset(df.values[k1:k3], y15[k1:k3])],
                  callbacks=[lgb.early_stopping(80, verbose=False)])
    p15 = m.predict(df.values)
    print(f'side-picker (no-leak 15s dir): test acc {((p15[k3:]>0.5).astype(int)==y15[k3:]).mean():.4f}')
    # calibrate band on dev [k1:k3], report frozen test [k3:]
    print('\nband sweep (dev):')
    best = None
    for bp in (0, 2, 3, 4, 6, 10):
        pnl = [r[1] for i in range(k1, k3)
               if (r := sim_tfo(cb, t0s[i], p15[i] > 0.5, bp, 0.02)) is not None]
        avg = np.mean(pnl)
        print(f'  band {bp:2d}bp: pnl/window {avg:+.4f}/share (n={len(pnl)})')
        if best is None or avg > best[0]: best = (avg, bp)
    _, BAND = best
    print(f'\nselected band: {BAND}bp | FROZEN TEST (last 25%):')
    from collections import Counter
    for cost in (0.02, 0.04, 0.06):
        res = [r for i in range(k3, n)
               if (r := sim_tfo(cb, t0s[i], p15[i] > 0.5, BAND, cost)) is not None]
        cls = Counter(k for k, _ in res)
        avg = np.mean([p for _, p in res])
        hw, hl = cls.get('hold_w', 0), cls.get('hold_l', 0)
        fw, fl = cls.get('flip_w', 0), cls.get('flip_l', 0)
        print(f'  cost {cost:.2f}: pnl/window {avg:+.4f}/share | holds {hw}W/{hl}L '
              f'({hw/max(hw+hl,1):.3f}) flips {fw}W/{fl}L ({fw/max(fw+fl,1):.3f}) '
              f'scratches {cls.get("scratch",0)}')
    pickle.dump(dict(side_model=m, feats=list(df.columns), band_bp=int(BAND)),
                open('tfo_config.pkl', 'wb'))
    print('\nsaved tfo_config.pkl -> run: python3 live.py')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--months', default='2026-05')
    main([x.strip() for x in ap.parse_args().months.split(',')])
