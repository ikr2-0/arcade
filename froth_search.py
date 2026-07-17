#!/usr/bin/env python3
"""
FROTH-FADE EXHAUSTIVE RULE SEARCH — SELF-CONTAINED PIPELINE
============================================================
Downloads Binance Vision monthly aggTrades for the configured tickers,
aggregates to 1m bars, builds a leak-free feature set, runs an
exhaustive complex-rule search (AND / OR / AND-NOT / cross-ticker
breadth counts / train-calibrated intensity thresholds) with a
per-rule CatBoost model, and prints full tier analysis per ticker.

HONEST PROTOCOL — ENFORCED BY THE SPLIT, DO NOT CIRCUMVENT:
  TRAIN  = all months except the last two
  VAL    = second-to-last month  (rule selection happens ONLY here)
  TEST   = last month            (single readout; printed at the end)
The TEST numbers are the system's true performance. If you rerun with
different rules/settings after seeing TEST, the TEST month is burned
and you must wait for a new month to measure anything.

Requirements: pip install pandas numpy catboost requests
Disk: ~300MB per ticker-month (zip). Runtime: download ~5-10 min/file,
processing ~2-4 min/file first run (cached after), search ~30-90 min.

Usage:
  python froth_search.py                 # last 6 full months, all tickers
  python froth_search.py --months 8      # more history
"""
import os, sys, io, zipfile, itertools, argparse, warnings, datetime as dt
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
try:
    import requests
except ImportError:
    sys.exit("pip install requests")
try:
    from catboost import CatBoostClassifier
except ImportError:
    sys.exit("pip install catboost")

import time as _time
_T0 = _time.time()
def log(msg):
    el = _time.time() - _T0
    print(f"[{el/60:6.1f}m] {msg}", flush=True)

# ----------------------------- CONFIG -----------------------------
TICKERS  = ['BTCUSDT','ETHUSDT','SOLUSDT','DOGEUSDT','BNBUSDT','XRPUSDT']
TKEY     = {'BTCUSDT':'B','ETHUSDT':'E','SOLUSDT':'S','DOGEUSDT':'G','BNBUSDT':'N','XRPUSDT':'X'}
HORIZON  = 5            # minutes ahead (target = close[t+H] vs close[t])
TOPK     = 3            # rules reported per target
MAX_RULES= 250          # cap on rule pool (exhaustive within cap)
ITER     = 120          # catboost iterations
DATA_DIR = 'binance_data'
BARS_DIR = 'bars_cache'
BASE_URL = 'https://data.binance.vision/data/spot/monthly/aggTrades'

# ------------------------- DOWNLOAD & BARS -------------------------
def month_list(n_months):
    today = dt.date.today().replace(day=1)
    out=[]
    for i in range(n_months, 0, -1):
        m = today - pd.DateOffset(months=i)
        out.append((m.year, m.month))
    return out

def download(sym, y, m):
    os.makedirs(DATA_DIR, exist_ok=True)
    fn = f"{sym}-aggTrades-{y}-{m:02d}.zip"
    path = os.path.join(DATA_DIR, fn)
    if os.path.exists(path) and os.path.getsize(path) > 1e6:
        return path
    url = f"{BASE_URL}/{sym}/{fn}"
    log(f"DOWNLOAD start {fn}")
    r = requests.get(url, stream=True, timeout=120)
    if r.status_code != 200:
        log(f"!! {fn} -> HTTP {r.status_code} (month not published yet, skipping)"); return None
    total = int(r.headers.get('content-length', 0)) / 1e6
    done = 0
    with open(path,'wb') as f:
        for chunk in r.iter_content(1<<22):
            f.write(chunk); done += len(chunk)/1e6
            if int(done) % 50 < 4: log(f"  {fn}: {done:.0f}/{total:.0f} MB")
    log(f"DOWNLOAD done {fn} ({done:.0f} MB)")
    return path

def zip_to_bars(path):
    """Stream aggTrades zip -> 1m bars. Columns: aggId,price,qty,firstId,lastId,ts,isBuyerMaker,isBestMatch.
    Timestamps may be ms (13 digit) or us (16 digit)."""
    os.makedirs(BARS_DIR, exist_ok=True)
    out = os.path.join(BARS_DIR, os.path.basename(path).replace('.zip','_1m.parquet'))
    if os.path.exists(out):
        log(f"CACHED {os.path.basename(out)}"); return out
    log(f"PROCESS start {os.path.basename(path)}")
    parts=[]; nch=0
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            for chunk in pd.read_csv(f, header=None, chunksize=5_000_000,
                                     names=['aid','price','qty','fid','lid','ts','ibm','ibx'],
                                     usecols=['price','qty','ts','ibm']):
                ts = chunk['ts'].values
                unit = 1_000_000 if ts[0] > 1e15 else 1_000
                minute = (ts // (60*unit)).astype(np.int64)
                chunk = chunk.assign(minute=minute,
                                     tb=np.where(chunk['ibm']==False, chunk['qty'], 0.0))
                g = chunk.groupby('minute', sort=True)
                parts.append(pd.DataFrame({
                    'open':g['price'].first(),'high':g['price'].max(),
                    'low':g['price'].min(),'close':g['price'].last(),
                    'vol':g['qty'].sum(),'tb':g['tb'].sum(),'n':g['qty'].size()}))
                nch += 1
                log(f"  chunk {nch} ({nch*5}M trades) -> {sum(len(p) for p in parts)} partial minute-rows")
    b = pd.concat(parts)
    # merge minutes split across chunk boundaries
    g = b.groupby(level=0)
    bars = pd.DataFrame({'open':g['open'].first(),'high':g['high'].max(),'low':g['low'].min(),
                         'close':g['close'].last(),'vol':g['vol'].sum(),'tb':g['tb'].sum(),'n':g['n'].sum()})
    bars.index.name='minute'
    bars.reset_index().to_parquet(out)
    log(f"PROCESS done {os.path.basename(out)}: {len(bars)} 1m bars")
    return out

# --------------------------- ASSEMBLE ---------------------------
def build(n_months):
    months = month_list(n_months)
    per_tk = {}
    for sym in TICKERS:
        frames=[]
        for y,m in months:
            p = download(sym,y,m)
            if p is None: continue
            frames.append(pd.read_parquet(zip_to_bars(p)))
        df = pd.concat(frames, ignore_index=True)
        per_tk[TKEY[sym]] = df
        log(f"ASSEMBLED {sym}: {len(df)} 1m bars")
    # align on common minute index (inner join)
    common = None
    for k,df in per_tk.items():
        s = set(df['minute'])
        common = s if common is None else (common & s)
    common = np.array(sorted(common))
    P={}
    for k,df in per_tk.items():
        df = df.set_index('minute').reindex(common)
        P[k] = {c: df[c].values for c in ['open','high','low','close','vol','tb']}
    return P, common, months

# --------------------- FEATURES (LEAK-FREE) ---------------------
# Every feature at row t uses ONLY bars <= t (np.roll k>=0, rolling on closed bars).
# Target y[t] = sign(close[t+H] - close[t]) — window starts at decision price.
def features(P, mins):
    lag = lambda x,k: np.roll(x,k)
    Fd={}
    for k,v in P.items():
        O,H,L,C,V,TB = v['open'],v['high'],v['low'],v['close'],v['vol'],v['tb']
        for j in range(3):
            Fd[f'{k}_body{j}']=np.roll((C-O)/O*1e4,j)
            Fd[f'{k}_cpos{j}']=np.roll((C-L)/np.maximum(H-L,1e-9),j)
        for w in [5,15,60,240]:
            Fd[f'{k}_mom{w}']=(C/np.roll(C,w)-1)*1e4
        Fd[f'{k}_rv15']=pd.Series((C-O)/O*1e4).rolling(15).std().values
        Fd[f'{k}_tbr3']=pd.Series(TB/np.maximum(V,1e-9)).rolling(3).mean().values
        Fd[f'{k}_volz']=((V-pd.Series(V).rolling(240).mean())/pd.Series(V).rolling(240).std()).values
    for a,b in itertools.combinations(P.keys(),2):
        Fd[f'rel5_{a}{b}']=Fd[f'{a}_mom5']-Fd[f'{b}_mom5']
    Fd['hour']=(mins//60)%24
    return pd.DataFrame(Fd)

# ------------------------ RULE GRAMMAR ------------------------
def rule_pool(P, train_mask):
    lag = lambda x,k: np.roll(x,k)
    n = len(train_mask)
    up={k:(P[k]['close']>lag(P[k]['close'],1)) for k in P}
    bk={k:(P[k]['high']>pd.Series(P[k]['high']).shift(1).rolling(5).max().values) for k in P}
    bk10={k:(P[k]['high']>pd.Series(P[k]['high']).shift(1).rolling(10).max().values) for k in P}
    bv={k:(P[k]['vol']>2*pd.Series(P[k]['vol']).shift(1).rolling(60).mean().values) for k in P}
    mom5={k:(P[k]['close']/np.roll(P[k]['close'],5)-1)*1e4 for k in P}
    prims={}
    for k in P:
        prims[f'{k}up']=up[k]; prims[f'{k}brk']=bk[k]
        prims[f'{k}brk10']=bk10[k]; prims[f'{k}vol']=bv[k]
        thr = np.nanpercentile(mom5[k][train_mask], 90)   # TRAIN-ONLY calibration
        prims[f'{k}hot']=mom5[k]>thr
    n_up  = sum(up[k].astype(int) for k in P)
    n_brk = sum(bk[k].astype(int) for k in P)
    n_vol = sum(bv[k].astype(int) for k in P)
    R={}
    K=len(P)
    R['ge_half_up']  = n_up  >= (K//2+1)
    R['ge_most_up']  = n_up  >= K-1
    R['ge2_brk']     = n_brk >= 2
    R['ge3_brk']     = n_brk >= 3
    R['ge2_vol']     = n_vol >= 2
    if 'S' in P and 'G' in P:
        R['spec_hot'] = prims['Shot']&prims['Ghot']
        R['froth']    = up['S']&up['G']&(n_vol>=1)
    base=list(prims.keys())
    for a,b in itertools.combinations(base,2):
        if len(R)>=MAX_RULES: break
        R[f'{a}&{b}']=prims[a]&prims[b]
        if len(R)<MAX_RULES: R[f'{a}&!{b}']=prims[a]&~prims[b]
        if len(R)<MAX_RULES and a[0]!=b[0]: R[f'{a}|{b}']=prims[a]|prims[b]
    for cn in ['ge_half_up','ge2_brk','ge2_vol']:
        for k in P:
            if len(R)>=MAX_RULES: break
            R[f'{cn}&{k}up']=R[cn]&up[k]
    return R

# --------------------------- SEARCH ---------------------------
def stk(w):
    mx=cur=0;r3=0
    for x in w:
        cur=cur+1 if x else 0; mx=max(mx,cur)
        if cur==3: r3+=1
    return mx, r3/max(len(w)-2,1)*100

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--months',type=int,default=6)
    args=ap.parse_args()
    P, mins, months = build(args.months)
    n=len(mins)
    # month id per row for splitting
    mo = pd.to_datetime(mins*60, unit='s')
    ym = np.array([d.year*100+d.month for d in mo])
    uniq = sorted(set(ym))
    if len(uniq)<4: sys.exit("need >=4 months for train/val/test")
    VAL_M, TEST_M = uniq[-2], uniq[-1]
    seg = np.where(ym==TEST_M,2,np.where(ym==VAL_M,1,0))
    print(f"\nmonths: {uniq}  TRAIN={uniq[:-2]}  VAL={VAL_M}  TEST={TEST_M}")
    F = features(P, mins); feats=list(F.columns)
    R = rule_pool(P, seg==0)
    print(f"rule pool: {len(R)}  horizon: {HORIZON}m  rows: {n}")
    okbase=(np.arange(n)>=240)&(np.arange(n)<n-HORIZON)&(~F.isna().any(axis=1).values)
    for tk in P:
        TC=P[tk]['close']
        fwd=np.r_[TC[HORIZON:],[np.nan]*HORIZON]
        y=np.where(fwd>TC,1,np.where(fwd<TC,0,-1))
        ok=okbase&(y>=0)
        scored=[]
        t_start=_time.time(); fitted=0
        eligible=[(rn,mask) for rn,mask in R.items()]
        log(f"SEARCH {tk}: scanning {len(eligible)} rules ...")
        for ri,(rn,mask) in enumerate(eligible):
            m=mask&ok; tr,va=m&(seg==0),m&(seg==1)
            if tr.sum()<3000 or va.sum()<600: continue
            mod=CatBoostClassifier(iterations=ITER,depth=4,learning_rate=0.07,
                                   l2_leaf_reg=8,random_seed=7,verbose=0)
            mod.fit(F[tr][feats],y[tr])
            p=mod.predict_proba(F[va][feats])[:,1]
            sh=p<0.5
            if sh.sum()<300: continue
            conf=0.5-p[sh]; kq=conf>=np.quantile(conf,0.6)
            accv=(y[va][sh][kq]==0).mean()
            scored.append((accv, rn, mod, mask))
            fitted+=1
            per=(_time.time()-t_start)/fitted
            log(f"  {tk} rule {ri+1}/{len(eligible)} fitted#{fitted} [{rn}] val={accv*100:.2f}%  (~{per:.1f}s/rule, ETA {(len(eligible)-ri-1)*per/60:.0f}m)")
        scored.sort(reverse=True)
        print(f"\n========== {tk} — top-{TOPK} by VAL, TEST readout ==========")
        print(f"{'rule':<26}{'VAL%':>7} || {'TEST acc%':>10}{'n':>6}{'/day':>7}{'maxstrk':>8}{'3run%':>7}{'SE±':>5}")
        for accv,rn,mod,mask in scored[:TOPK]:
            m=mask&ok; te=m&(seg==2)
            if te.sum()<100:
                print(f"{rn:<26}{accv*100:>7.2f} || too few TEST rows"); continue
            p=mod.predict_proba(F[te][feats])[:,1]
            sh=p<0.5; conf=0.5-p[sh]
            kq=conf>=np.quantile(conf,0.6)
            win=(y[te][sh][kq]==0); mx,r3=stk(~win)
            se=100*np.sqrt(max(win.mean()*(1-win.mean()),1e-9)/max(kq.sum(),1))
            days=(seg==2).sum()/1440
            print(f"{rn:<26}{accv*100:>7.2f} || {win.mean()*100:>10.2f}{kq.sum():>6}{kq.sum()/days:>7.1f}{mx:>8}{r3:>7.2f}{se:>5.1f}")
    print("\nTEST readout complete. These numbers are the truth. Rerunning with new")
    print("rules/settings after seeing them burns the TEST month — wait for a fresh month.")

if __name__=='__main__':
    main()
