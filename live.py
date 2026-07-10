#!/usr/bin/env python3
"""live.py — TOUCH-FLIP-OUT PRO live bot (BTC 5m Polymarket). Needs tfo_config.pkl.

  python3 live.py            (PAPER_MODE=1 default; PAPER_MODE=0 trades real USDC)

RULES per 5m window:
  t=0   enter side (no-leak side-picker; prev-candle fallback) at the open ask
  live  1s klines vs band = open +/- BAND_BP:
        1st close beyond band against us -> FLIP (buy opposite; pair merges -> new side)
        2nd band cross                   -> SCRATCH (buy opposite again; flat; window done)
  t=300 unresolved position resolves with the candle

Every action logs the REAL quotes it saw (open ask both sides, flip ask, scratch ask):
live_log.csv is the measurement that decides whether the backtest economics are real.
State: tfo_state.json | Telegram: TELEGRAM_TOKEN/TELEGRAM_CHAT_ID (console is teed).

pip install numpy lightgbm websockets requests
"""
import asyncio, csv, json, os, pickle, time
import numpy as np
import requests, websockets

CFG = pickle.load(open('tfo_config.pkl', 'rb'))
if os.environ.get('PAPER_MODE', '1') == '0' and os.environ.get('KEY_ROTATED', '') != 'YES':
    raise SystemExit('LIVE MODE BLOCKED: your old private key was exposed in plaintext and is '
                     'compromised. Create a NEW Polygon wallet, move funds, set POLY_PK to the new '
                     'key, then set KEY_ROTATED=YES to confirm. Paper mode runs without this.')
BAND_BP = int(os.environ.get('BAND_BP', CFG.get('band_bp', 3)))
PAPER = os.environ.get('PAPER_MODE', '1') != '0'
STAKE = float(os.environ.get('STAKE_USDC', '5'))
PRICE_CAP = float(os.environ.get('PRICE_CAP', '0.58'))
TG_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')
TG_CHAT = os.environ.get('TELEGRAM_CHAT_ID', '')
POLY_PK = os.environ.get('POLY_PK', '')
POLY_FUNDER = os.environ.get('POLY_FUNDER', '')
POLY_SIG = int(os.environ.get('POLY_SIG_TYPE', '1'))
GAMMA, CLOB = 'https://gamma-api.polymarket.com', 'https://clob.polymarket.com'
STATE, LOG = 'tfo_state.json', 'live_log.csv'
TICKS = ['btc', 'eth', 'sol']
HZ = 7300

_TG_Q = []
def tg_send(text):
    if not TG_TOKEN or not TG_CHAT: return
    try:
        r = requests.post(f'https://api.telegram.org/bot{TG_TOKEN}/sendMessage',
                          json={'chat_id': TG_CHAT, 'text': text}, timeout=4)
        if r.status_code != 200: _real_print('telegram error:', r.text[:150])
    except Exception as e:
        _real_print('telegram failed:', e)

_real_print = print
def print(*a, **k):
    _real_print(*a, **k)
    try: _TG_Q.append(' '.join(str(x) for x in a))
    except Exception: pass

class Buf:
    def __init__(s):
        s.base = None
        s.close = np.full(HZ, np.nan)
        s.vol = np.zeros(HZ); s.buyvol = np.zeros(HZ)
    def put(s, sec, c, v, bv):
        if s.base is None: s.base = sec - 60
        i = sec - s.base
        if i >= HZ:
            sh = i - HZ + 600
            for arr, fill in ((s.close, np.nan), (s.vol, 0.0), (s.buyvol, 0.0)):
                arr[:] = np.roll(arr, -sh); arr[-sh:] = fill
            s.base += sh; i -= sh
        if 0 <= i < HZ:
            s.close[i] = c; s.vol[i] = v; s.buyvol[i] = bv
    def px(s, sec):
        if s.base is None: return np.nan
        i = sec - s.base
        if not (0 <= i < HZ): return np.nan
        j = i
        while j >= 0 and not np.isfinite(s.close[j]): j -= 1
        return s.close[j] if j >= 0 else np.nan

BN = {t: Buf() for t in TICKS}
CLOCK_SKEW = [0.0]

def side_pick(t0):
    b = BN['btc']
    try:
        def rr(sb):
            p0, p1 = b.px(t0 - sb), b.px(t0)
            return p1/p0 - 1.0 if np.isfinite(p0) and p0 > 0 and np.isfinite(p1) else 0.0
        i0 = t0 - b.base
        v1 = b.vol[i0-15:i0].sum(); bv1 = b.buyvol[i0-15:i0].sum()
        v4 = b.vol[i0-60:i0].sum(); bv4 = b.buyvol[i0-60:i0].sum()
        f = dict(r15=rr(15), r30=rr(30), r60=rr(60), r5m=rr(300), r15m=rr(900),
                 fl15=(2*bv1-v1)/v1 if v1 > 0 else 0.0,
                 fl60=(2*bv4-v4)/v4 if v4 > 0 else 0.0,
                 hour=(t0//3600) % 24)
        for s in ('eth', 'sol'):
            p0, p1 = BN[s].px(t0-15), BN[s].px(t0)
            f[f'{s}_r15'] = p1/p0-1.0 if np.isfinite(p0) and p0 > 0 and np.isfinite(p1) else 0.0
        x = np.array([[f[k] for k in CFG['feats']]], dtype=float)
        return CFG['side_model'].predict(x)[0] > 0.5
    except Exception:
        pc0, pc1 = BN['btc'].px(t0-300), BN['btc'].px(t0-1)
        return (pc1 > pc0) if np.isfinite(pc0) and np.isfinite(pc1) else True

def poly_market(t0):
    slug = f'btc-updown-5m-{t0}'
    try:
        r = requests.get(f'{GAMMA}/events/slug/{slug}', timeout=4)
        if r.status_code != 200: return None
        mk = (r.json().get('markets') or [None])[0]
        if not mk: return None
        toks = mk.get('clobTokenIds', '[]')
        toks = json.loads(toks) if isinstance(toks, str) else toks
        return dict(up=toks[0], down=toks[1], slug=slug) if len(toks) >= 2 else None
    except Exception as e:
        print('gamma:', e); return None

def ask(token):
    try:
        r = requests.get(f'{CLOB}/price', params={'token_id': token, 'side': 'SELL'}, timeout=3)
        if r.status_code == 200:
            p = r.json().get('price')
            return float(p) if p is not None else None
    except Exception: pass
    return None

_clob = None
def clob_buy(token, usdc):
    global _clob
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import MarketOrderArgs, OrderType
    if _clob is None:
        _clob = ClobClient(CLOB, key=POLY_PK, chain_id=137,
                           signature_type=POLY_SIG, funder=POLY_FUNDER or None)
        _clob.set_api_creds(_clob.create_or_derive_api_creds())
    o = _clob.create_market_order(MarketOrderArgs(token_id=token, amount=float(usdc), side='BUY'))
    return _clob.post_order(o, OrderType.FOK)

def buy(token, label, shares=None):
    """Fetch real ask (the measurement), then paper-log or live FOK fill.
    entry: spends STAKE USDC. flip/scratch: buys `shares` of the opposite token
    (share-matched pair -> $1/share at resolution; no on-chain merge needed)."""
    a = ask(token)
    if a is None: return None, 'no_quote'
    if a > PRICE_CAP and label == 'entry': return a, 'skip_price'
    if PAPER: return a, 'paper'
    try:
        usdc = STAKE if label == 'entry' else min(shares * a * 1.02, STAKE * 2.5)
        clob_buy(token, usdc)
        return a, 'LIVE'
    except Exception as e:
        print('ORDER ERROR:', e); return a, 'order_failed'

def st_load(): return json.load(open(STATE)) if os.path.exists(STATE) else {'pnl': 0.0, 'w': 0, 'n': 0}
def st_save(s): json.dump(s, open(STATE, 'w'))
def log(row):
    new = not os.path.exists(LOG)
    with open(LOG, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['ts','window','event','side','quote','status',
                                          'band_bp','flips','pnl','total_pnl'])
        if new: w.writeheader()
        w.writerow(row)

async def binance_ws():
    streams = '/'.join(f'{t}usdt@kline_1s' for t in TICKS)
    host = os.environ.get('BINANCE_WS_HOST', 'stream.binance.com:9443')
    while True:
        try:
            async with websockets.connect(f'wss://{host}/stream?streams={streams}',
                                          ping_interval=20) as ws:
                async for msg in ws:
                    k = json.loads(msg)['data']['k']
                    if not k['x']: continue
                    t = k['s'].lower().replace('usdt', '')
                    sec = int(k['t']) // 1000
                    BN[t].put(sec, float(k['c']), float(k['v']), float(k['V']))
                    CLOCK_SKEW[0] = 0.98*CLOCK_SKEW[0] + 0.02*(time.time() - (sec+1))
                    if t == 'btc': await EVENTS.put(sec)
        except Exception as e:
            print('ws:', e); await asyncio.sleep(3)

def prefill():
    host = os.environ.get('BINANCE_REST_HOST', 'api.binance.com')
    now = int(time.time()*1000)
    for t in TICKS:
        try:
            start = now - 4000*1000; got = 0
            for _ in range(5):
                r = requests.get(f'https://{host}/api/v3/klines',
                                 params=dict(symbol=f'{t.upper()}USDT', interval='1s',
                                             startTime=start, limit=1000), timeout=10)
                kl = r.json()
                if not isinstance(kl, list) or not kl: break
                for k in kl: BN[t].put(int(k[0])//1000, float(k[4]), float(k[5]), float(k[9]))
                got += len(kl); start = int(kl[-1][0]) + 1000
                if len(kl) < 1000: break
            print(f'prefill {t}: {got} bars')
        except Exception as e:
            print(f'prefill {t} failed: {e}')

EVENTS = None

async def engine():
    st = st_load()
    win = None    # active window dict
    print(f'TOUCH-FLIP-OUT PRO | band {BAND_BP}bp | paper={PAPER} stake=${STAKE} | pnl ${st["pnl"]}')
    while True:
        sec = await EVENTS.get()
        px = BN['btc'].px(sec)
        if not np.isfinite(px): continue
        # ---- window open ----
        if sec % 300 == 0:
            if win and not win['done']:
                _resolve(win, st, px)
            up = bool(side_pick(sec))
            mkt = poly_market(sec)
            open_px = px
            d = open_px * BAND_BP * 1e-4
            q, status = (None, 'no_market')
            if mkt:
                q, status = buy(mkt['up' if up else 'down'], 'entry')
            active = status in ('paper', 'LIVE')
            win = dict(t0=sec, open=open_px, band=d, up=up, mkt=mkt, flips=0,
                       entry=q, shares=(STAKE / q if q else 0.0), done=not active)
            ts = time.strftime('%H:%M:%S', time.localtime(sec))
            print(f'{ts} OPEN {"UP" if up else "DOWN"} @ {open_px:.1f} | entry ask {q} [{status}] band ±{d:.1f}')
            log(dict(ts=ts, window=sec, event='entry', side='UP' if up else 'DOWN',
                     quote=q, status=status, band_bp=BAND_BP, flips=0, pnl='', total_pnl=st['pnl']))
            continue
        if win is None or win['done'] or win['t0'] != (sec//300)*300: 
            if win and not win['done'] and sec - win['t0'] >= 300:
                _resolve(win, st, px)
            continue
        # ---- band logic ----
        op, d = win['open'], win['band']
        crossed_dn = win['up'] and px <= op - d
        crossed_up = (not win['up']) and px >= op + d
        if crossed_dn or crossed_up:
            opp = win['mkt']['down' if win['up'] else 'up'] if win['mkt'] else None
            q, status = buy(opp, 'flip', win.get('shares', 0.0)) if opp else (None, 'no_market')
            ts = time.strftime('%H:%M:%S', time.localtime(sec))
            if win['flips'] == 0:
                win['up'] = not win['up']; win['flips'] = 1
                print(f'{ts} FLIP -> {"UP" if win["up"] else "DOWN"} @ {px:.1f} | opp ask {q} [{status}]')
                log(dict(ts=ts, window=win['t0'], event='flip', side='UP' if win['up'] else 'DOWN',
                         quote=q, status=status, band_bp=BAND_BP, flips=1, pnl='', total_pnl=st['pnl']))
            else:
                pnl = -0.04 * STAKE / 0.51 if PAPER else ''
                if pnl != '': st['pnl'] = round(st['pnl'] + pnl, 2)
                win['done'] = True; st['n'] += 1; st_save(st)
                print(f'{ts} SCRATCH OUT @ {px:.1f} | opp ask {q} [{status}] | pnl {pnl} total ${st["pnl"]}')
                log(dict(ts=ts, window=win['t0'], event='scratch', side='', quote=q, status=status,
                         band_bp=BAND_BP, flips=2, pnl=pnl, total_pnl=st['pnl']))

def _resolve(win, st, last_px):
    won = (last_px > win['open']) == win['up']
    entry = win['entry'] or 0.51
    pnl = round(((1.0 - entry) if won else -entry) * STAKE / 0.51 - win['flips']*0.02*STAKE/0.51, 2)
    st['pnl'] = round(st['pnl'] + pnl, 2)
    st['n'] += 1; st['w'] += int(won); st_save(st)
    win['done'] = True
    ts = time.strftime('%H:%M:%S')
    print(f'{ts} RESOLVED {"WIN" if won else "LOSS"} ({"flip" if win["flips"] else "hold"}) '
          f'| pnl {pnl} | total ${st["pnl"]} | wr {st["w"]}/{st["n"]}')
    log(dict(ts=ts, window=win['t0'], event='resolved', side='UP' if win['up'] else 'DOWN',
             quote='', status='win' if won else 'loss', band_bp=BAND_BP,
             flips=win['flips'], pnl=pnl, total_pnl=st['pnl']))
    tg_send(f'{"✅" if won else "❌"} {"WIN" if won else "LOSS"} pnl {pnl} total ${st["pnl"]}')

async def tg_flusher():
    while True:
        await asyncio.sleep(2.0)
        if not _TG_Q: continue
        batch, _TG_Q[:] = _TG_Q[:], []
        text = '\n'.join(batch)
        for i in range(0, len(text), 3900):
            chunk = text[i:i+3900]
            await asyncio.get_event_loop().run_in_executor(None, tg_send, chunk)

async def watchdog():
    while True:
        await asyncio.sleep(60)
        sk = CLOCK_SKEW[0]
        warn = f' | ⚠️ clock off ~{sk:.0f}s' if abs(sk) > 10 else ''
        print(f'alive | clock drift {sk:+.1f}s{warn}')

async def _main():
    global EVENTS
    EVENTS = asyncio.Queue()
    prefill()
    tg_send(f'🤖 TFO-PRO online | band {BAND_BP}bp | paper={PAPER}')
    await asyncio.gather(binance_ws(), engine(), tg_flusher(), watchdog())

if __name__ == '__main__':
    asyncio.run(_main())
