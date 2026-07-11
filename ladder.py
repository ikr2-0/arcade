#!/usr/bin/env python3
"""ladder.py — 15m-cycle maker-ladder bot (Polymarket BTC 5m). Runs BESIDE live.py.

CYCLE (aligned epoch%900==0, 3 windows):
  rung1: GTC limit at LIMIT_PX (0.50) on SIDE (default DOWN). Unfilled at window end -> cancel, rung skipped.
  win (candle our color) -> cycle DONE, skip remaining rungs.
  fill+loss -> next rung sized to recover cycle losses + 10% of base.
  all 3 lose -> cycle over, RESET to base (no carry: carry = account bomb; CARRY=1 to override, not advised).
Base = BASE_PCT of balance (live) or paper bankroll. PAPER_MODE=1 default.
"""
import asyncio, csv, json, os, time
import numpy as np, requests, websockets

PAPER = os.environ.get('PAPER_MODE', '1') != '0'
if not PAPER and os.environ.get('KEY_ROTATED', '') != 'YES':
    raise SystemExit('LIVE BLOCKED: set KEY_ROTATED=YES after rotating the exposed key.')
SIDE = os.environ.get('SIDE', 'DOWN').upper()          # we bet each candle is this color
LIMIT_PX = float(os.environ.get('LIMIT_PX', '0.50'))
BASE_PCT = float(os.environ.get('BASE_PCT', '0.005'))
TARGET = float(os.environ.get('TARGET', '0.10'))       # +10% of base per winning cycle
BANK0 = float(os.environ.get('BANKROLL_START', '100'))
CARRY = os.environ.get('CARRY', '0') == '1'
POLY_PK = os.environ.get('POLY_PK',''); POLY_FUNDER = os.environ.get('POLY_FUNDER','')
POLY_SIG = int(os.environ.get('POLY_SIG_TYPE','1'))
TG = os.environ.get('TELEGRAM_TOKEN',''); TGC = os.environ.get('TELEGRAM_CHAT_ID','')
GAMMA, CLOB = 'https://gamma-api.polymarket.com', 'https://clob.polymarket.com'
STATE, LOG = 'ladder_state.json', 'ladder_log.csv'

def tg(t):
    if TG and TGC:
        try: requests.post(f'https://api.telegram.org/bot{TG}/sendMessage',
                           json={'chat_id': TGC, 'text': t}, timeout=4)
        except Exception: pass
def log(row):
    new = not os.path.exists(LOG)
    with open(LOG,'a',newline='') as f:
        w = csv.DictWriter(f, fieldnames=['ts','cycle','rung','event','stake','px','pnl','total'])
        if new: w.writeheader()
        w.writerow(row)
def st_load(): return json.load(open(STATE)) if os.path.exists(STATE) else {'pnl':0.0,'cyc':0,'win':0}
def st_save(s): json.dump(s, open(STATE,'w'))

_cl = None
def clob():
    global _cl
    if _cl is None:
        from py_clob_client_v2.client import ClobClient
        _cl = ClobClient(CLOB, 137, key=POLY_PK, signature_type=POLY_SIG, funder=POLY_FUNDER or None)
        _cl.set_api_creds(_cl.create_or_derive_api_key())
    return _cl
def balance():
    if PAPER: return None
    try:
        from py_clob_client_v2.clob_types import BalanceAllowanceParams, AssetType
        r = clob().get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
        b = float(r.get('balance',0))/1e6
        return b if b > 0 else None
    except Exception: return None
def market(t0):
    try:
        r = requests.get(f'{GAMMA}/events/slug/btc-updown-5m-{t0}', timeout=4)
        mk = (r.json().get('markets') or [None])[0]
        if not mk: return None
        tk = mk.get('clobTokenIds','[]'); tk = json.loads(tk) if isinstance(tk,str) else tk
        return {'UP': tk[0], 'DOWN': tk[1]} if len(tk) >= 2 else None
    except Exception: return None
def place_limit(token, usdc):
    """GTC limit at LIMIT_PX. Returns order_id or 'paper'."""
    if PAPER: return 'paper'
    try:
        from py_clob_client_v2.clob_types import OrderArgs, OrderType
        o = clob().create_order(OrderArgs(price=LIMIT_PX, size=round(usdc/LIMIT_PX, 2),
                                          side='BUY', token_id=token))
        r = clob().post_order(o, OrderType.GTC)
        return str(r.get('orderID', r))[:48] if isinstance(r, dict) else str(r)[:48]
    except Exception as e:
        print('ORDER ERROR:', e); return None
def order_filled(oid):
    if oid == 'paper': return True
    try:
        o = clob().get_order(oid)
        if isinstance(o, dict):
            for k in ('size_matched','sizeMatched','matched_amount','matchedAmount','filled_size','filledSize'):
                v = o.get(k)
                if v is not None and float(v) > 0: return True
            s = str(o.get('status','')).upper()
            if s in ('MATCHED','FILLED','COMPLETE','EXECUTED'): return True
            if s in ('LIVE','OPEN','PENDING','UNMATCHED'): return False
    except Exception as e:
        print('get_order:', e)
    # last resort: a filled order cannot be canceled
    try:
        clob().cancel(oid)
        return False          # cancel succeeded -> it was resting -> unfilled
    except Exception:
        return True           # cancel refused -> already matched
def cancel(oid):
    if oid and oid != 'paper':
        try: clob().cancel(oid)
        except Exception: pass

CLOSE = {}   # sec -> btc close
async def ws():
    host = os.environ.get('BINANCE_WS_HOST', 'stream.binance.com:9443')
    while True:
        try:
            async with websockets.connect(f'wss://{host}/stream?streams=btcusdt@kline_1s',
                                          ping_interval=20) as w:
                async for m in w:
                    k = json.loads(m)['data']['k']
                    if k['x']:
                        CLOSE[int(k['t'])//1000] = float(k['c'])
                        if len(CLOSE) > 4000:
                            for s in sorted(CLOSE)[:1000]: CLOSE.pop(s, None)
        except Exception as e:
            print('ws:', e); await asyncio.sleep(3)
def px_at(sec):
    for b in range(0, 10):
        if sec-b in CLOSE: return CLOSE[sec-b]
    return None

async def engine():
    st = st_load()
    print(f'LADDER | side={SIDE} limit={LIMIT_PX} base={BASE_PCT:.1%} target={TARGET:.0%}/cycle '
          f'| paper={PAPER} | pnl ${st["pnl"]:.2f}')
    tg(f'🪜 LADDER online | {SIDE}@{LIMIT_PX} | paper={PAPER}')
    while True:
        now = int(time.time())
        nxt = ((now//900)+1)*900
        await asyncio.sleep(max(1, nxt-now))
        bank = BANK0 + st['pnl']   # balance() undercounts unredeemed wins; wire redeem later
        base = max(1.0, BASE_PCT*bank)
        losses = 0.0; won = False; st['cyc'] += 1
        for rung in (0,1,2):
            t0 = nxt + rung*300
            stake = base if rung == 0 or not CARRY and losses == 0 else (losses + TARGET*base)
            if losses > 0: stake = losses + TARGET*base
            mk = market(t0)
            oid = place_limit(mk[SIDE], stake) if mk else None
            log(dict(ts=time.strftime('%H:%M:%S'), cycle=st['cyc'], rung=rung+1,
                     event='limit_placed' if oid else 'no_market', stake=round(stake,2),
                     px=LIMIT_PX, pnl='', total=round(st['pnl'],2)))
            await asyncio.sleep(max(1, (t0+300) - int(time.time())))
            o, cl = px_at(t0), px_at(t0+299)
            if oid is None or o is None or cl is None:
                cancel(oid); continue
            filled = order_filled(oid)
            if not filled:
                pass  # cancel already attempted inside order_filled
                log(dict(ts=time.strftime('%H:%M:%S'), cycle=st['cyc'], rung=rung+1,
                         event='unfilled', stake=round(stake,2), px='', pnl=0, total=round(st['pnl'],2)))
                continue
            red = cl < o
            win = red if SIDE == 'DOWN' else (cl > o)
            pnl = stake*(1.0/LIMIT_PX - 1.0) if win else -stake
            st['pnl'] = round(st['pnl'] + pnl, 2); st_save(st)
            log(dict(ts=time.strftime('%H:%M:%S'), cycle=st['cyc'], rung=rung+1,
                     event='win' if win else 'loss', stake=round(stake,2), px=LIMIT_PX,
                     pnl=round(pnl,2), total=st['pnl']))
            tg(f'{"✅" if win else "❌"} rung{rung+1} {"WIN" if win else "LOSS"} '
               f'stake ${stake:.2f} pnl {pnl:+.2f} | total ${st["pnl"]:.2f}')
            if win:
                won = True; st['win'] += 1; st_save(st)
                tg('💰 WIN not auto-redeemed: claim it in Polymarket UI (Portfolio > Claim)')
                break
            losses += stake
        if not won and losses > 0:
            tg(f'🔻 cycle {st["cyc"]} failed: -${losses:.2f} | reset to base | total ${st["pnl"]:.2f}')

async def main():
    await asyncio.gather(ws(), engine())
if __name__ == '__main__':
    asyncio.run(main())
