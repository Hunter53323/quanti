"""
Key Strategy Improvements — Focused Batch Test
===============================================
Tests:
  1. Cross-Asset ETF Trend Rotation
  2. Volatility Regime Filter
  3. Walk-Forward Validation
  4. Weekly vs Monthly Rebalancing
  5. Gradient Stop-Loss

Train: 2015-2021 | Test: 2022-2025
"""
import sys, os, itertools, time, numpy as np
from datetime import datetime, timedelta
os.chdir(r"C:\study\AIWorkspace\quanti")
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
np.seterr(divide='ignore', invalid='ignore')

from quanti.data.storage import DataStorage

CAPITAL = 90000; COMM = 0.00025
T_START = time.time()
print("=" * 90)
print("KEY STRATEGY IMPROVEMENTS — FOCUSED BATCH TEST")
print("=" * 90)

# ═══════════════════ 1. Load ═══════════════════
print("\n[1] Loading data...")
storage = DataStorage()

raw300 = storage.load_bars("510300")
CSI_D = np.array([r.trade_date for r in raw300])
CSI_C = np.array([r.close for r in raw300], dtype=np.float64)
CSI_H = np.array([r.high for r in raw300], dtype=np.float64)
CSI_L = np.array([r.low for r in raw300], dtype=np.float64)

raw_cash = storage.load_bars("511880")
CASH_D = np.array([r.trade_date for r in raw_cash])
CASH_C = np.array([r.close for r in raw_cash], dtype=np.float64)

ETF_CODES = ["510300", "510500", "159915", "510880", "518880"]
ETF_DATA = {}
for code in ETF_CODES:
    raw = storage.load_bars(code)
    if raw:
        ETF_DATA[code] = {
            "d": np.array([r.trade_date for r in raw]),
            "c": np.array([r.close for r in raw], dtype=np.float64),
            "h": np.array([r.high for r in raw], dtype=np.float64),
            "l": np.array([r.low for r in raw], dtype=np.float64),
        }

all_files = sorted(storage.clean_dir.glob("*.parquet"))
stock_codes = [p.stem for p in all_files if len(p.stem)==6 and not p.stem.startswith(("51","58","15","56"))]
STOCK = {}
all_ds = set(CSI_D)
for code in stock_codes:
    raw = storage.load_bars(code)
    if not raw or len(raw) < 200: continue
    STOCK[code] = {
        "d": np.array([r.trade_date for r in raw]),
        "c": np.array([r.close for r in raw], dtype=np.float64),
        "h": np.array([r.high for r in raw], dtype=np.float64),
        "l": np.array([r.low for r in raw], dtype=np.float64),
        "v": np.array([r.volume for r in raw], dtype=np.float64),
    }
    all_ds.update(r.trade_date for r in raw)
ALL_D = sorted(all_ds)
print(f"  Stocks:{len(STOCK)} ETFs:{len(ETF_DATA)} Dates:{len(ALL_D)}")

# ═══════════════════ 2. Indicators ═══════════════════
def _sma(arr, p):
    if len(arr) < p: return np.nan
    return float(np.mean(arr[-p:]))

def _adx_val(h, l, c, p=14):
    n = len(c)
    if n < p*2: return np.nan
    tr = np.zeros(n); tr[0] = h[0]-l[0]
    for i in range(1,n): tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    pdm = np.zeros(n); mdm = np.zeros(n)
    for i in range(1,n):
        up=h[i]-h[i-1]; dn=l[i-1]-l[i]
        if up>dn and up>0: pdm[i]=up
        if dn>up and dn>0: mdm[i]=dn
    atr=np.full(n,np.nan); atr[p]=float(np.mean(tr[1:p+1]))
    for i in range(p+1,n): atr[i]=(tr[i]+(p-1)*atr[i-1])/p
    ps=float(np.mean(pdm[1:p+1])); ms=float(np.mean(mdm[1:p+1]))
    pi=np.full(n,np.nan); mi=np.full(n,np.nan)
    pi[p]=ps/max(atr[p],0.001)*100; mi[p]=ms/max(atr[p],0.001)*100
    for i in range(p+1,n):
        ps=(pdm[i]+(p-1)*ps)/p; ms=(mdm[i]+(p-1)*ms)/p
        pi[i]=min(ps/max(atr[i],0.001)*100,1000); mi[i]=min(ms/max(atr[i],0.001)*100,1000)
    dx=np.abs(pi-mi)/(pi+mi+1e-10)*100
    ax=np.full(n,np.nan)
    seed=float(np.nanmean(dx[p:p*2]))
    ax[p*2-1]=0.0 if np.isnan(seed) else seed; ds=ax[p*2-1]
    for i in range(p*2,n):
        vi=dx[i] if not np.isnan(dx[i]) else ds; ds=(vi+(p-1)*ds)/p; ax[i]=ds
    return ax[-1]

def _rv(arr, w=20):
    if len(arr)<w+2: return np.nan
    rets=np.diff(arr[-w-1:])/(arr[-w-1:-1]+1e-10)
    return np.nanstd(rets)*np.sqrt(252)

def _price(code, dt):
    if code in STOCK:
        sd=STOCK[code]; idx=np.searchsorted(sd["d"],dt,side="right")-1
        return sd["c"][idx] if idx>=0 else None
    if code in ETF_DATA:
        ed=ETF_DATA[code]; idx=np.searchsorted(ed["d"],dt,side="right")-1
        return ed["c"][idx] if idx>=0 else None
    return None

def _cp(dt):
    idx=np.searchsorted(CASH_D,dt,side="right")-1
    return CASH_C[idx] if idx>=0 else 100.0

def get_monthly(s,e):
    m=[]
    for d in ALL_D:
        if d<s or d>e: continue
        dm=d[4:6]
        if not m or dm!=m[-1][4:6]: m.append(d)
    return m

def get_weekly(s,e):
    w=[]; seen=set()
    for d in ALL_D:
        if d<s or d>e: continue
        dt=datetime(int(d[:4]),int(d[4:6]),int(d[6:8]))
        wk=dt.isocalendar()[1]; yrwk=f"{d[:4]}{wk:02d}"
        if yrwk not in seen: seen.add(yrwk); w.append(d)
    return sorted(w)

# ═══════════════════ 3. State Machine (same as V1 best) ═══════════════════
print("\n[2] Building state machine...")
n_csi=len(CSI_C)

stock_ma={}
for code,sd in STOCK.items():
    c=sd["c"]; d=sd["d"]
    if len(c)<21: continue
    cs=np.cumsum(np.insert(c,0,0.0)); m20=np.full(len(c),np.nan)
    m20[19:]=(cs[20:]-cs[:-20])/20.0
    stock_ma[code]=(d,c>m20)

def _breadth(dt):
    cnt,tot=0,0
    for code,(da,aa) in stock_ma.items():
        idx=np.searchsorted(da,dt,side="right")-1
        if idx<19: continue
        tot+=1;cnt+=1 if aa[idx] else 0
    return cnt/tot*100 if tot>0 else 50

ma120_sma=np.full(n_csi,np.nan)
cs120=np.cumsum(np.insert(CSI_C,0,0.0))
ma120_sma[119:]=(cs120[120:]-cs120[:-120])/120.0

# Full ADX
adx_full=np.full(n_csi,np.nan); px=14
tr_x=np.zeros(n_csi); tr_x[0]=CSI_H[0]-CSI_L[0]
for i in range(1,n_csi): tr_x[i]=max(CSI_H[i]-CSI_L[i],abs(CSI_H[i]-CSI_C[i-1]),abs(CSI_L[i]-CSI_C[i-1]))
pdm_x=np.zeros(n_csi); mdm_x=np.zeros(n_csi)
for i in range(1,n_csi):
    up=CSI_H[i]-CSI_H[i-1]; dn=CSI_L[i-1]-CSI_L[i]
    if up>dn and up>0: pdm_x[i]=up
    if dn>up and dn>0: mdm_x[i]=dn
atr_x=np.full(n_csi,np.nan); atr_x[px]=float(np.mean(tr_x[1:px+1]))
for i in range(px+1,n_csi): atr_x[i]=(tr_x[i]+(px-1)*atr_x[i-1])/px
ps_x=float(np.mean(pdm_x[1:px+1])); ms_x=float(np.mean(mdm_x[1:px+1]))
pi_x=np.full(n_csi,np.nan); mi_x=np.full(n_csi,np.nan)
pi_x[px]=ps_x/max(atr_x[px],0.001)*100; mi_x[px]=ms_x/max(atr_x[px],0.001)*100
for i in range(px+1,n_csi):
    ps_x=(pdm_x[i]+(px-1)*ps_x)/px; ms_x=(mdm_x[i]+(px-1)*ms_x)/px
    pi_x[i]=min(ps_x/max(atr_x[i],0.001)*100,1000); mi_x[i]=min(ms_x/max(atr_x[i],0.001)*100,1000)
dx_x=np.abs(pi_x-mi_x)/(pi_x+mi_x+1e-10)*100
seed_x=float(np.nanmean(dx_x[px:px*2]))
adx_full[px*2-1]=0.0 if np.isnan(seed_x) else seed_x; ds_x=adx_full[px*2-1]
for i in range(px*2,n_csi):
    vi=dx_x[i] if not np.isnan(dx_x[i]) else ds_x; ds_x=(vi+(px-1)*ds_x)/px; adx_full[i]=ds_x

breadth_arr=np.array([_breadth(d) for d in CSI_D])
above_ma=(CSI_C>ma120_sma)&(~np.isnan(ma120_sma))

def build_map(adx_th=25, br_th=45, cbr=5, crb=2):
    raw=np.full(n_csi,0,dtype=int)
    for i in range(120,n_csi):
        if above_ma[i]:
            a_ok=not np.isnan(adx_full[i]) and adx_full[i]>adx_th
            b_ok=not np.isnan(breadth_arr[i]) and breadth_arr[i]>br_th
            raw[i]=2 if (a_ok and b_ok) else 1
    conf=np.full(n_csi,0,dtype=int)
    for i in range(1,n_csi):
        rs=raw[i]
        if conf[i-1]==0:
            if i>=cbr-1 and np.all(raw[i-cbr+1:i+1]>=1):
                if i>=crb-1 and np.all(raw[i-crb+1:i+1]==2): conf[i]=2
                else: conf[i]=1
            else: conf[i]=0
        elif conf[i-1]==1:
            if rs==0: conf[i]=0
            elif i>=crb-1 and np.all(raw[i-crb+1:i+1]==2): conf[i]=2
            else: conf[i]=1
        elif conf[i-1]==2:
            if rs==0: conf[i]=0
            elif rs==1: conf[i]=1
            else: conf[i]=2
    return {CSI_D[i]: int(conf[i]) for i in range(n_csi)}

BEST_MAP=build_map(25,45,5,2)
nb=sum(1 for d in get_monthly("20220101","20251231") if BEST_MAP.get(d,0)>0)
print(f"  State machine: {nb} non-BEAR months in Test")

# Precompute stock scores
print("  Precomputing stock scores...")
all_ms=get_monthly("20150101","20251231")
PRE={}
for code,sd in STOCK.items():
    PRE[code]={}
    for rd in all_ms:
        idx=np.searchsorted(sd["d"],rd,side="right")
        if idx<260: continue
        c=sd["c"][idx-260:idx];h=sd["h"][idx-260:idx];l=sd["l"][idx-260:idx];v=sd["v"][idx-260:idx];n=len(c)
        ma120s=np.full(n,np.nan); cs120s=np.cumsum(np.insert(c,0,0.0)); ma120s[119:]=(cs120s[120:]-cs120s[:-120])/120.0
        above=1.0 if (c[-1]>ma120s[-1] and not np.isnan(ma120s[-1])) else 0.0
        rh=np.max(h[-20:]);ph=np.max(h[-60:-20]);rl=np.min(l[-20:]);pl=np.min(l[-60:-20])
        hhll=1.0 if(rh>ph and rl>pl)else 0.0
        m20=np.full(n,np.nan);cs20=np.cumsum(np.insert(c,0,0.0));m20[19:]=(cs20[20:]-cs20[:-20])/20.0
        m60=np.full(n,np.nan);cs60=np.cumsum(np.insert(c,0,0.0));m60[59:]=(cs60[60:]-cs60[:-60])/60.0
        al=0.0
        if(not np.isnan(m20[-1])and not np.isnan(m60[-1])and not np.isnan(ma120s[-1])and m20[-1]>m60[-1]>ma120s[-1]):al=1.0
        av=_adx_val(h,l,c,14);an=min(max((av-15)/35,0),1)if not np.isnan(av)else 0
        r3=c[-1]/c[-63]-1 if c[-63]>1e-6 else 0;r6=c[-1]/c[-126]-1 if c[-126]>1e-6 else 0
        m3=min(max(r3/0.5,0),1)if r3>0 else max(r3/0.3,-1);m6=min(max(r6/0.8,0),1)if r6>0 else max(r6/0.5,-1)
        ms=(0.5*m3+0.5*m6)*100
        w=c[-61:];dr=np.diff(w)/(w[:-1]+1e-10);vs=(1-min(np.nanstd(dr)/0.05,1))*100
        tc=(0.35*above+0.25*al+0.20*an+0.20*hhll)*100
        PRE[code][rd]=0.60*ms+0.30*tc+0.10*vs
print(f"  Precomputed {len(PRE)} stocks")

# ═══════════════════ 4. Core Backtest Engine ═══════════════════
def run_bt(smap, s, e, freq="monthly", gs=False, vol_th=0.35, vol_pen=0.5):
    rebal=get_weekly(s,e) if freq=="weekly" else get_monthly(s,e)
    if len(rebal)<6: return None
    cash=CAPITAL; hld={}; cetf=0.0; eq=[CAPITAL]; max_e=CAPITAL; pst=-1; sc={0:0,1:0,2:0}
    for rd in rebal:
        mkt=smap.get(rd,0); sc[mkt]+=1; scd=(mkt!=pst)
        for sym in list(hld.keys()):
            p=_price(sym,rd)
            if p is None or p<0.01: cash+=hld[sym].get("qty",0)*hld[sym].get("price",10)*0.7;del hld[sym];continue
            hld[sym]["price"]=p
        cp=_cp(rd); cv=cetf*cp
        tot=cash+cv+sum(h["qty"]*h.get("price",0) for h in hld.values())
        for sym in list(hld.keys()):
            p=hld[sym].get("price",0)
            if p<=0: continue
            hwm=hld[sym].get("hwm",p)
            if p>hwm: hld[sym]["hwm"]=p
            ddp=(p/hwm-1)*100 if hwm>0 else 0
            if gs:
                if ddp<-15: cash+=hld[sym]["qty"]*p*(1-COMM);del hld[sym]
                elif ddp<-10:
                    q=hld[sym]["qty"];sq=q//2
                    if sq>=100: cash+=sq*p*(1-COMM);hld[sym]["qty"]-=sq
                elif ddp<-5:
                    q=hld[sym]["qty"];sq=q//4
                    if sq>=100: cash+=sq*p*(1-COMM);hld[sym]["qty"]-=sq
            else:
                if ddp<-10: cash+=hld[sym]["qty"]*p*(1-COMM);del hld[sym]
        tot=cash+cetf*cp+sum(h["qty"]*h.get("price",0) for h in hld.values())
        if tot>max_e: max_e=tot
        dd=(max_e-tot)/max_e if max_e>0 else 0
        if dd>0.15:
            for sym in list(hld.keys()): cash+=hld[sym]["qty"]*hld[sym].get("price",0)*(1-COMM);del hld[sym]
            cash+=cetf*cp*(1-COMM);cetf=0;tot=cash+cetf*cp
        if mkt==0:
            if scd or not cetf:
                for sym in list(hld.keys()): cash+=hld[sym]["qty"]*hld[sym].get("price",0)*(1-COMM);del hld[sym]
                cash+=cetf*cp*(1-COMM);cetf=0
                if cp>0 and cash>0: cetf=cash/cp;cash=0.0
            eq.append(cash+cetf*cp);pst=mkt;continue
        if cetf: cash+=cetf*cp*(1-COMM);cetf=0
        psz=0.5 if mkt==1 else 1.0;tn=3 if mkt==1 else 5
        # Vol filter
        csi_rv=_rv(CSI_C[:np.searchsorted(CSI_D,rd,side="right")],20)
        if not np.isnan(csi_rv) and csi_rv>vol_th: psz*=vol_pen
        scored=[(cd,PRE[cd][rd]) for cd in PRE if rd in PRE[cd]]
        if not scored:eq.append(tot);pst=mkt;continue
        scored.sort(key=lambda x:x[1],reverse=True)
        sel={s[0] for s in scored[:tn]}
        for sym in list(hld.keys()):
            if sym not in sel: cash+=hld[sym]["qty"]*hld[sym].get("price",0)*(1-COMM);del hld[sym]
        np2=max(len(sel),1);per_s=tot/np2*0.90
        for sym in sel:
            p=_price(sym,rd)
            if p is None or p<0.01: continue
            tq=int(per_s/p/100)*100
            if tq<100: continue
            if sym in hld:
                diff=tq-hld[sym]["qty"]
                if abs(diff)>=100:
                    cst=abs(diff)*p
                    if diff>0 and cash>=cst*(1+COMM): cash-=cst*(1+COMM);hld[sym]["qty"]=tq
                    elif diff<0: cash+=cst*(1-COMM);hld[sym]["qty"]=tq
            else:
                cst=tq*p
                if cash>=cst*(1+COMM): cash-=cst*(1+COMM);hld[sym]={"qty":tq,"price":p,"hwm":p}
        leftover=cash
        if cp>0 and leftover>0: cetf=leftover/cp;cash=0.0
        else: cash=leftover;cetf=0.0
        tot=cash+cetf*cp+sum(h["qty"]*h.get("price",0) for h in hld.values())
        eq.append(tot);pst=mkt
    eq_arr=np.array(eq)
    if len(eq_arr)<2 or eq_arr[0]<=0: return None
    ny=len(eq_arr)/12.0
    if ny<0.5: return None
    cagr=((eq_arr[-1]/eq_arr[0])**(1/ny)-1)*100
    mr=np.diff(eq_arr)/(eq_arr[:-1]+1e-10)
    sh=np.mean(mr)/(np.std(mr)+1e-10)*np.sqrt(12) if len(mr)>1 else 0
    peak=eq_arr[0];mdd=0.0
    for v in eq_arr:
        if v>peak:peak=v
        if(peak-v)/peak*100>mdd:mdd=(peak-v)/peak*100
    return {"cagr":cagr,"sharpe":sh,"maxdd":mdd,"total_ret":(eq_arr[-1]/eq_arr[0]-1)*100,"sc":sc}

# ═══════════════════ 5. Run All Tests ═══════════════════
print("\n"+"="*90)
print("RESULTS")
print("="*90)

# 5a. Baseline
r_bl=run_bt(BEST_MAP,"20220101","20251231")
r_bl_tr=run_bt(BEST_MAP,"20150101","20211231")
print(f"\nBaseline (V1 best): Test C={r_bl['cagr']:+.1f}% D={r_bl['maxdd']:.1f}% Sh={r_bl['sharpe']:.3f} | Train C={r_bl_tr['cagr']:+.1f}%")

# 5b. Cross-Asset ETF Rotation
print(f"\n--- Test 1: Cross-Asset ETF Trend Rotation ---")
ETF_SC={}
for code,ed in ETF_DATA.items():
    ETF_SC[code]={}
    for rd in all_ms:
        idx=np.searchsorted(ed["d"],rd,side="right")
        if idx<130: continue
        c=ed["c"][idx-130:idx];n=len(c)
        ma120e=np.full(n,np.nan);cs120e=np.cumsum(np.insert(c,0,0.0));ma120e[119:]=(cs120e[120:]-cs120e[:-120])/120.0
        above_e=1.0 if(c[-1]>ma120e[-1] and not np.isnan(ma120e[-1]))else 0.0
        ret60=(c[-1]/c[-60]-1)if c[-60]>1e-6 else 0
        rv=_rv(c,20)if not np.isnan(_rv(c,20))else 0.20
        ret_n=min(max(ret60/0.3,-1),1);vol_n=min(1.0/max(rv,0.05),1.0)
        ETF_SC[code][rd]=0.4*above_e+0.3*ret_n+0.3*vol_n

def run_etf_rot(s,e,tn=2):
    rebal=get_monthly(s,e)
    if len(rebal)<6: return None
    cash=CAPITAL;hld={};eq=[CAPITAL];max_e=CAPITAL
    for rd in rebal:
        for sym in list(hld.keys()):
            p=_price(sym,rd)
            if p is None: del hld[sym];continue
            hld[sym]["price"]=p
        tot=cash+sum(h["qty"]*h.get("price",0) for h in hld.values())
        if tot>max_e: max_e=tot
        scored=[(cd,ETF_SC[cd][rd]) for cd in ETF_SC if rd in ETF_SC[cd]]
        if not scored:eq.append(tot);continue
        scored.sort(key=lambda x:x[1],reverse=True)
        sel={s[0] for s in scored[:tn]}
        for sym in list(hld.keys()):
            if sym not in sel: cash+=hld[sym]["qty"]*hld[sym].get("price",0)*(1-COMM);del hld[sym]
        np2=max(len(sel),1);per_s=tot/np2*0.90
        for sym in sel:
            p=_price(sym,rd)
            if p is None or p<0.01: continue
            tq=int(per_s/p/100)*100
            if tq<100: continue
            if sym in hld:
                diff=tq-hld[sym]["qty"]
                if abs(diff)>=100:
                    cst=abs(diff)*p
                    if diff>0 and cash>=cst*(1+COMM): cash-=cst*(1+COMM);hld[sym]["qty"]=tq
                    elif diff<0: cash+=cst*(1-COMM);hld[sym]["qty"]=tq
            else:
                cst=tq*p
                if cash>=cst*(1+COMM): cash-=cst*(1+COMM);hld[sym]={"qty":tq,"price":p,"hwm":p}
        tot=cash+sum(h["qty"]*h.get("price",0) for h in hld.values())
        eq.append(tot)
    eq_arr=np.array(eq)
    if len(eq_arr)<2:return None
    ny=len(eq_arr)/12.0;cagr=((eq_arr[-1]/eq_arr[0])**(1/ny)-1)*100
    mr=np.diff(eq_arr)/(eq_arr[:-1]+1e-10)
    sh=np.mean(mr)/(np.std(mr)+1e-10)*np.sqrt(12)if len(mr)>1 else 0
    peak=eq_arr[0];mdd=0.0
    for v in eq_arr:
        if v>peak:peak=v
        if(peak-v)/peak*100>mdd:mdd=(peak-v)/peak*100
    return{"cagr":cagr,"sharpe":sh,"maxdd":mdd,"total_ret":(eq_arr[-1]/eq_arr[0]-1)*100}

for tn in[2,3]:
    r=run_etf_rot("20220101","20251231",tn);rt=run_etf_rot("20150101","20211231",tn)
    if r and rt: print(f"  ETF Rot (Top {tn}): Test C={r['cagr']:+.1f}% D={r['maxdd']:.1f}% Sh={r['sharpe']:.3f} | Train C={rt['cagr']:+.1f}% D={rt['maxdd']:.1f}%")
print(f"  ETFs: {', '.join(ETF_CODES)}")

# 5c. Vol Regime Filter
print(f"\n--- Test 2: Volatility Regime Filter ---")
CSI_RV={}
for i in range(20,n_csi):
    rets=np.diff(CSI_C[i-20:i+1])/(CSI_C[i-20:i]+1e-10)
    CSI_RV[CSI_D[i]]=np.nanstd(rets)*np.sqrt(252)
test_rvs=[CSI_RV[d] for d in CSI_RV if"20220101"<=d<="20251231"]
print(f"  CSI300 20d RV: mean={np.mean(test_rvs)*100:.1f}% median={np.median(test_rvs)*100:.1f}% >30%: {sum(1 for v in test_rvs if v>0.30)}/{len(test_rvs)} days")
for vt in[0.25,0.30,0.35]:
    for vp in[0.5,0.75]:
        r=run_bt(BEST_MAP,"20220101","20251231",vol_th=vt,vol_pen=vp)
        r_tr=run_bt(BEST_MAP,"20150101","20211231",vol_th=vt,vol_pen=vp)
        if r and r_tr: print(f"  th={vt:.0%} pen={vp:.0%}: Test C={r['cagr']:+.1f}% D={r['maxdd']:.1f}% | Train C={r_tr['cagr']:+.1f}%")

# 5d. Walk-Forward
print(f"\n--- Test 3: Walk-Forward Validation ---")
wf_grid=list(itertools.product([22,25],[45,50],[3,5],[2,3]))
wf_anchors=[f"{yr}0101" for yr in range(2018,2025)]
wf_res=[]
for anchor in wf_anchors:
    dt_anchor=datetime(int(anchor[:4]),1,1)
    train_e=dt_anchor-timedelta(days=1)
    train_s=train_e-timedelta(days=36*30)
    ts=train_s.strftime("%Y%m%d");te=train_e.strftime("%Y%m%d")
    test_s=anchor;dt2=dt_anchor+timedelta(days=365);test_e=dt2.strftime("%Y%m%d")
    best_c=-999;best_p=None
    for at,br,cb,cr in wf_grid:
        sm=build_map(at,br,cb,cr)
        r=run_bt(sm,ts,te)
        if r and r["cagr"]>best_c: best_c=r["cagr"];best_p=(at,br,cb,cr)
    if best_p:
        sm2=build_map(*best_p)
        r2=run_bt(sm2,test_s,test_e)
        if r2:
            csi_r=(CSI_C[np.searchsorted(CSI_D,test_e,side="right")-1]/CSI_C[np.searchsorted(CSI_D,test_s)]-1)*100
            wf_res.append({"anchor":anchor,"train":f"{ts[:6]}-{te[:6]}","test":f"{test_s[:6]}-{test_e[:6]}",
                           "params":best_p,"cagr":round(r2["cagr"],2),"dd":round(r2["maxdd"],2),"csi":round(csi_r,2)})

for w in wf_res:
    ps=f"ADX={w['params'][0]} B={w['params'][1]} N={w['params'][2]} M={w['params'][3]}"
    print(f"  {w['anchor']} {w['train']}->{w['test']}: {ps} | C={w['cagr']:+.1f}% D={w['dd']:.1f}% CSI300={w['csi']:+.1f}%")
if wf_res:
    avg_c=np.mean([w["cagr"] for w in wf_res]); pos=sum(1 for w in wf_res if w["cagr"]>0)
    print(f"  Summary: {len(wf_res)} windows, {pos}/{len(wf_res)} positive, avg CAGR={avg_c:+.1f}%")

# 5e. Weekly vs Monthly
print(f"\n--- Test 4: Weekly vs Monthly ---")
for freq in["monthly","weekly"]:
    r=run_bt(BEST_MAP,"20220101","20251231",freq);rt=run_bt(BEST_MAP,"20150101","20211231",freq)
    if r and rt: print(f"  {freq:8s}: Test C={r['cagr']:+.1f}% D={r['maxdd']:.1f}% Sh={r['sharpe']:.3f} | Train C={rt['cagr']:+.1f}% D={rt['maxdd']:.1f}%")

# 5f. Gradient Stop
print(f"\n--- Test 5: Gradient Stop-Loss ---")
for gs in[False,True]:
    label="Gradient(-5/25%,-10/50%,-15/100%)" if gs else "Fixed -10%"
    r=run_bt(BEST_MAP,"20220101","20251231",gs=gs);rt=run_bt(BEST_MAP,"20150101","20211231",gs=gs)
    if r and rt: print(f"  {label}: Test C={r['cagr']:+.1f}% D={r['maxdd']:.1f}% Sh={r['sharpe']:.3f} | Train C={rt['cagr']:+.1f}%")

# Final summary
si=np.searchsorted(CSI_D,"20220101");ei=np.searchsorted(CSI_D,"20251231",side="right")-1
seg=CSI_C[si:ei+1];cagr_bm=((seg[-1]/seg[0])**(1/4)-1)*100
print(f"\n{'='*90}")
print(f"SUMMARY")
print(f"{'='*90}")
print(f"  CSI300 B&H:    CAGR={cagr_bm:+.1f}%")
print(f"  V1 State Machine (best): CAGR=+7.0% MaxDD=5.1% Sharpe=0.787")
print(f"  V1 State Machine (base): CAGR=+4.4% MaxDD=9.1% Sharpe=0.515")
print(f"  Total time: {time.time()-T_START:.0f}s")
