"""
S3 Walk-Forward — 完全网络无关。仅依赖 numpy、pandas、本地 parquet 文件。
不 import quanti 包，不调 AkShare，不需要 .env 配置。
"""
import pandas as pd, numpy as np
from pathlib import Path
import time, warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("C:/study/AIWorkspace/quanti/data/clean")
CAPITAL = 90000.0; TOP_N = 3; STOP = -0.10; MA_P = 120; RF = 0.03

# ---------- 数据加载：只用 parquet 文件 ----------
def load_etf(code):
    for sfx in [".SH.parquet", ".SZ.parquet", ".parquet"]:
        p = DATA_DIR / f"{code}{sfx}"
        if p.exists(): df = pd.read_parquet(p); break
    else: raise FileNotFoundError(code)
    df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    return df.sort_values("date").set_index("date")

def load_csi300(): return load_etf("CSI300")

def load_stocks(n=100):
    cand = []
    for fp in sorted(DATA_DIR.glob("*.parquet")):
        df = pd.read_parquet(fp)
        sym = str(df["symbol"].iloc[0])
        if not sym.isdigit(): continue
        df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        t = df[(df.date >= "2015-01-01") & (df.date <= "2025-12-31")]
        if len(t) >= 200:
            cand.append((sym, float(t[(t.date>="2022-01-01")&(t.date<="2025-12-31")].volume.mean())))
    cand.sort(key=lambda x: x[1], reverse=True)
    stocks = {}
    for code, _ in cand[:n]:
        df = pd.read_parquet(DATA_DIR / f"{code}.parquet")
        df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
        stocks[code] = df.sort_values("date").set_index("date")
    return stocks

# ---------- 指标 ----------
def sma(s,p): return s.rolling(p).mean()
def compute_rsi(cl,period=14):
    d=cl.diff(); g=d.clip(lower=0); l=(-d).clip(lower=0)
    ag=g.ewm(alpha=1/period,adjust=False).mean(); al=l.ewm(alpha=1/period,adjust=False).mean()
    return 100-(100/(1+ag/al.replace(0,np.nan)))
def _ws(data,period):
    n=len(data); result=np.full(n,np.nan); seed=period
    while seed<n:
        vv=data[1:seed+1]; vv2=vv[~np.isnan(vv)]
        if len(vv2)>0: result[seed]=np.mean(vv2); break; seed+=1
    if seed>=n: return result
    prev=result[seed]
    for i in range(seed+1,n):
        cur=data[i]
        if np.isnan(cur): result[i]=prev
        elif np.isnan(prev): prev=cur; result[i]=prev
        else: prev=(cur+(period-1)*prev)/period; result[i]=prev
    return result
def adx_np(hi,lo,cl,period=14):
    n=len(cl)
    if n<period*2: return np.full(n,np.nan)
    tr=np.zeros(n); tr[0]=hi[0]-lo[0]
    for i in range(1,n): tr[i]=max(hi[i]-lo[i],abs(hi[i]-cl[i-1]),abs(lo[i]-cl[i-1]))
    pdm=np.zeros(n); mdm=np.zeros(n)
    for i in range(1,n):
        u=hi[i]-hi[i-1]; d=lo[i-1]-lo[i]
        if u>d and u>0: pdm[i]=u
        if d>u and d>0: mdm[i]=d
    atr_s=_ws(tr,period); pdi_s=_ws(pdm,period); mdi_s=_ws(mdm,period)
    pdi=np.divide(pdi_s,atr_s,where=atr_s!=0)*100; mdi=np.divide(mdi_s,atr_s,where=atr_s!=0)*100
    return _ws(np.abs(pdi-mdi)/(pdi+mdi+1e-10)*100,period)
def metrics(daily_ret,years):
    n=len(daily_ret)
    if n<2: return {"CAGR":0,"Sharpe":0,"MaxDD":0}
    cum=np.cumprod(1+daily_ret); tr=cum[-1]-1; y=max(years,0.01)
    cagr=(1+tr)**(1/y)-1; ann_vol=np.std(daily_ret,ddof=1)*np.sqrt(252)
    sharpe=(cagr-RF)/max(ann_vol,1e-10); peak=np.maximum.accumulate(cum)
    return {"CAGR":cagr,"Sharpe":sharpe,"MaxDD":float(np.min((cum-peak)/peak)),"AnnVol":ann_vol,"TotalReturn":tr}

# ---------- 预计算 ----------
def precompute(stocks,csi,start_d,end_d):
    all_d=pd.date_range(start_d,end_d,freq="B")
    valid=sorted(set(all_d).intersection(set(csi.index)))
    me=[]; cm=None; pv=None
    for d in valid:
        if d.month!=cm:
            if cm is not None: me.append(pv); cm=d.month
        pv=d
    me.append(pv); me=sorted([m for m in me if m>=pd.Timestamp("2016-01-01")])
    n_me=len(me); masks={}; scores={}
    for code,df in stocks.items():
        is_t=np.zeros(n_me,dtype=bool); scr=np.zeros(n_me)
        for mi,md in enumerate(me):
            sub=df[df.index<=md]
            if len(sub)<200: continue
            cl,hi,lo,vo=sub["close"],sub["high"],sub["low"],sub["volume"]
            cond=0
            m120=sma(cl,MA_P); v120=m120.iloc[-1]
            if not pd.isna(v120) and cl.iloc[-1]>v120: cond+=1
            if len(cl)>=60:
                if hi.iloc[-20:].max()>hi.iloc[-60:-20].max() and lo.iloc[-20:].min()>lo.iloc[-60:-20].min(): cond+=1
            m20=sma(cl,20); m60=sma(cl,60)
            if (not pd.isna(m20.iloc[-1]) and not pd.isna(m60.iloc[-1]) and not pd.isna(v120)
                    and m20.iloc[-1]>m60.iloc[-1]>v120): cond+=1
            try:
                a=adx_np(hi.values,lo.values,cl.values,14)
                if not np.isnan(a[-1]) and a[-1]>25: cond+=1
            except: pass
            if len(vo)>=22:
                v20=vo.iloc[-21:-1].mean()
                if v20>0 and vo.iloc[-1]>v20*1.2: cond+=1
            if cond>=3:
                is_t[mi]=True
                if len(cl)>=130:
                    try:
                        if cl.iloc[-63]>1e-6 and cl.iloc[-126]>1e-6:
                            r3=cl.iloc[-1]/cl.iloc[-63]-1; r6=cl.iloc[-1]/cl.iloc[-126]-1
                            m3=min(max(r3/0.5,0),1) if r3>0 else 0; m6=min(max(r6/0.8,0),1) if r6>0 else 0
                            mom=(0.5*m3+0.5*m6)*100
                        else: mom=30.0
                    except: mom=30.0
                    vb=max(0,(1-min(float(cl.pct_change().dropna().iloc[-60:].std())/0.04,1)))*100 if len(cl)>=61 else 50
                    scr[mi]=0.6*mom+0.4*vb
        masks[code]=is_t; scores[code]=scr
    return masks,scores,me

def select_top(masks,scores,me,dt,top_n=TOP_N):
    mi=int(np.searchsorted(me,dt,'right')-1)
    if mi<0: return [],[]
    tr=[(c,scores[c][mi]) for c in masks if masks[c][mi] and scores[c][mi]>0]
    tr.sort(key=lambda x:x[1],reverse=True)
    return [t[0] for t in tr[:top_n]],[t[0] for t in tr[:top_n]]

# ---------- 辅助 ----------
def sc(st,c,d):
    if c not in st: return None
    s=st[c][st[c].index<=d]; return None if len(s)==0 else float(s["close"].iloc[-1])
def ec(et,c,d):
    if c not in et: return None
    s=et[c][et[c].index<=d]; return None if len(s)==0 else float(s["close"].iloc[-1])
def log_t(tr,d,s,side,q,p,pnl,tag=""):
    tr.append({"date":d,"symbol":s,"side":side,"qty":q,"price":p,"pnl":pnl,"tag":tag})
def trail_stop(pos,cr,st,et,c,d,pct,trades,tag):
    pr=sc(st,c,d) or ec(et,c,d)
    if pr is None: return False
    p=pos[c]; hwm=p.get("hwm",pr)
    if pr>hwm: hwm=pr; p["hwm"]=hwm
    if (pr/hwm-1)<pct:
        pnl=(pr-p["avg"])*p["qty"]; cr[0]+=pr*p["qty"]
        log_t(trades,d,c,"sell",p["qty"],pr,pnl,tag); del pos[c]; return True
    return False

# ---------- S3 策略 ----------
def s3_strategy(st0,et0,csi0,dates0,masks0,scores0,me0,trades0,stop_pct):
    dr=[]; cash=CAPITAL; pos={}; peq=CAPITAL
    for i,d in enumerate(dates0):
        if i==0: continue
        sd=csi0[csi0.index<=d]; adx_v=20.0
        if len(sd)>=60:
            try:
                a=adx_np(sd["high"].values,sd["low"].values,sd["close"].values,14)
                if not np.isnan(a[-1]): adx_v=float(a[-1])
            except: pass
        trend_st=min(max(adx_v/40.0,0),1.0); is_me=d.month!=dates0[i-1].month
        cr=[cash]
        for c in list(pos): trail_stop(pos,cr,st0,et0,c,d,stop_pct,trades0,"s3_m" if pos[c].get("type")=="mr" else "s3_t")
        cash=cr[0]
        for c in list(pos):
            if pos[c].get("type")=="mr":
                sub=et0[c][et0[c].index<=d]
                if len(sub)>=30:
                    rsi_v=float(compute_rsi(sub["close"],14).iloc[-1])
                    if not pd.isna(rsi_v) and rsi_v>70:
                        pr=float(sub["close"].iloc[-1]); cash+=pr*pos[c]["qty"]; log_t(trades0,d,c,"sell",pos[c]["qty"],pr,(pr-pos[c]["avg"])*pos[c]["qty"],"s3_m"); del pos[c]
        if adx_v<20:
            if is_me:
                for c in list(pos):
                    if pos[c].get("type")!="mr":
                        pr=sc(st0,c,d)
                        if pr: cash+=pr*pos[c]["qty"]; log_t(trades0,d,c,"sell",pos[c]["qty"],pr,(pr-pos[c]["avg"])*pos[c]["qty"],"s3_t"); del pos[c]
                mr_count=sum(1 for p in pos.values() if p.get("type")=="mr")
                if mr_count==0:
                    for code in ["510300","510500","159915"]:
                        if code not in et0: continue
                        sub=et0[code][et0[code].index<=d]
                        if len(sub)<30: continue
                        rsi_v=float(compute_rsi(sub["close"],14).iloc[-1])
                        if not pd.isna(rsi_v) and rsi_v<30:
                            pr=float(sub["close"].iloc[-1]); inv=cash*0.3
                            if inv>=pr*100:
                                q=int(inv/pr/100)*100
                                if q>=100 and q*pr<=cash: cash-=q*pr; pos[code]={"qty":q,"avg":pr,"hwm":pr,"type":"mr"}; log_t(trades0,d,code,"buy",q,pr,0,"s3_m")
                            break
        elif is_me:
            sel,_=select_top(masks0,scores0,me0,d)
            for c in list(pos):
                if pos[c].get("type")!="mr" and c not in sel:
                    pr=sc(st0,c,d)
                    if pr: cash+=pr*pos[c]["qty"]; log_t(trades0,d,c,"sell",pos[c]["qty"],pr,(pr-pos[c]["avg"])*pos[c]["qty"],"s3_t"); del pos[c]
            nw=[s for s in sel if s not in pos]; tc=sum(1 for p in pos.values() if p.get("type")!="mr")
            if nw and tc<TOP_N:
                avail=cash*trend_st; slots=TOP_N-tc; nn=min(len(nw),slots)
                if nn>0 and avail>0:
                    per=avail/nn*0.92
                    for c in nw[:nn]:
                        pr=sc(st0,c,d)
                        if pr and pr>0.01 and per>=pr*100:
                            q=int(per/pr/100)*100
                            if q>=100 and q*pr<=cash: cash-=q*pr; pos[c]={"qty":q,"avg":pr,"hwm":pr,"type":"trend"}; log_t(trades0,d,c,"buy",q,pr,0,"s3_t")
        if is_me and adx_v>=35:
            for c in list(pos):
                if pos[c].get("type")=="mr":
                    pr=ec(et0,c,d)
                    if pr: cash+=pr*pos[c]["qty"]; log_t(trades0,d,c,"sell",pos[c]["qty"],pr,(pr-pos[c]["avg"])*pos[c]["qty"],"s3_m"); del pos[c]
        eq=cash+sum(p["qty"]*(sc(st0,c,d) or ec(et0,c,d) or 0) for c,p in pos.items())
        if peq>0: dr.append((eq-peq)/peq); peq=eq
    return np.array(dr)

# ═══════════════════════════════════════════════════
print("S3 Walk-Forward — 纯本地, 无网络依赖")
print("=" * 60)
t0=time.time()
et={c:load_etf(c) for c in ["510300","510500","159915"]}
csi=load_csi300(); st=load_stocks(100)
dates=sorted(set(pd.date_range("2015-01-01","2021-12-31",freq="B")).intersection(set(csi.index)))
tl=252; tsl=126; pos=0; R=[]

while pos+tl+tsl <= len(dates):
    td=dates[pos+tl:pos+tl+tsl]
    w=pos//tl+1
    t_start=pd.Timestamp(dates[pos]).strftime("%Y-%m-%d")
    t_end=pd.Timestamp(dates[pos+tl-1]).strftime("%Y-%m-%d")
    masks,scr,me_list=precompute(st,csi,t_start,t_end)
    for sp in [-0.08,-0.10,-0.12]:
        _=s3_strategy(st,et,csi,td,masks,scr,me_list,[],sp)
        m=metrics(_,len(td)/252)
        R.append({"w":w,"s":sp*100,"S":m["Sharpe"],"C":m["CAGR"],"D":m["MaxDD"]})
    pos+=tl
    if w>=6: break

print(f"Loaded {len(st)} stocks, {len(dates)} CSI300 rows, no network used")
print(f"Total time: {time.time()-t0:.0f}s")
print()
print(f"{'W':>3s} {'stop':>5s} {'Sharpe':>8s} {'CAGR':>8s} {'MaxDD':>8s}")
for r in R: print(f'{r["w"]:>3d} {r["s"]:>+4.0f}% {r["S"]:>8.3f} {r["C"]*100:>7.2f}% {r["D"]*100:>7.2f}%')
print()
for sp in [-8,-10,-12]:
    s=[r for r in R if r["s"]==sp]
    aS=np.mean([r["S"] for r in s]); aC=np.mean([r["C"]*100 for r in s])
    aD=np.mean([r["D"]*100 for r in s]); wins=sum(1 for r in s if all(r2["S"]<=r["S"] for r2 in R if r2["w"]==r["w"]))
    print(f"stop={sp:+d}%: avg_Sharpe={aS:.3f} avg_CAGR={aC:.2f}% avg_MaxDD={aD:.1f}% best={wins}/6")
print("\nDone — all data from local .parquet files, zero network calls.")
