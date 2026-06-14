"""Multi-industry ETF rotation: MONTHLY rebalance (matching 6-ETF approach)."""
import sys, os; os.chdir(r"C:\study\AIWorkspace\quanti"); sys.path.insert(0,".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from quanti.data.storage import DataStorage
from quanti.strategy.etf_universe import ETF_CATEGORY_MAP, get_eligible_etfs

CAP=90000; COMM=0.00025
storage = DataStorage()

# Load all 25 ETFs
etf_data = {}
for sym in ETF_CATEGORY_MAP:
    raw = storage.load_bars(sym)
    if raw:
        dates=[r.trade_date for r in raw]
        closes=np.array([r.close for r in raw], dtype=np.float64)
        etf_data[sym]=(closes,dates)

all_d=sorted(etf_data["510300"][1])

def sma(arr,p):
    if len(arr)<p:return np.full(len(arr),np.nan)
    o=np.full(len(arr),np.nan); cs=np.cumsum(np.insert(arr,0,0.0))
    o[p-1:]=(cs[p:]-cs[:-p])/p; return o

def price(code,date_str):
    if code not in etf_data:return None
    c,d=etf_data[code]
    for i in range(len(d)-1,-1,-1):
        if d[i]<=date_str:return c[i]
    return None

def score_etf(code,date_str):
    if code not in etf_data:return 0
    c,d=etf_data[code]
    idx=None
    for i in range(len(d)-1,-1,-1):
        if d[i]<=date_str:idx=i+1;break
    if idx is None or idx<140:return 0
    c2=c[:idx]
    # Rising 120MA
    ma_now=np.mean(c2[-120:]); ma_ago=np.mean(c2[-140:-20])
    if not ma_now>ma_ago: return 0
    # Trend
    ma120=sma(c2,120)
    trend=1.0 if not np.isnan(ma120[-1]) and c2[-1]>ma120[-1] else 0.0
    # ADX simplified
    n=len(c2)
    tr=np.zeros(n,dtype=np.float64); tr[0]=0.001
    for i in range(1,n):tr[i]=max(c2[i]*1.005-c2[i]*0.995,abs(c2[i]*1.005-c2[i-1]),abs(c2[i]*0.995-c2[i-1]))
    pdm=np.zeros(n);mdm=np.zeros(n)
    for i in range(1,n):
        up=c2[i]*1.005-c2[i-1]*1.005;dn=c2[i-1]*0.995-c2[i]*0.995
        if up>dn and up>0:pdm[i]=up
        if dn>up and dn>0:mdm[i]=dn
    atr_v=float(np.mean(tr[1:15]));ps=float(np.mean(pdm[1:15]));ms=float(np.mean(mdm[1:15]))
    for i in range(15,n):
        atr_v=(tr[i]+13*atr_v)/14;ps=(pdm[i]+13*ps)/14;ms=(mdm[i]+13*ms)/14
    dx=abs(ps-ms)/(ps+ms+1e-10)*100;adx_val=min(dx/50,1.0)
    # Momentum
    ret=(c2[-1]/c2[-21]-1)*100 if c2[-21]>1e-6 else 0
    mom=min(max(ret/15,0),1) if ret>0 else 0
    return 0.35*trend+0.40*adx_val+0.25*mom

def get_monthly(dates,start,end):
    m=[d for d in dates if start<=d<=end];mo=[]
    for d in m:
        if not mo or d[4:6]!=mo[-1][4:6]:mo.append(d)
    return mo

def run(label,start,end,top_n=3,max_per_cat=2,dd_pct=15):
    rebal=get_monthly(all_d,start,end)
    cash=CAP;hld={};eq=[CAP];pk=CAP
    for rd in rebal:
        total=cash+sum(h['q']*price(c,rd) for c,h in hld.items() if price(c,rd))
        if total>pk:pk=total
        dd=(pk-total)/pk*100 if pk>0 else 0
        if dd>dd_pct:
            for sym in list(hld.keys()):
                p=price(sym,rd) or hld[sym]['p'];cash+=hld[sym]['q']*p*(1-COMM);del hld[sym]
            eq.append(cash);continue

        # Get eligible ETFs (progressive enrollment)
        eligible=get_eligible_etfs(rd)
        scored=[]
        for sym in eligible:
            s=score_etf(sym,rd)
            if s>0:scored.append((sym,s,ETF_CATEGORY_MAP.get(sym,"Other")))

        if not scored:
            for sym in list(hld.keys()):
                p=price(sym,rd) or hld[sym]['p'];cash+=hld[sym]['q']*p*(1-COMM);del hld[sym]
            eq.append(cash);continue

        scored.sort(key=lambda x:x[1],reverse=True)

        # Category cap
        sel=[];cc={}
        for sym,s,cat in scored:
            if cc.get(cat,0)>=max_per_cat:continue
            sel.append(sym);cc[cat]=cc.get(cat,0)+1
            if len(sel)>=top_n:break

        sel_set=set(sel)

        # Rotate out
        for sym in list(hld.keys()):
            if sym not in sel_set:
                p=price(sym,rd) or hld[sym]['p'];cash+=hld[sym]['q']*p*(1-COMM);del hld[sym]

        total2=cash+sum(h['q']*price(c,rd) for c,h in hld.items() if price(c,rd))
        n=len(sel)
        if n==0:
            for sym in list(hld.keys()):
                p=price(sym,rd) or hld[sym]['p'];cash+=hld[sym]['q']*p*(1-COMM);del hld[sym]
            eq.append(cash);continue

        per=total2/n*0.90
        for sym in sel:
            p=price(sym,rd)
            if p is None or p<0.01:continue
            tq=int(per/p/100)*100
            if tq<100:continue
            if sym in hld:
                diff=tq-hld[sym]['q']
                if abs(diff)>=100:
                    cst=abs(diff)*p
                    if diff>0 and cash>=cst*(1+COMM):cash-=cst*(1+COMM);hld[sym]['q']=tq
                    elif diff<0:cash+=cst*(1-COMM);hld[sym]['q']=tq
            else:
                cst=tq*p
                if cash>=cst*(1+COMM):cash-=cst*(1+COMM);hld[sym]={'q':tq,'p':p}
        eq.append(cash+sum(h['q']*price(c,rd) for c,h in hld.items() if price(c,rd)))

    eq=np.array(eq)
    ny=(int(end[:4])-int(start[:4]))+(int(end[4:6])-int(start[4:6]))/12.0
    cagr=((eq[-1]/eq[0])**(1/ny)-1)*100 if ny>0 and eq[0]>0 else 0
    mr=np.diff(eq)/(eq[:-1]+1e-10)
    sh=np.mean(mr)/(np.std(mr)+1e-10)*np.sqrt(12) if len(mr)>1 else 0
    pk2=eq[0];md=0.0
    for v in eq:
        if v>pk2:pk2=v
        d=(pk2-v)/pk2*100
        if d>md:md=d
    total_ret=(eq[-1]/eq[0]-1)*100
    return eq,cagr,sh,md,total_ret,ny

PERIODS=[
    ("Train(2015-21)","20150101","20211231"),
    ("Test(2022-25)","20220101","20251231"),
    ("2022","20220101","20221231"),
    ("2023","20230101","20231231"),
    ("2024","20240101","20241231"),
    ("2025","20250101","20251231"),
]

for label,ps,pe in PERIODS:
    eq,cagr,sh,md,tr,ny=run(label,ps,pe)
    print(f"{label:<20s} | CAGR={cagr:+5.1f}% | Sharpe={sh:6.3f} | MaxDD={md:5.1f}% | Total={tr:+6.1f}% | {ny:.1f}y")
