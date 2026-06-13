"""Rapid sweep: deployment Top-N + dual momentum + round lot."""
import sys,os,time,numpy as np
os.chdir(r"C:\study\AIWorkspace\quanti");sys.path.insert(0,".")
from quanti.data.storage import DataStorage
CAP=90000.0; CM=0.00025

st=DataStorage()
r3=st.load_bars("510300")
CD=np.array([r.trade_date for r in r3]);CC=np.array([r.close for r in r3],dtype=np.float64)
CH=np.array([r.high for r in r3],dtype=np.float64);CL=np.array([r.low for r in r3],dtype=np.float64)
rc=st.load_bars("511880")
CshD=np.array([r.trade_date for r in rc]);CshC=np.array([r.close for r in rc],dtype=np.float64)

af=sorted(st.clean_dir.glob("*.parquet"))
sc=[p.stem for p in af if len(p.stem)==6 and not p.stem.startswith(("51","58","15","56"))]
ST={};ads=set(CD)
for c in sc:
    r=st.load_bars(c)
    if not r or len(r)<200:continue
    ST[c]={"d":np.array([x.trade_date for x in r]),"c":np.array([x.close for x in r],dtype=np.float64),
           "h":np.array([x.high for x in r],dtype=np.float64),"l":np.array([x.low for x in r],dtype=np.float64),
           "v":np.array([x.volume for x in r],dtype=np.float64)}
    ads.update(x.trade_date for x in r)
AD=sorted(ads)
print(f"Stocks:{len(ST)} Dates:{len(AD)}")

def sma(a,p):
    if len(a)<p:return np.full(len(a),np.nan)
    o=np.full(len(a),np.nan);cs=np.cumsum(np.insert(a,0,0.0));o[p-1:]=(cs[p:]-cs[:-p])/p;return o
def adx(h,l,c,p=14):
    n=len(c)
    if n<p*2:return np.full(n,np.nan)
    tr=np.zeros(n);tr[0]=h[0]-l[0]
    for i in range(1,n):tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
    pd=np.zeros(n);md=np.zeros(n)
    for i in range(1,n):
        u=h[i]-h[i-1];d=l[i-1]-l[i]
        if u>d and u>0:pd[i]=u
        if d>u and d>0:md[i]=d
    at=np.full(n,np.nan);at[p]=float(np.mean(tr[1:p+1]))
    for i in range(p+1,n):at[i]=(tr[i]+(p-1)*at[i-1])/p
    ps=float(np.mean(pd[1:p+1]));ms=float(np.mean(md[1:p+1]))
    pi=np.full(n,np.nan);mi=np.full(n,np.nan)
    pi[p]=ps/max(at[p],0.001)*100;mi[p]=ms/max(at[p],0.001)*100
    for i in range(p+1,n):
        ps=(pd[i]+(p-1)*ps)/p;ms=(md[i]+(p-1)*ms)/p
        pi[i]=min(ps/max(at[i],0.001)*100,1000);mi[i]=min(ms/max(at[i],0.001)*100,1000)
    dx=np.abs(pi-mi)/(pi+mi+1e-10)*100;ax=np.full(n,np.nan)
    sd=float(np.nanmean(dx[p:p*2]));ax[p*2-1]=0.0 if np.isnan(sd) else sd;ds=ax[p*2-1]
    for i in range(p*2,n):
        vi=dx[i] if not np.isnan(dx[i]) else ds;ds=(vi+(p-1)*ds)/p;ax[i]=ds
    return ax

# State machine
stkma={}
for c,sd in ST.items():
    m20=sma(sd["c"],20);stkma[c]=(sd["d"],sd["c"]>m20)
def brd(dt):
    cn,tt=0,0
    for c,(da,aa)in stkma.items():
        idx=np.searchsorted(da,dt,side="right")-1
        if idx<19:continue
        tt+=1;cn+=1 if aa[idx] else 0
    return cn/tt*100 if tt>0 else 50

ma120=sma(CC,120);ax14=adx(CH,CL,CC,14);ba=np.array([brd(d) for d in CD])
ab=(CC>ma120)&(~np.isnan(ma120))
n=len(CC);raw_s=np.full(n,0,dtype=int)
for i in range(120,n):
    if ab[i]:
        ao=not np.isnan(ax14[i]) and ax14[i]>25;bo=not np.isnan(ba[i]) and ba[i]>45
        raw_s[i]=2 if (ao and bo) else 1
conf_s=np.full(n,0,dtype=int)
for i in range(1,n):
    rs=raw_s[i]
    if conf_s[i-1]==0:
        if i>=4 and np.all(raw_s[i-4:i+1]>=1):
            if i>=1 and np.all(raw_s[i-1:i+1]==2):conf_s[i]=2
            else:conf_s[i]=1
        else:conf_s[i]=0
    elif conf_s[i-1]==1:
        if rs==0:conf_s[i]=0
        elif i>=1 and np.all(raw_s[i-1:i+1]==2):conf_s[i]=2
        else:conf_s[i]=1
    elif conf_s[i-1]==2:
        if rs==0:conf_s[i]=0
        elif rs==1:conf_s[i]=1
        else:conf_s[i]=2
SM={CD[i]:int(conf_s[i]) for i in range(n)}

def gm(s,e):
    m=[]
    for d in AD:
        if d<s or d>e:continue
        mon=d[4:6]
        if not m or mon!=m[-1][4:6]:m.append(d)
    return m
def gp(c,dt):
    if c not in ST:return None
    sd=ST[c];idx=np.searchsorted(sd["d"],dt,side="right")-1
    return sd["c"][idx] if idx>=0 else None
def cp(dt):
    idx=np.searchsorted(CshD,dt,side="right")-1
    return CshC[idx] if idx>=0 else 100.0

# Precompute
ams=gm("20150101","20251231")
PRE={}
for c,sd in ST.items():
    PRE[c]={}
    for rd in ams:
        idx=np.searchsorted(sd["d"],rd,side="right")
        if idx<260:continue
        cs=sd["c"][idx-260:idx];hs=sd["h"][idx-260:idx];ls=sd["l"][idx-260:idx];vs=sd["v"][idx-260:idx];nn=len(cs)
        m120=sma(cs,120);abv=1.0 if(not np.isnan(m120[-1])and cs[-1]>m120[-1])else 0.0
        rh=np.max(hs[-20:]);ph=np.max(hs[-60:-20]);rl=np.min(ls[-20:]);pl=np.min(ls[-60:-20])
        hh=1.0 if(rh>ph and rl>pl)else 0.0
        m20=sma(cs,20);m60=sma(cs,60);al=0.0
        if(not np.isnan(m20[-1])and not np.isnan(m60[-1])and not np.isnan(m120[-1])and m20[-1]>m60[-1]>m120[-1]):al=1.0
        av=adx(hs,ls,cs,14)[-1];an=min(max((av-15)/35,0),1)if not np.isnan(av)else 0
        r3=cs[-1]/cs[-63]-1 if cs[-63]>1e-6 else 0;r6=cs[-1]/cs[-126]-1 if cs[-126]>1e-6 else 0
        m3=min(max(r3/0.5,0),1)if r3>0 else max(r3/0.3,-1);m6=min(max(r6/0.8,0),1)if r6>0 else max(r6/0.5,-1)
        ms=(0.5*m3+0.5*m6)*100
        w=cs[-61:];dr=np.diff(w)/(w[:-1]+1e-10);vs=(1-min(np.nanstd(dr)/0.05,1))*100
        tc=(0.35*abv+0.25*al+0.20*an+0.20*hh)*100
        ret60=(cs[-1]/cs[-60]-1)if cs[-60]>1e-6 else 0
        PRE[c][rd]=(0.60*ms+0.30*tc+0.10*vs,float(ret60))

print(f"Precomputed {len(PRE)} stocks")

# Backtest
def bt(s,e,max_pos=5,dm=True,top_n_bull=5,top_n_range=3):
    reb=gm(s,e)
    if len(reb)<6:return None
    cash=CAP;hld={};etf=0.0;eq=[CAP];me=CAP;ps=-1;sc={0:0,1:0,2:0}
    for rd in reb:
        mk=SM.get(rd,0);sc[mk]+=1;scd=(mk!=ps)
        for sy in list(hld.keys()):
            p=gp(sy,rd)
            if p is None or p<0.01:cash+=hld[sy].get("qty",0)*hld[sy].get("price",10)*0.7;del hld[sy];continue
            hld[sy]["price"]=p
        cpv=cp(rd);cv=etf*cpv
        tot=cash+cv+sum(h["qty"]*h.get("price",0)for h in hld.values())
        for sy in list(hld.keys()):
            p=hld[sy].get("price",0)
            if p<=0:continue
            hwm=hld[sy].get("hwm",p)
            if p>hwm:hld[sy]["hwm"]=p
            if hwm>0 and(p/hwm-1)*100<-10:cash+=hld[sy]["qty"]*p*(1-CM);del hld[sy]
        tot=cash+etf*cpv+sum(h["qty"]*h.get("price",0)for h in hld.values())
        if tot>me:me=tot
        dd=(me-tot)/me if me>0 else 0
        if dd>0.15:
            for sy in list(hld.keys()):cash+=hld[sy]["qty"]*hld[sy].get("price",0)*(1-CM);del hld[sy]
            cash+=etf*cpv*(1-CM);etf=0;tot=cash
        if mk==0:
            if scd or not etf:
                for sy in list(hld.keys()):cash+=hld[sy]["qty"]*hld[sy].get("price",0)*(1-CM);del hld[sy]
                cash+=etf*cpv*(1-CM);etf=0
                if cpv>0 and cash>0:etf=cash/cpv;cash=0.0
            eq.append(cash+etf*cpv);ps=mk;continue
        if etf>0:cash+=etf*cpv*(1-CM);etf=0.0
        psz=0.5 if mk==1 else 1.0;tn=top_n_range if mk==1 else top_n_bull
        scr=[(cd,PRE[cd][rd][0])for cd in PRE if rd in PRE[cd]]
        if dm:scr=[(cd,s)for cd,s in scr if PRE[cd][rd][1]>0]
        if not scr:eq.append(tot);ps=mk;continue
        scr.sort(key=lambda x:x[1],reverse=True)
        selc=[s[0]for s in scr[:tn]]
        for sy in list(hld.keys()):
            if sy not in selc:cash+=hld[sy]["qty"]*hld[sy].get("price",0)*(1-CM);del hld[sy]
        np2=max(len(selc),1);pps=tot*psz/np2*0.90
        for sy in selc:
            p=gp(sy,rd)
            if p is None or p<0.01:continue
            tq=int(pps/p/100)*100
            if tq<100 or tq*p<10000:continue
            if sy in hld:
                df=tq-hld[sy]["qty"]
                if abs(df)>=100:
                    cst=abs(df)*p
                    if df>0 and cash>=cst*(1+CM):cash-=cst*(1+CM);hld[sy]["qty"]=tq
                    elif df<0:cash+=cst*(1-CM);hld[sy]["qty"]=tq
            else:
                cst=tq*p
                if cash>=cst*(1+CM):cash-=cst*(1+CM);hld[sy]={"qty":tq,"price":p,"hwm":p}
        leftover=cash;cpv2=cp(rd)
        if cpv2>0 and leftover>0:etf=leftover/cpv2;cash=0.0
        else:cash=leftover;etf=0.0
        tot=cash+etf*cpv2+sum(h["qty"]*h.get("price",0)for h in hld.values())
        eq.append(tot);ps=mk
    ea=np.array(eq)
    if len(ea)<2:return None
    ny=len(ea)/12.0
    if ny<0.5:return None
    cagr=((ea[-1]/ea[0])**(1/ny)-1)*100
    mr=np.diff(ea)/(ea[:-1]+1e-10);sh=np.mean(mr)/(np.std(mr)+1e-10)*np.sqrt(12)if len(mr)>1 else 0
    pk=ea[0];md=0.0
    for v in ea:
        if v>pk:pk=v
        if(pk-v)/pk*100>md:md=(pk-v)/pk*100
    return{"c":cagr,"s":sh,"d":md,"tr":(ea[-1]/ea[0]-1)*100,"sc":sc}

print("="*80)
print("DEPLOYMENT CONFIG SWEEP")
print("="*80)
for tnb in[5,8]:
    for tnr in[3,5]:
        for dmf in[True,False]:
            r=bt("20220101","20251231",max_pos=tnb,dm=dmf,top_n_bull=tnb,top_n_range=tnr)
            rt=bt("20150101","20211231",max_pos=tnb,dm=dmf,top_n_bull=tnb,top_n_range=tnr)
            if r and rt:
                print(f"  BULL={tnb} RANGE={tnr} DM={dmf}: Test C={r['c']:+.1f}% D={r['d']:.1f}% Sh={r['s']:.3f} | Train C={rt['c']:+.1f}% D={rt['d']:.1f}%")

print("\nYearly (BULL=5 RANGE=3 DM=True):")
for yr in[2022,2023,2024,2025]:
    ry=bt(f"{yr}0101",f"{yr}1231",max_pos=5,dm=True,top_n_bull=5,top_n_range=3)
    if ry:print(f"  {yr}: C={ry['c']:+.1f}% D={ry['d']:.1f}% Sh={ry['s']:.3f} BEAR={ry['sc'][0]}m RANGE={ry['sc'][1]}m BULL={ry['sc'][2]}m")
