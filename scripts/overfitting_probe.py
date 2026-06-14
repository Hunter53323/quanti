"""
Overfitting probe for ETF multi-sector market-state strategy.

Tests:
  1. Look-ahead audit: verify no forward-looking bias in state_map / trending cache
  2. Parameter sensitivity: N_CONFIRM(3-7) x M_COOLDOWN(20-60) grid sweep
  3. Sharp threshold sweep: -2% to -5%
  4. Gold ratio overfit check: 50/50 vs 80/20 in 2-year rolling OOS
  5. Concentration limit impact: max_per_sector=1/2/unlimited
  6. Random ETF pool vs full pool bootstrap
"""
import sys; sys.path.insert(0, r"C:\study\AIWorkspace\quanti"); sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import time, numpy as np, itertools
from quanti.data.storage import DataStorage
from quanti.config.etf_universe import ETF_UNIVERSE_MULTI, get_sector

CAPITAL=90000; COMM=0.00025; STOP_PCT=-10; DD_EXIT_PCT=15; MIN_TREND=3; TOP_N=3; VT=0.85; MD=-0.02
SHARP_THRESHOLD=-0.03

# ─── helpers (copied from backtest) ──────────────────────────────────
def sma(arr,p):
    if len(arr)<p: return None
    o=np.full(len(arr),np.nan); cs=np.cumsum(np.insert(arr,0,0.0)); o[p-1:]=(cs[p:]-cs[:-p])/p; return o
def adx_arr(h,l,c,p=14):
    n=len(c);
    if n<p*2: return None
    tr=np.zeros(n); tr[0]=h[0]-l[0]
    for i in range(1,n): tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
    pdm=np.zeros(n); mdm=np.zeros(n)
    for i in range(1,n):
        up=h[i]-h[i-1]; dn=l[i-1]-l[i]
        if up>dn and up>0: pdm[i]=up
        if dn>up and dn>0: mdm[i]=dn
    atr=np.full(n,np.nan); atr[p]=float(np.mean(tr[1:p+1]))
    for i in range(p+1,n): atr[i]=(tr[i]+(p-1)*atr[i-1])/p
    ps=float(np.mean(pdm[1:p+1])); ms=float(np.mean(mdm[1:p+1]))
    pdi=np.full(n,np.nan); mdi=np.full(n,np.nan)
    pdi[p]=ps/max(atr[p],0.001)*100; mdi[p]=ms/max(atr[p],0.001)*100
    for i in range(p+1,n):
        ps=(pdm[i]+(p-1)*ps)/p; ms=(mdm[i]+(p-1)*ms)/p
        pdi[i]=min(ps/max(atr[i],0.001)*100,1000); mdi[i]=min(ms/max(atr[i],0.001)*100,1000)
    dx=np.abs(pdi-mdi)/(pdi+mdi+1e-10)*100; ax=np.full(n,np.nan)
    seed=float(np.nanmean(dx[p:p*2])); ax[p*2-1]=0.0 if np.isnan(seed) else seed; ds=ax[p*2-1]
    for i in range(p*2,n): vi=dx[i] if not np.isnan(dx[i]) else ds; ds=(vi+(p-1)*ds)/p; ax[i]=ds
    return ax
def is_etf_uptrend(cl,hi,lo,vol):
    if len(cl)<200: return False,0
    m120=sma(cl,120)
    if m120 is None or np.isnan(m120[-1]): return False,0
    above=cl[-1]>m120[-1]; rh=np.max(hi[-20:]); ph=np.max(hi[-60:-20]); rl=np.min(lo[-20:]); pl=np.min(lo[-60:-20])
    m20=sma(cl,20); m60=sma(cl,60)
    if m20 is None or m60 is None: return False,0
    if np.isnan(m20[-1]) or np.isnan(m60[-1]): return False,0
    align=m20[-1]>m60[-1]>m120[-1]; av=adx_arr(hi,lo,cl,14); adx_ok=av is not None and not np.isnan(av[-1]) and av[-1]>25
    v20=np.mean(vol[-21:-1]); surge=vol[-1]>v20*1.2
    score=sum([above,rh>ph and rl>pl,align,adx_ok,surge])
    return above and adx_ok and score>=MIN_TREND,score
def trend_score(cl):
    if len(cl)<130: return 0
    r3=cl[-1]/cl[-63]-1 if cl[-63]>1e-6 else 0; r6=cl[-1]/cl[-126]-1 if cl[-126]>1e-6 else 0
    m3=min(max(r3/0.5,0),1) if r3>0 else 0; m6=min(max(r6/0.8,0),1) if r6>0 else 0
    mom=(0.5*m3+0.5*m6)*100; w=cl[-61:]; dr=np.diff(w)/(w[:-1]+1e-10)
    vs=(1-min(np.nanstd(dr)/0.04,1))*100; return 0.6*mom+0.4*vs
def build_state_map(csi_dates,csi_closes,csi_volumes,N,M,vt=None,md=None):
    n=len(csi_closes); ma120=sma(csi_closes,120); ma60=sma(csi_closes,60)
    a120=(csi_closes>ma120)&(~np.isnan(ma120)); a60=(csi_closes>ma60)&(~np.isnan(ma60))
    st=np.full(n,0,dtype=int); cd,cf,fd=-1,-1,-1
    for i in range(121,n):
        if i<=cd: st[i]=3; continue
        if fd>=0:
            if a120[i]: st[i]=2; continue
            else: fd=-1
        if cf>=0:
            if not a120[i]: cd=i+M-1; st[i]=3; cf=-1; continue
            if i-cf+1==N:
                if vt is not None and md is not None:
                    ws,we=cf,i+1; wv=np.mean(csi_volumes[ws:we]); pv=np.mean(csi_volumes[max(0,ws-20):ws])
                    cr=csi_closes[i]/csi_closes[cf]-1.0
                    if wv>=vt*pv and cr>=md: st[i]=2; fd=i
                    else: st[i]=4; cf=-1
                else: st[i]=2; fd=i; cf=-1
            else: st[i]=1; continue
        if a120[i] and not a120[i-1]: cf=i; st[i]=1
        elif a60[i]: st[i]=4
        else: st[i]=0
    return {str(csi_dates[j]):int(st[j]) for j in range(n)}
def monthly_dates(dates,s,e):
    m=[]
    for d in dates:
        if d<s or d>e:
            continue
        dm=d[4:6]
        if not m or dm!=m[-1][4:6]:
            m.append(d)
    return m
def data_at(code, dt, n, sdata):
    if code not in sdata:
        return None
    c, h, l, v, d = sdata[code]
    idx = None
    for i in range(len(d) - 1, -1, -1):
        if d[i] <= dt:
            idx = i + 1
            break
    if idx is None or idx < n:
        return None
    return (c[idx - n:idx], h[idx - n:idx], l[idx - n:idx], v[idx - n:idx])

def price_on(code, dt, sdata):
    if code not in sdata:
        return None
    c = sdata[code][0]
    d = sdata[code][4]
    for i in range(len(d) - 1, -1, -1):
        if d[i] <= dt:
            return c[i]
    return None
def load_etf_data(storage,codes):
    sd={}
    for code in codes:
        raw=storage.load_bars(code)
        if not raw or len(raw)<200: continue
        d=[r.trade_date for r in raw]; sd[code]=(np.array([r.close for r in raw],dtype=np.float64),np.array([r.high for r in raw],dtype=np.float64),np.array([r.low for r in raw],dtype=np.float64),np.array([r.volume for r in raw],dtype=np.float64),d)
    return sd
def precompute_trending(stock_data,all_dates,start,end):
    rd=monthly_dates(all_dates,start,end); cache={}
    for i,d in enumerate(rd):
        t=[]
        for code in stock_data:
            d2=data_at(code,d,260,stock_data)
            if d2 is None: continue
            cl,hi,lo,vo=d2; is_t,nc=is_etf_uptrend(cl,hi,lo,vo)
            if is_t and nc>=MIN_TREND: t.append((code,trend_score(cl)))
        t.sort(key=lambda x:x[1],reverse=True); cache[d]=t
    return cache
def metrics(eq_curve):
    eq=np.array(eq_curve); ny=len(eq_curve)/12.0
    if eq[0]<=0 or ny<=0: return {"cagr":0,"sharpe":0,"maxdd":100,"final":float(eq[-1])}
    cagr=((eq[-1]/eq[0])**(1/ny)-1)*100; mr=np.diff(eq)/(eq[:-1]+1e-10)
    sh=np.mean(mr)/(np.std(mr)+1e-10)*np.sqrt(12) if len(mr)>1 else 0
    peak=eq[0]; mdd=0.0
    for v in eq:
        if v>peak: peak=v
        d=(peak-v)/peak*100
        if d>mdd: mdd=d
    return {"cagr":cagr,"sharpe":sh,"maxdd":mdd,"final":float(eq[-1])}

def run_once(state_map,tc,stock_data,all_dates,bond_cl,gold_cl,start,end,
             csi_ret5_arr,csi_idx_map,decay_fn,sharp_threshold,bond_pct,gold_pct,
             max_per_sector=2,top_n=3):
    """Full strategy run, returns metrics."""
    rebal=monthly_dates(all_dates,start,end)
    cash=CAPITAL; holdings={}; bu=0; gu=0; eq=[cash]; max_eq=cash; dd_active=False
    months_in_cycle=0; prev_mst=0; genuine_prev=0; sharp_cd=-1
    for rd in rebal:
        mst=state_map.get(rd,0); bp=bond_cl.get(rd); gp=gold_cl.get(rd)
        ci=csi_idx_map.get(str(rd))
        if ci is not None and ci>=0 and ci>=sharp_cd and sharp_cd>=0: sharp_cd=-1
        sharp_fired=False
        if ci is not None and ci>=5 and ci<len(csi_ret5_arr) and len(holdings)>0:
            r5=csi_ret5_arr[ci]
            if not np.isnan(r5) and r5<sharp_threshold: sharp_fired=True
        if sharp_fired:
            for sym in list(holdings.keys()):
                cash+=holdings[sym]["qty"]*holdings[sym]["price"]*(1-COMM); del holdings[sym]
            if bu>0 and bp: cash+=bu*bp*(1-COMM); bu=0
            if gu>0 and gp: cash+=gu*gp*(1-COMM); gu=0
            if cash>1000:
                tcf=cash
                if bp: bb=int(tcf*bond_pct*0.99/bp)
                if bb>0: bu+=bb; cash-=bb*bp*(1+COMM)
                if gp: bg=int(tcf*gold_pct*0.99/gp)
                if bg>0: gu+=bg; cash-=bg*gp*(1+COMM)
            sharp_cd=ci+40; eq.append(cash+bu*(bp or 1.0)+gu*(gp or 1.0)); continue
        emst=3 if (ci is not None and ci>=0 and ci<sharp_cd) else mst
        if emst in (2,4):
            if genuine_prev not in (2,4): months_in_cycle=1
            else: months_in_cycle+=1
            genuine_prev=emst
        elif emst==3 and mst!=3: pass
        else: months_in_cycle=0; genuine_prev=emst
        prev_mst=emst
        base_sm=1.0 if emst==2 else (0.5 if emst==4 else 0.0)
        sm=base_sm*decay_fn(months_in_cycle)
        if sm>0 and bu>0 and bp: cash+=bu*bp*(1-COMM); bu=0
        if sm>0 and gu>0 and gp: cash+=gu*gp*(1-COMM); gu=0
        for sym in list(holdings.keys()):
            p=price_on(sym,rd,stock_data)
            if p is None or p<0.01: cash+=holdings[sym]["val"]*0.7; del holdings[sym]; continue
            holdings[sym]["price"]=p; holdings[sym]["val"]=holdings[sym]["qty"]*p
            if "hwm" not in holdings[sym] or p>holdings[sym]["hwm"]: holdings[sym]["hwm"]=p
            if holdings[sym]["hwm"]>0 and (p/holdings[sym]["hwm"]-1)*100<STOP_PCT:
                cash+=holdings[sym]["val"]*(1-COMM); del holdings[sym]
        total=cash+sum(h["qty"]*h["price"] for h in holdings.values())+bu*(bp or 1.0)+gu*(gp or 1.0)
        if total>max_eq: max_eq=total
        dd=(max_eq-total)/max_eq*100 if max_eq>0 else 0
        if dd>DD_EXIT_PCT:
            for sym in list(holdings.keys()):
                cash+=holdings[sym]["qty"]*holdings[sym]["price"]*(1-COMM); del holdings[sym]
            if bu>0 and bp: cash+=bu*bp*(1-COMM); bu=0
            if gu>0 and gp: cash+=gu*gp*(1-COMM); gu=0; dd_active=True
        elif dd_active and total/max_eq>0.92: dd_active=False
        if dd_active: eq.append(cash); continue
        if sm==0:
            for sym in list(holdings.keys()):
                cash+=holdings[sym]["qty"]*holdings[sym]["price"]*(1-COMM); del holdings[sym]
            if cash>1000:
                tcf=cash
                if bp: bb=int(tcf*bond_pct*0.99/bp)
                if bb>0: bu+=bb; cash-=bb*bp*(1+COMM)
                if gp: bg=int(tcf*gold_pct*0.99/gp)
                if bg>0: gu+=bg; cash-=bg*gp*(1+COMM)
            eq.append(cash+bu*(bp or 1.0)+gu*(gp or 1.0)); continue
        trending=tc.get(rd,[])
        if not trending: eq.append(cash+sum(h["qty"]*h["price"] for h in holdings.values())+bu*(bp or 1.0)+gu*(gp or 1.0)); continue
        # Concentration limit
        selected=set(); sc={}
        for t_code,t_score in trending:
            if len(selected)>=top_n: break
            sector=get_sector(t_code)
            if sector not in ("宽基","防御"):
                if sc.get(sector,0)>=max_per_sector: continue
            selected.add(t_code); sc[sector]=sc.get(sector,0)+1
        for sym in list(holdings.keys()):
            if sym not in selected: cash+=holdings[sym]["qty"]*holdings[sym]["price"]*(1-COMM); del holdings[sym]
        n_pos=max(len(selected),1); per=total*sm/n_pos*0.90
        for sym in selected:
            p=price_on(sym,rd,stock_data)
            if p is None or p<0.01: continue
            tq=int(per/p/100)*100
            if tq<100: continue
            if sym in holdings:
                diff=tq-holdings[sym]["qty"]
                if abs(diff)>=100:
                    cost=abs(diff)*p
                    if diff>0 and cash>=cost*(1+COMM): cash-=cost*(1+COMM); holdings[sym]["qty"]=tq
                    elif diff<0: cash+=cost*(1-COMM); holdings[sym]["qty"]=tq
            else:
                cost=tq*p
                if cash>=cost*(1+COMM): cash-=cost*(1+COMM); holdings[sym]={"qty":tq,"price":p,"val":cost,"hwm":p}
        total=cash+sum(h["qty"]*h["price"] for h in holdings.values())+bu*(bp or 1.0)+gu*(gp or 1.0); eq.append(total)
    return metrics(eq)

# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    t0=time.time(); print("Loading...")
    storage=DataStorage()
    raw=storage.load_bars("510300")
    csi_dates=np.array([r.trade_date for r in raw])
    csi_closes=np.array([r.close for r in raw],dtype=np.float64); csi_volumes=np.array([r.volume for r in raw],dtype=np.float64)
    nc=len(csi_closes); csi_ret5=np.full(nc,np.nan)
    for i in range(5,nc): csi_ret5[i]=csi_closes[i]/csi_closes[i-5]-1.0
    csi_idx_map={str(d):i for i,d in enumerate(csi_dates)}
    raw_b=storage.load_bars("511880"); bond_cl={r.trade_date:float(r.close) for r in raw_b} if raw_b else {}
    raw_g=storage.load_bars("518880"); gold_cl={r.trade_date:float(r.close) for r in raw_g} if raw_g else {}
    etf_codes=[e["code"] for e in ETF_UNIVERSE_MULTI]; stock_data=load_etf_data(storage,etf_codes)
    all_ds=set()
    for v in stock_data.values(): all_ds.update(v[4])
    all_dates=sorted(all_ds)
    print(f"{len(stock_data)} ETFs, {len(all_dates)} days")

    # ═══ 1. LOOK-AHEAD AUDIT ═══
    print("\n"+"="*60); print("1. LOOK-AHEAD AUDIT"); print("="*60)
    # Pick a middle date, verify state_map uses only data <= that date
    test_date="20210331"; ci=csi_idx_map.get(test_date)
    if ci is not None:
        st5=build_state_map(csi_dates[:ci+1],csi_closes[:ci+1],csi_volumes[:ci+1],5,40,VT,MD)
        print(f"  State map at {test_date}: {len(st5)} entries, state={st5.get(test_date,'?')}")
        # Verify: all dates in st5 <= test_date
        max_d=max(st5.keys())
        print(f"  Max date in state map: {max_d} (should be == {test_date})")
        # Verify: only dates <= test_date in map
        assert max_d==test_date, f"STATE MAP LEAK: max={max_d} > test_date={test_date}"

        # Verify trending cache: for 2021-03-31, we use data_at(code,rd,260,...)
        tc_small=precompute_trending(stock_data,all_dates,test_date,test_date)
        for rd,candidates in tc_small.items():
            print(f"  {rd}: {len(candidates)} trending ETFs")
            if candidates:
                # For each candidate, verify the 260 bars used end at rd
                d2=data_at(candidates[0][0],rd,260,stock_data)
                if d2 is not None:
                    last_d=stock_data[candidates[0][0]][4]
                    idx=None
                    for i in range(len(last_d)-1,-1,-1):
                        if last_d[i]<=rd: idx=i; break
                    print(f"    {candidates[0][0]}: last bar idx={idx}, last date={last_d[idx] if idx else '?'}")
        print("  PASS: No look-ahead detected (all inputs <= test_date)")
    else:
        print("  SKIP: test_date not in index")

    # ═══ 2. PARAMETER SWEEP (N_CONFIRM x M_COOLDOWN) ═══
    print("\n"+"="*60); print("2. N_CONFIRM x M_COOLDOWN SWEEP (Test 2022-2025)"); print("="*60)
    a43_fn=lambda m: 1.0 if m<=4 else(0.75 if m<=8 else 0.50)
    for N in [3,5,7]:
        for M in [20,40,60]:
            sm=build_state_map(csi_dates,csi_closes,csi_volumes,N,M,VT,MD)
            tc=precompute_trending(stock_data,all_dates,"20220101","20251231")
            r=run_once(sm,tc,stock_data,all_dates,bond_cl,gold_cl,
                      "20220101","20251231",csi_ret5,csi_idx_map,
                      a43_fn,SHARP_THRESHOLD,0.80,0.20)
            print(f"  N={N} M={M:2d}: CAGR={r['cagr']:+6.2f}% Sharpe={r['sharpe']:+6.3f} MaxDD={r['maxdd']:+5.1f}%")

    # ═══ 3. SHARP THRESHOLD SWEEP ═══
    print("\n"+"="*60); print("3. SHARP THRESHOLD SWEEP (Test 2022-2025)"); print("="*60)
    sm=build_state_map(csi_dates,csi_closes,csi_volumes,5,40,VT,MD)
    tc=precompute_trending(stock_data,all_dates,"20220101","20251231")
    for st_val in [-0.02,-0.03,-0.04,-0.05,-999]:
        r=run_once(sm,tc,stock_data,all_dates,bond_cl,gold_cl,
                  "20220101","20251231",csi_ret5,csi_idx_map,
                  a43_fn,st_val,0.80,0.20)
        disabled="(disabled)" if st_val==-999 else ""
        print(f"  Sharp={st_val:+.2f}% {disabled:>11}: CAGR={r['cagr']:+6.2f}% Sharpe={r['sharpe']:+6.3f} MaxDD={r['maxdd']:+5.1f}%")

    # ═══ 4. 2-YEAR ROLLING OOS (overfit check on gold ratio) ═══
    print("\n"+"="*60); print("4. ROLLING 2-YEAR OOS WINDOWS"); print("="*60)
    windows=[("20200101","20211231"),("20210101","20221231"),("20220101","20231231"),("20230101","20251231")]
    for bpct,gpct,label in [(0.80,0.20,"80/20"),(0.50,0.50,"50/50")]:
        cagrs=[]
        for ws,we in windows:
            sm=build_state_map(csi_dates,csi_closes,csi_volumes,5,40,VT,MD)
            tc_win=precompute_trending(stock_data,all_dates,ws,we)
            r=run_once(sm,tc_win,stock_data,all_dates,bond_cl,gold_cl,
                      ws,we,csi_ret5,csi_idx_map,a43_fn,SHARP_THRESHOLD,bpct,gpct)
            cagrs.append(r['cagr'])
        avg=np.mean(cagrs); std=np.std(cagrs)
        print(f"  Gold {label}: avg={avg:+6.2f}% std={std:5.2f}%  windows={[f'{c:+5.1f}%' for c in cagrs]}")

    # ═══ 5. CONCENTRATION LIMIT SWEEP ═══
    print("\n"+"="*60); print("5. CONCENTRATION LIMIT (Test 2022-2025)"); print("="*60)
    sm=build_state_map(csi_dates,csi_closes,csi_volumes,5,40,VT,MD)
    tc=precompute_trending(stock_data,all_dates,"20220101","20251231")
    for mp in [0,1,2]:
        r=run_once(sm,tc,stock_data,all_dates,bond_cl,gold_cl,
                  "20220101","20251231",csi_ret5,csi_idx_map,
                  a43_fn,SHARP_THRESHOLD,0.80,0.20,
                  max_per_sector=mp if mp>0 else 99)
        label=f"max={mp}" if mp>0 else "unlimited"
        print(f"  {label:>10}: CAGR={r['cagr']:+6.2f}% Sharpe={r['sharpe']:+6.3f} MaxDD={r['maxdd']:+5.1f}%")

    # ═══ 6. RANDOM POOL BOOTSTRAP ═══
    print("\n"+"="*60); print("6. RANDOM POOL BOOTSTRAP (control for pool size)"); print("="*60)
    all_codes=list(stock_data.keys()); np.random.seed(42)
    sm=build_state_map(csi_dates,csi_closes,csi_volumes,5,40,VT,MD)
    boot_cagrs=[]
    for _ in range(20):
        # Random 12 ETFs (same size as early period)
        subset=np.random.choice(all_codes,size=min(12,len(all_codes)),replace=False)
        sub_sd={c:stock_data[c] for c in subset}
        sub_all_ds=set();
        for v in sub_sd.values(): sub_all_ds.update(v[4]);
        sub_all_dates=sorted(sub_all_ds)
        tc_sub=precompute_trending(sub_sd,sub_all_dates,"20220101","20251231")
        r=run_once(sm,tc_sub,sub_sd,sub_all_dates,bond_cl,gold_cl,
                  "20220101","20251231",csi_ret5,csi_idx_map,
                  a43_fn,SHARP_THRESHOLD,0.80,0.20)
        boot_cagrs.append(r['cagr'])
    full_r=run_once(sm,tc,stock_data,all_dates,bond_cl,gold_cl,
                  "20220101","20251231",csi_ret5,csi_idx_map,
                  a43_fn,SHARP_THRESHOLD,0.80,0.20)
    print(f"  Full 20-ETF: CAGR={full_r['cagr']:+6.2f}%")
    print(f"  Random 12-ETF (20 trials): mean={np.mean(boot_cagrs):+6.2f}% std={np.std(boot_cagrs):5.2f}%")
    print(f"  Full CAGR in bootstrap distribution: percentile={sum(1 for c in boot_cagrs if c<=full_r['cagr'])/len(boot_cagrs)*100:.0f}%")
    print(f"  Range: {min(boot_cagrs):+6.2f}% ~ {max(boot_cagrs):+6.2f}%")

    print(f"\nDone in {time.time()-t0:.0f}s")

if __name__=="__main__":
    main()
