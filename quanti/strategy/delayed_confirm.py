"""
Delayed Confirmation Strategy (production grade).

State machine on daily CSI300: 0=defensive, 2=full stocks, 4=half stocks.
Full stack: N=5/M=40 confirm + 60MA half-tier + Sharp3pct exit + A43 decay
            + 511880 bond defensive allocation.

All parameters independently verified (run_backtest.py --verify, --check-vtmd).
"""
import numpy as np
from quanti.config import settings
from quanti.indicators import sma as _sma_indicator, adx
from quanti.strategy.base import BaseStrategy
from quanti.types import Bar, MarketData, Order, OrderSide, Portfolio, Position, Signal

DECAY_SCHEDULES = {"none": lambda m: 1.0, "A43": lambda m: 1.0 if m<=4 else (0.75 if m<=8 else 0.50)}

class DelayedConfirmStrategy(BaseStrategy):
    """All verified: confirm + 60MA half + Sharp exit + A43 decay + bond defensive."""

    name = "delayed_confirm"

    def __init__(self, confirm_days=5, cooldown_days=40, top_n=5, stop_loss_pct=-10.0,
                 min_trend_score=3, dd_exit_pct=15.0, stock_universe=None,
                 decay_schedule="A43", use_sharp_exit=True, sharp_threshold=-0.03,
                 bond_etf="511880"):
        self.confirm_days=confirm_days; self.cooldown_days=cooldown_days
        self.top_n=top_n; self.stop_loss_pct=stop_loss_pct
        self.min_trend_score=min_trend_score; self.dd_exit_pct=dd_exit_pct
        self.decay_schedule=decay_schedule
        self._decay_fn=DECAY_SCHEDULES.get(decay_schedule, DECAY_SCHEDULES["A43"])
        self.use_sharp_exit=use_sharp_exit; self.sharp_threshold=sharp_threshold
        self.bond_etf=bond_etf

        self._hwm={}; self._max_equity=0.0; self._dd_exit_active=False; self._dd_exit_days=0
        self._market_state={}; self._csi_ret5={}; self._csi_bar_index={}
        self._csi300_bars_loaded=False; self._sharp_cd=-1; self._sharp_fired_recent=False
        self._last_rebalance_month=""
        self._months_in_cycle=0; self._prev_entry_state=-1; self._genuine_prev=-1
        self._stock_universe=set(stock_universe) if stock_universe else None

    def generate_signals(self, md: MarketData) -> list[Signal]:
        sigs=[]
        if not md.bars: return sigs
        if not self._csi300_bars_loaded: self._build_market_state(md)
        td=md.timestamp.strftime("%Y%m%d"); mst=self._market_state.get(td,0)
        if td[:6]==self._last_rebalance_month: return sigs
        self._last_rebalance_month=td[:6]
        ci=self._csi_bar_index.get(td,-1)
        if ci>=0 and ci>=self._sharp_cd and self._sharp_cd>=0: self._sharp_cd=-1
        if self.use_sharp_exit and ci>=5:
            r5=self._csi_ret5.get(td,0)
            if not np.isnan(r5) and r5<self.sharp_threshold:
                self._sharp_fired_recent=True; self._sharp_cd=ci+self.cooldown_days
                sigs.append(Signal(symbol=self.bond_etf,side=OrderSide.BUY,strength=1.0,
                    reason=f"Sharp exit: 5d={r5*100:.1f}%"))
                return sigs
        emst=0 if (ci>=0 and ci<self._sharp_cd) else mst
        inv=emst in (2,4); forced=(emst==0 and mst in (2,4))
        if inv:
            if self._genuine_prev not in (2,4): self._months_in_cycle=1
            else: self._months_in_cycle+=1
            self._genuine_prev=emst
        elif forced: pass
        else: self._months_in_cycle=0; self._genuine_prev=emst
        self._prev_entry_state=emst
        bm=1.0 if emst==2 else (0.5 if emst==4 else 0.0)
        sz=bm*self._decay_fn(self._months_in_cycle)
        if sz>0.02:
            tr=self._score_stocks(md)
            if not tr: return sigs
            tr.sort(key=lambda x:x[1],reverse=True)
            for sym,sc,_ in tr[:self.top_n]:
                sigs.append(Signal(symbol=sym,side=OrderSide.BUY,
                    strength=round(min(sc/100.0*sz,1.0),4),
                    reason=f"ENTRY st={emst} sc={sc:.0f} dec={self._decay_fn(self._months_in_cycle):.0%} mo={self._months_in_cycle}"))
            sigs.append(Signal(symbol=self.bond_etf,side=OrderSide.SELL,strength=1.0,reason="ENTRY:sell bonds"))
        else:
            sigs.append(Signal(symbol=self.bond_etf,side=OrderSide.BUY,strength=1.0,reason=f"DEF st={emst}"))
        return sigs

    def size_positions(self,signals,capital,pf,market_data=None):
        md=market_data; odrs=[]
        buys=[s for s in signals if s.side==OrderSide.BUY]
        sells=[s for s in signals if s.side==OrderSide.SELL]
        td_str=""
        if md: td_str=md.timestamp.strftime("%Y%m%d"); mst=self._market_state.get(td_str,0)
        else: mst=0
        ci=self._csi_bar_index.get(td_str,-1) if td_str else -1
        emst=0 if (ci>=0 and ci<self._sharp_cd) else mst
        bm=1.0 if emst==2 else (0.5 if emst==4 else 0.0)
        dm=self._decay_fn(self._months_in_cycle) if self._months_in_cycle>0 else 1.0; sz=bm*dm
        sb=[s for s in buys if s.symbol!=self.bond_etf]
        bb=[s for s in buys if s.symbol==self.bond_etf]
        bs=[s for s in sells if s.symbol==self.bond_etf]
        ss={s.symbol for s in sb}
        sharp_sold = set()
        if self._sharp_fired_recent:
            for sym,pos in pf.positions.items():
                if sym==self.bond_etf or pos.quantity<=0: continue
                odrs.append(Order(symbol=sym,side=OrderSide.SELL,quantity=pos.quantity,price=None,order_type="market",signal_ref=f"Sharp exit:{sym}"))
                sharp_sold.add(sym)
                self._hwm.pop(sym,None)  # prevent stale HWM on re-entry
            self._sharp_fired_recent=False
        for sym,pos in pf.positions.items():
            if sym==self.bond_etf or pos.quantity<=0 or sym in sharp_sold: continue
            sell=False; reason=""
            # In defense state, ss (selected_syms) is empty → all stocks get sold here.
            # This is the *only* place defense liquidation happens — no explicit SELL signals.
            if sym not in ss: sell=True; reason=f"Rotated(st={emst})"
            elif sym in self._hwm and pos.current_price>0:
                loss=(pos.current_price/self._hwm[sym]-1)*100
                if loss<self.stop_loss_pct: sell=True; reason=f"Stop:{loss:.1f}%"
            if sell:
                odrs.append(Order(symbol=sym,side=OrderSide.SELL,quantity=pos.quantity,price=None,order_type="market",signal_ref=reason))
                self._hwm.pop(sym,None)  # prevent stale HWM on re-entry
        # Sell bonds FIRST so cash is available for stock BUYs below.
        # Otherwise when entering from defense the engine processes BUYs before
        # the bond SELL, checks cash, and skips the stock entry.
        bp_pos=pf.positions.get(self.bond_etf); bu_units=bp_pos.quantity if bp_pos else 0
        for sig in bs:
            if bu_units>0: odrs.append(Order(symbol=self.bond_etf,side=OrderSide.SELL,quantity=bu_units,price=None,order_type="market",signal_ref=sig.reason))

        ne=[s for s in sb if s.symbol not in pf.positions]
        if ne and sz>0.02:
            tc=capital+sum(p.quantity*p.current_price for p in pf.positions.values())
            ps=tc*sz/max(len(ss),1)*0.92
            for sig in ne:
                px=self._get_price(sig.symbol,pf,md)
                if px and px>0.01:
                    q=int(ps/px/100)*100
                    if q>=100: odrs.append(Order(symbol=sig.symbol,side=OrderSide.BUY,quantity=q,price=px,order_type="limit",signal_ref=sig.reason))
        for sig in bb:
            bpx=self._get_price(self.bond_etf,pf,md)
            if bpx and bpx>0.01 and capital>1000:
                q=int(capital*0.99/bpx)
                if q>0: odrs.append(Order(symbol=self.bond_etf,side=OrderSide.BUY,quantity=q,price=bpx,order_type="limit",signal_ref=sig.reason))
        return odrs

    def risk_check(self,odrs,pf,market_data=None,risk_checker=None):
        md=market_data; appr=[]
        tv=pf.cash+sum(p.quantity*p.current_price for p in pf.positions.values())
        if tv>self._max_equity: self._max_equity=tv; self._dd_exit_active=False
        if self._max_equity>0 and self.dd_exit_pct>0:
            dd=(self._max_equity-tv)/self._max_equity*100
            if dd>self.dd_exit_pct:
                self._dd_exit_active=True; self._dd_exit_days=0
                self._months_in_cycle=0; self._prev_entry_state=-1; self._genuine_prev=-1
            elif self._dd_exit_active:
                self._dd_exit_days+=1
                if tv/self._max_equity>0.92 or (self._dd_exit_days>60 and dd<=self.dd_exit_pct):
                    self._dd_exit_active=False; self._max_equity=tv; self._dd_exit_days=0
        if self._dd_exit_active:
            for sym,pos in pf.positions.items():
                if pos.quantity>0: appr.append(Order(symbol=sym,side=OrderSide.SELL,quantity=pos.quantity,price=None,order_type="market",signal_ref=f"DD breaker:-{dd:.1f}%"))
            return appr
        for sym,pos in pf.positions.items():
            if sym==self.bond_etf: continue
            if sym not in self._hwm or pos.current_price>self._hwm[sym]: self._hwm[sym]=pos.current_price
            if pos.current_price>0 and self._hwm.get(sym,0)>0:
                lp=(pos.current_price/self._hwm[sym]-1)*100
                if lp<self.stop_loss_pct: appr.append(Order(symbol=sym,side=OrderSide.SELL,quantity=pos.quantity,price=None,order_type="market",signal_ref=f"HWM:{lp:.1f}%"))
        for o in odrs:
            cost=o.quantity*(o.price or 0); tc=getattr(settings,"TRADING_CAPITAL",90000)
            if cost>tc*0.25: continue
            appr.append(o)
        return appr

    def get_decay_info(self): return {"schedule":self.decay_schedule,"months_in_cycle":self._months_in_cycle,"current_multiplier":self._decay_fn(self._months_in_cycle),"prev_entry_state":self._prev_entry_state}
    def get_sharp_info(self): return {"use_sharp_exit":self.use_sharp_exit,"sharp_threshold":self.sharp_threshold,"sharp_cd":self._sharp_cd,"sharp_fired_recent":self._sharp_fired_recent}

    def _build_market_state(self,md):
        csi300=None
        for sym in ("510300","159919"):
            if md.index_bars and sym in md.index_bars: csi300=md.index_bars[sym]; break
        if not csi300:
            for sym in ("510300","159919"):
                if sym in md.bars: csi300=md.bars[sym]; break
        if not csi300 or len(csi300)<200: self._csi300_bars_loaded=True; return
        closes=np.array([b.close for b in csi300],dtype=np.float64)
        dates=[b.datetime.strftime("%Y%m%d") for b in csi300]; n=len(closes)
        for i in range(5,n): self._csi_ret5[dates[i]]=closes[i]/closes[i-5]-1.0
        for i,d in enumerate(dates): self._csi_bar_index[d]=i
        ma120=self._sma(closes,120); ma60=self._sma(closes,60)
        if ma120 is None or ma60 is None: self._csi300_bars_loaded=True; return
        a120=(closes>ma120)&(~np.isnan(ma120)); a60=(closes>ma60)&(~np.isnan(ma60))
        N=self.confirm_days; M=self.cooldown_days; cd=cf=fd=-1
        for i in range(121,n):
            if i<=cd: self._market_state[dates[i]]=3; continue
            if fd>=0:
                if a120[i]: self._market_state[dates[i]]=2; continue
                else: fd=-1
            if cf>=0:
                if not a120[i]: cd=i+M-1; self._market_state[dates[i]]=3; cf=-1; continue
                if i-cf+1==N: self._market_state[dates[i]]=2; fd=i; cf=-1
                else: self._market_state[dates[i]]=1; continue
            if a120[i] and not a120[i-1]: cf=i; self._market_state[dates[i]]=1
            elif a60[i]: self._market_state[dates[i]]=4
            else: self._market_state[dates[i]]=0
        self._csi300_bars_loaded=True

    @staticmethod
    def _sma(arr,p):
        if len(arr)<p: return None
        return _sma_indicator(arr,p)
    @staticmethod
    def _adx(h,l,c,p=14): return adx(h,l,c,p)

    def _score_stocks(self,md):
        tr=[]
        for sym,bars in md.bars.items():
            if sym==self.bond_etf: continue
            if self._stock_universe and sym not in self._stock_universe: continue
            if len(bars)<200: continue
            it,cc=self._is_stock_trending(bars)
            if not it or cc<self.min_trend_score: continue
            sc=self._trend_strength_score(bars)
            if sc>0: tr.append((sym,sc,cc))
        return tr

    def _is_stock_trending(self,bars):
        if len(bars)<200: return False,0
        closes=np.array([b.close for b in bars],dtype=np.float64)
        highs=np.array([b.high for b in bars],dtype=np.float64)
        lows=np.array([b.low for b in bars],dtype=np.float64)
        vols=np.array([b.volume for b in bars],dtype=np.float64)
        cnt=0
        m120=self._sma(closes,120)
        above = m120 is not None and not np.isnan(m120[-1]) and closes[-1]>m120[-1]
        if above: cnt+=1
        rh=np.max(highs[-20:]); ph=np.max(highs[-60:-20])
        rl=np.min(lows[-20:]); pl=np.min(lows[-60:-20])
        if rh>ph and rl>pl: cnt+=1
        m20=self._sma(closes,20); m60=self._sma(closes,60)
        if (m20 is not None and m60 is not None and m120 is not None and
            not np.isnan(m20[-1]) and not np.isnan(m60[-1]) and
            not np.isnan(m120[-1]) and m20[-1]>m60[-1]>m120[-1]): cnt+=1
        ax=self._adx(highs,lows,closes,14)
        adx_ok = ax is not None and not np.isnan(ax[-1]) and ax[-1]>25
        if adx_ok: cnt+=1
        v20=np.mean(vols[-21:-1])
        if v20>0 and vols[-1]>v20*1.2: cnt+=1
        return (above and adx_ok and cnt>=self.min_trend_score),cnt

    def _trend_strength_score(self,bars):
        closes=np.array([b.close for b in bars],dtype=np.float64)
        if len(closes)<130: return 0.0
        if closes[-63]<1e-6 or closes[-126]<1e-6: ms=0.0
        else:
            r3=closes[-1]/closes[-63]-1; r6=closes[-1]/closes[-126]-1
            m3=min(max(r3/0.5,0),1) if r3>0 else 0; m6=min(max(r6/0.8,0),1) if r6>0 else 0
            ms=(0.5*m3+0.5*m6)*100
        if len(closes)>=61:
            w=closes[-61:]; dr=np.diff(w)/(w[:-1]+1e-10); vs=max(0,(1-min(np.nanstd(dr)/0.04,1)))*100
        else: vs=50.0
        return 0.6*ms+0.4*vs

    @staticmethod
    def _get_price(symbol,pf,md):
        pos=pf.positions.get(symbol)
        if pos and pos.current_price>0: return pos.current_price
        if md and symbol in md.bars:
            bars=md.bars[symbol]
            if bars: return bars[-1].close
        return None
