"""v4: Equity score=0 unless 120MA slope rising."""
import warnings; warnings.filterwarnings("ignore")
from _funcs import load, metrics, year_bt, backtest, bench, P, T0, T1

if __name__=="__main__":
    data = load()
    a = {k:v for k,v in P.items() if k!="vt"}
    bts = [("v2 Baseline", backtest(data,T0,T1,**a,vt=0.18)),
           ("v3 120MA",    backtest(data,T0,T1,**P,ef="above_120")),
           ("v4 Rising MA",backtest(data,T0,T1,**P,ef="rising"))]

    print("v4: Rising MA Filter\n")
    print(f"{'Strategy':20s} {'AnnRet':>8} {'MaxDD':>8} {'Sharpe':>7} {'Calmar':>7}")
    print("-"*55)
    for name, bt in bts:
        m = metrics(bt)
        y = year_bt(bt, 2022)
        yrs = " | ".join(f"{yr}:{year_bt(bt,yr)['return_']:.1%}" for yr in [2022,2023,2024,2025])
        print(f"{name:20s} {m['annual_return']:>7.2%} {m['max_drawdown']:>7.2%} "
              f"{m['sharpe_ratio']:>6.2f} {m['calmar_ratio']:>6.2f}  [{yrs}]")
    print(f"\nBenchmarks:")
    for bm in ("510300","510500","159915","518880"):
        bm_ret = bench(data, pool=[bm])
        bm_cr = (bm_ret.iloc[-1] - 1) if len(bm_ret) > 0 else 0
        print(f"  {bm}: TotalRet={bm_cr:.2%}")
