import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.titlesize":9})


def main():
    v1=json.load(open("outputs/track_b/decomposition_24.json"))["per_fold"]
    v2=json.load(open("outputs/track_b/decomposition_24_detv2.json"))["per_fold"]
    C=json.load(open("outputs/track_b/capacity_vs_leakage.json"))
    keys=C["folds"]
    g=lambda d,k,p: d[k]["protocols"][p]["f1"]
    d1=np.array([g(v1,k,"P0")-g(v1,k,"P2") for k in keys])
    d2=np.array([g(v2,k,"P0")-g(v2,k,"P2") for k in keys])
    ru=lambda x,s=False: (("+" if s and x>0 else "")+f"{x:.4f}").replace(".",",")

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,4.6),gridspec_kw={"width_ratios":[1.55,1]})

    # --- слева: пофолдово ---
    o=np.argsort(d1); y=np.arange(len(keys))
    ax1.hlines(y,d1[o],d2[o],color="0.75",lw=1.2,zorder=1)
    ax1.scatter(d1[o],y,s=26,color="#4a6fa5",label="компактная (17 тыс. параметров)",zorder=3)
    ax1.scatter(d2[o],y,s=26,color="#b5482f",label="усиленная (1,03 млн)",zorder=3)
    ax1.axvline(0,color="black",lw=0.8,ls="--")
    ax1.set_yticks(y); ax1.set_yticklabels([keys[i].replace("ID_","") for i in o],fontsize=7)
    ax1.set_xlabel("разрыв Δ = F1(P₀) − F1(P₂)")
    ax1.set_title(f"Пофолдово: разрыв вырос в {C['increased_in']} фолдах из {C['n_folds']}",pad=6)
    ax1.legend(fontsize=7.5,loc="lower right",framealpha=0.95)
    ax1.grid(axis="x",ls=":",lw=0.5,color="0.8"); ax1.set_axisbelow(True)
    for s in ("top","right"): ax1.spines[s].set_visible(False)

    # --- справа: куда пошёл прирост мощности ---
    p=C["per_protocol"]
    x=np.arange(2); w=0.34
    b1=ax2.bar(x-w/2,[p["P0"]["v1"],p["P2"]["v1"]],w,color="#4a6fa5",label="компактная")
    b2=ax2.bar(x+w/2,[p["P0"]["v2"],p["P2"]["v2"]],w,color="#b5482f",label="усиленная")
    for b in list(b1)+list(b2):
        ax2.text(b.get_x()+b.get_width()/2,b.get_height()+0.008,ru(b.get_height()),
                 ha="center",fontsize=7.5)
    # прирост подписывается НАД парой столбцов, чтобы не перекрывать значения на них
    for xi,(a,b,gain) in enumerate([(p["P0"]["v1"],p["P0"]["v2"],C["gain_P0"]),
                                    (p["P2"]["v1"],p["P2"]["v2"],C["gain_P2"])]):
        top=max(a,b)+0.055
        ax2.annotate("",xy=(xi-w/2,top),xytext=(xi+w/2,top),
                     arrowprops=dict(arrowstyle="<->",lw=1,color="black"))
        ax2.text(xi,top+0.012,f"прирост {ru(gain,True)}",ha="center",fontsize=8,fontweight="bold")
    ax2.set_xticks(x); ax2.set_xticklabels(["P₀ — с утечкой","P₂ — честный"])
    ax2.set_ylabel("F1-мера, среднее по фолдам"); ax2.set_ylim(0,0.68)
    share=round((C["gain_P0"]-C["gain_P2"])/C["gain_P0"]*100)
    ax2.set_title(f"Прирост мощности: {share} % достаётся утечке",pad=6)
    ax2.legend(fontsize=7.5,loc="upper right"); ax2.grid(axis="y",ls=":",lw=0.5,color="0.8")
    ax2.set_axisbelow(True)
    for s in ("top","right"): ax2.spines[s].set_visible(False)

    fig.suptitle("Рисунок 16 – Зависимость измеряемого завышения от мощности модели\n"
                 f"Δ: {ru(C['delta_v1_mean'],True)} → {ru(C['delta_v2_mean'],True)} "
                 f"(в {str(round(C['ratio'],2)).replace('.',',')} раза), прирост "
                 f"[{ru(C['gain_diff_ci'][0],True)}; {ru(C['gain_diff_ci'][1],True)}], "
                 f"Уилкоксон p = {ru(C['wilcoxon_p'])}",fontsize=9,y=1.02)
    fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(f"figures_final/fig-3-9-moshchnost.{ext}",dpi=200,bbox_inches="tight",facecolor="white")
    print("figures_final/fig-3-9-moshchnost.png")


if __name__ == "__main__":
    main()
