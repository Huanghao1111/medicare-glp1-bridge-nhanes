#!/usr/bin/env python3
"""
Measurement-based eligibility for the Medicare GLP-1 Bridge program
===================================================================
Reproduces all estimates and the figure reported in the research letter
"Half of Medicare Beneficiaries Eligible for the GLP-1 Bridge Program Are
Invisible to Claims Data".

Data source : NHANES 2017–March 2020 pre-pandemic cycle (public CDC files;
              downloaded automatically). This is the most recent NHANES cycle
              whose public release includes prescription-drug names (RXQ_RX),
              required to identify current GLP-1 users and antihypertensive
              medication classes.

Usage:
    python analysis.py [--data-dir data] [--out-dir outputs]

Outputs:
    figure_bridge_eligibility.png / .pdf   main two-panel figure
    headline_estimates.csv                 key weighted estimates with 95% CI
    subgroup_estimates.csv                 eligibility by subgroup
    mace_scenarios.csv                     MACE-averted scenario grid
"""

import argparse
import os
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# --------------------------------------------------------------------------
# 1. Data files (NHANES 2017–March 2020 pre-pandemic cycle, hosted under the
#    2017 public data directory)
# --------------------------------------------------------------------------
CDC_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles"
NHANES_FILES = [
    "P_DEMO",    # demographics, sampling weights, design variables
    "P_BMX",     # body measures (BMI)
    "P_BPXO",    # oscillometric blood pressure
    "P_MCQ",     # medical conditions (CVD history)
    "P_DIQ",     # diabetes questionnaire
    "P_BPQ",     # blood pressure & cholesterol questionnaire
    "P_GHB",     # glycohemoglobin (HbA1c)
    "P_BIOPRO",  # standard biochemistry (serum creatinine)
    "P_RXQ_RX",  # prescription medications (drug names)
    "P_HIQ",     # health insurance (Medicare coverage)
]

# --------------------------------------------------------------------------
# 2. Operationalization of the CMS Bridge criteria (announced May 6, 2026)
# --------------------------------------------------------------------------
# Tier A: BMI >= 35
# Tier B: BMI 30–34.9 + HFpEF / uncontrolled hypertension / CKD stage >= 3a
# Tier C: BMI 27–29.9 + prediabetes / prior MI / prior stroke
# Exclusions: type 2 diabetes, current GLP-1 use (OSA/MASH/PAD not measurable
# in NHANES -> estimates are conservative).

GLP1_DRUGS = ["SEMAGLUTIDE", "LIRAGLUTIDE", "DULAGLUTIDE", "EXENATIDE",
              "ALBIGLUTIDE", "LIXISENATIDE", "TIRZEPATIDE"]

ANTIHYPERTENSIVE_CLASSES = {
    "ACEi": ["LISINOPRIL", "ENALAPRIL", "BENAZEPRIL", "RAMIPRIL", "QUINAPRIL",
             "FOSINOPRIL", "PERINDOPRIL", "CAPTOPRIL", "MOEXIPRIL", "TRANDOLAPRIL"],
    "ARB":  ["LOSARTAN", "VALSARTAN", "IRBESARTAN", "OLMESARTAN", "CANDESARTAN",
             "TELMISARTAN", "AZILSARTAN", "EPROSARTAN"],
    "CCB":  ["AMLODIPINE", "NIFEDIPINE", "DILTIAZEM", "VERAPAMIL", "FELODIPINE",
             "ISRADIPINE", "NICARDIPINE", "NISOLDIPINE"],
    "THZ":  ["HYDROCHLOROTHIAZIDE", "CHLORTHALIDONE", "INDAPAMIDE", "METOLAZONE",
             "CHLOROTHIAZIDE"],
    "BB":   ["METOPROLOL", "ATENOLOL", "CARVEDILOL", "PROPRANOLOL", "BISOPROLOL",
             "LABETALOL", "NADOLOL", "NEBIVOLOL", "ACEBUTOLOL", "BETAXOLOL", "PINDOLOL"],
    "MRA":  ["SPIRONOLACTONE", "EPLERENONE"],
    "A1B":  ["DOXAZOSIN", "PRAZOSIN", "TERAZOSIN"],
    "CEN":  ["CLONIDINE", "METHYLDOPA", "GUANFACINE"],
    "VAS":  ["HYDRALAZINE", "MINOXIDIL"],
    "REN":  ["ALISKIREN"],
}

# Benefit-model constants (SELECT trial, NEJM 2023;389:2221-2232)
SELECT_HR = 0.80                 # hazard ratio for MACE, semaglutide vs placebo
SELECT_PLACEBO_RATE_YR = 0.0241  # placebo-arm MACE rate per year
BRIDGE_WINDOW_YR = 1.5           # 18-month demonstration window


# --------------------------------------------------------------------------
# 3. Download / load / merge
# --------------------------------------------------------------------------
def download(data_dir: str) -> None:
    os.makedirs(data_dir, exist_ok=True)
    for f in NHANES_FILES:
        dest = os.path.join(data_dir, f + ".xpt")
        if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
            continue
        url = f"{CDC_BASE}/{f}.xpt"
        print(f"  downloading {f} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(dest, "wb") as fh:
            fh.write(r.read())


def load_and_merge(data_dir: str) -> pd.DataFrame:
    def rd(name, cols):
        path = os.path.join(data_dir, name + ".xpt")
        return pd.read_sas(path, format="xport")[cols]

    demo = rd("P_DEMO", ["SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "INDFMPIR",
                         "WTMECPRP", "SDMVPSU", "SDMVSTRA"])
    parts = [
        rd("P_BMX",   ["SEQN", "BMXBMI"]),
        rd("P_BPXO",  ["SEQN", "BPXOSY1", "BPXODI1", "BPXOSY2", "BPXODI2",
                       "BPXOSY3", "BPXODI3"]),
        rd("P_MCQ",   ["SEQN", "MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]),
        rd("P_DIQ",   ["SEQN", "DIQ010", "DIQ050", "DIQ070", "DIQ160"]),
        rd("P_BPQ",   ["SEQN", "BPQ020", "BPQ040A"]),
        rd("P_GHB",   ["SEQN", "LBXGH"]),
        rd("P_BIOPRO", ["SEQN", "LBXSCR"]),
        rd("P_HIQ",   ["SEQN", "HIQ032B"]),
    ]
    df = demo
    for p in parts:
        df = df.merge(p, on="SEQN", how="left")

    # Prescription drugs: current GLP-1 use + count of antihypertensive classes
    rx = rd("P_RXQ_RX", ["SEQN", "RXDDRUG"]).copy()
    rx["drug"] = rx["RXDDRUG"].astype(str).str.upper()
    glp1_users = set(rx.loc[rx["drug"].str.contains("|".join(GLP1_DRUGS), na=False), "SEQN"])

    def n_classes(drugs: pd.Series) -> int:
        s = " | ".join(drugs)
        return sum(any(k in s for k in kws) for kws in ANTIHYPERTENSIVE_CLASSES.values())

    df["n_antihtn"] = (rx.groupby("SEQN")["drug"].apply(n_classes)
                         .reindex(df["SEQN"]).fillna(0).to_numpy())
    df["on_glp1"] = df["SEQN"].isin(glp1_users)
    return df


def egfr_ckd_epi_2021(scr, age, female):
    """CKD-EPI 2021 creatinine equation (race-free). Inker et al., NEJM 2021."""
    k = np.where(female, 0.7, 0.9)
    a = np.where(female, -0.241, -0.302)
    return (142 * np.minimum(scr / k, 1) ** a * np.maximum(scr / k, 1) ** -1.200
            * 0.9938 ** age * np.where(female, 1.012, 1.0))


# --------------------------------------------------------------------------
# 4. Variable definitions and analytic sample
# --------------------------------------------------------------------------
def build_analytic_sample(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["medicare"] = d["HIQ032B"] == 2          # checkbox value 2 = Medicare
    d["sbp"] = d[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1)
    d["dbp"] = d[["BPXODI1", "BPXODI2", "BPXODI3"]].mean(axis=1)
    d["egfr"] = egfr_ckd_epi_2021(d["LBXSCR"], d["RIDAGEYR"], d["RIAGENDR"] == 2)

    qcols = ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F",
             "DIQ010", "DIQ050", "DIQ070", "DIQ160", "BPQ020", "BPQ040A"]
    for c in qcols:  # keep 1=yes / 2=no; refused (7) / don't know (9) -> NaN
        d[c] = d[c].where(d[c].isin([1, 2]))

    d["prior_mi"] = d["MCQ160D"] == 1
    d["prior_stroke"] = d["MCQ160F"] == 1
    d["chf"] = d["MCQ160E"] == 1               # HFpEF proxy (conservative)
    d["chd"] = (d["MCQ160B"] == 1) | (d["MCQ160C"] == 1)
    d["ascvd_strict"] = d["prior_mi"] | d["prior_stroke"]   # SELECT-like ASCVD

    d["diabetes"] = ((d["DIQ010"] == 1) | (d["LBXGH"] >= 6.5)
                     | (d["DIQ050"] == 1) | (d["DIQ070"] == 1))
    d["prediab"] = (~d["diabetes"]) & d["LBXGH"].between(5.7, 6.4)  # ADA HbA1c range
    d["htn_unctrl"] = (d["BPQ040A"] == 1) & ((d["sbp"] >= 140) | (d["dbp"] >= 90))
    d["htn_unctrl_rx2"] = (d["n_antihtn"] >= 2) & ((d["sbp"] >= 140) | (d["dbp"] >= 90))
    d["ckd3a"] = d["egfr"] < 60

    ana = d[(d["RIDAGEYR"] >= 65) & d["medicare"]
            & (d["WTMECPRP"] > 0) & d["BMXBMI"].notna()].copy()

    ok = (~ana["diabetes"]) & (~ana["on_glp1"])
    ana["tierA"] = ok & (ana["BMXBMI"] >= 35)
    ana["tierB"] = ok & ana["BMXBMI"].between(30, 34.99) & (ana["chf"] | ana["htn_unctrl"] | ana["ckd3a"])
    ana["tierC"] = ok & ana["BMXBMI"].between(27, 29.99) & (ana["prediab"] | ana["prior_mi"] | ana["prior_stroke"])
    ana["eligible"] = ana["tierA"] | ana["tierB"] | ana["tierC"]
    ana["tierB_rx2"] = ok & ana["BMXBMI"].between(30, 34.99) & (ana["chf"] | ana["htn_unctrl_rx2"] | ana["ckd3a"])
    ana["eligible_sens"] = ana["tierA"] | ana["tierB_rx2"] | ana["tierC"]
    ana["bmi27plus"] = ana["BMXBMI"] >= 27
    ana["crit_pre"] = ((ana["BMXBMI"] >= 35)
                       | (ana["BMXBMI"].between(30, 34.99) & (ana["chf"] | ana["htn_unctrl"] | ana["ckd3a"]))
                       | (ana["BMXBMI"].between(27, 29.99) & (ana["prediab"] | ana["prior_mi"] | ana["prior_stroke"])))
    claims_detectable = ana["chf"] | ana["ckd3a"] | ana["prior_mi"] | ana["prior_stroke"] | ana["chd"]
    ana["lab_only"] = ana["eligible"] & (~claims_detectable)  # claims-invisible

    ana["sex"] = np.where(ana["RIAGENDR"] == 1, "Men", "Women")
    ana["agegrp"] = np.where(ana["RIDAGEYR"] < 75, "Age 65–74", "Age ≥75")
    ana["race"] = ana["RIDRETH3"].map({1: "Hispanic", 2: "Hispanic", 3: "NH White",
                                       4: "NH Black", 6: "NH Asian", 7: "Other/Multi"})
    ana["pir"] = pd.cut(ana["INDFMPIR"], [0, 1.3, 3.5, 100],
                        labels=["PIR <1.3", "PIR 1.3–3.5", "PIR >3.5"])
    return ana


# --------------------------------------------------------------------------
# 5. Design-based (Taylor-linearized) survey estimation
# --------------------------------------------------------------------------
def svy_total(y, dat, domain):
    y = np.asarray(y, dtype=float)
    domain = np.asarray(domain, dtype=bool)
    u = np.where(domain, dat["WTMECPRP"].to_numpy() * np.nan_to_num(y), 0.0)
    total = u.sum()
    tmp = pd.DataFrame({"u": u, "strata": dat["SDMVSTRA"].to_numpy(),
                        "psu": dat["SDMVPSU"].to_numpy()})
    var = 0.0
    for _, g in tmp.groupby(["strata", "psu"])["u"].sum().reset_index().groupby("strata"):
        n_h = len(g)
        if n_h > 1:
            var += n_h / (n_h - 1) * ((g["u"] - g["u"].mean()) ** 2).sum()
    return total, np.sqrt(var)


def svy_prop(y, dat, domain):
    y = np.asarray(y, dtype=float)
    domain = np.asarray(domain)
    n_tot, _ = svy_total(np.ones(len(dat)), dat, domain)
    y_tot, _ = svy_total(y, dat, domain)
    p = y_tot / n_tot
    u = np.where(domain, dat["WTMECPRP"].to_numpy() * (np.nan_to_num(y) - p), 0.0) / n_tot
    tmp = pd.DataFrame({"u": u, "strata": dat["SDMVSTRA"].to_numpy(),
                        "psu": dat["SDMVPSU"].to_numpy()})
    var = 0.0
    for _, g in tmp.groupby(["strata", "psu"])["u"].sum().reset_index().groupby("strata"):
        n_h = len(g)
        if n_h > 1:
            var += n_h / (n_h - 1) * ((g["u"] - g["u"].mean()) ** 2).sum()
    return p, np.sqrt(var)


def ci(est, se):
    return est - 1.96 * se, est + 1.96 * se


# --------------------------------------------------------------------------
# 6. Analyses
# --------------------------------------------------------------------------
def run_analysis(ana: pd.DataFrame, out_dir: str):
    dom = np.ones(len(ana), bool)
    M = 1e6
    rows = []
    print("\n== Headline estimates (weighted) ==")
    for key, label in [
        ("bmi27plus", "BMI ≥27"),
        ("crit_pre", "Meet clinical criteria (before exclusions)"),
        ("eligible", "Bridge-eligible"),
        ("tierA", "  Tier A: BMI ≥35"),
        ("tierB", "  Tier B: BMI 30–34.9 + comorbidity"),
        ("tierC", "  Tier C: BMI 27–29.9 + cardiometabolic"),
        ("lab_only", "  of which claims-invisible (lab/exam only)"),
        ("eligible_sens", "Bridge-eligible (≥2 antihypertensive classes, sensitivity)"),
    ]:
        t, se_t = svy_total(ana[key].to_numpy(), ana, dom)
        p, se_p = svy_prop(ana[key].to_numpy(), ana, dom)
        lo_p, hi_p = ci(p * 100, se_p * 100)
        lo_t, hi_t = ci(t / M, se_t / M)
        print(f"{label:55s} {p*100:5.1f}% ({lo_p:4.1f}–{hi_p:4.1f})   {t/M:6.2f}M ({lo_t:5.2f}–{hi_t:5.2f})")
        rows.append(dict(measure=label.strip(), pct=100 * p, pct_lo=lo_p, pct_hi=hi_p,
                         millions=t / M, millions_lo=lo_t, millions_hi=hi_t))

    # sensitivity: remove the HFpEF proxy (self-reported CHF) criterion entirely
    ok = (~ana["diabetes"]) & (~ana["on_glp1"])
    elig_nochf = (ana["tierA"] | ana["tierC"]
                  | (ok & ana["BMXBMI"].between(30, 34.99) & (ana["htn_unctrl"] | ana["ckd3a"])))
    p_nc, se_nc = svy_prop(elig_nochf.to_numpy(), ana, dom)
    t_nc, _ = svy_total(elig_nochf.to_numpy(), ana, dom)
    lo_p, hi_p = ci(p_nc * 100, se_nc * 100)
    print(f"{'Bridge-eligible (HFpEF/CHF criterion removed, sensitivity)':55s} "
          f"{p_nc*100:5.1f}% ({lo_p:4.1f}–{hi_p:4.1f})   {t_nc/M:6.2f}M")
    rows.append(dict(measure="Bridge-eligible (HFpEF criterion removed, sensitivity)",
                     pct=100 * p_nc, pct_lo=lo_p, pct_hi=hi_p,
                     millions=t_nc / M, millions_lo=np.nan, millions_hi=np.nan))

    n0, _ = svy_total(np.ones(len(ana)), ana, dom)
    n_ascvd, _ = svy_total((ana["eligible"] & ana["ascvd_strict"]).to_numpy(), ana, dom)
    n3 = rows[2]["millions"] * M
    n_lab = rows[6]["millions"] * M
    print(f"65+ Medicare non-institutionalized population: {n0/M:.1f}M")
    print(f"Eligible with established ASCVD (prior MI/stroke, SELECT-like): {n_ascvd/M:.2f}M")
    p_glp1, _ = svy_prop(ana["on_glp1"].to_numpy(), ana, dom)
    print(f"Current GLP-1 use among 65+ Medicare: {p_glp1*100:.1f}%")
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "headline_estimates.csv"), index=False)

    # ---- subgroup table ----
    print("\n== Eligibility by subgroup ==")
    srows = []

    def sub_row(label, mask):
        p, se = svy_prop(ana["eligible"].to_numpy(), ana, mask)
        pa, _ = svy_prop(ana["tierA"].to_numpy(), ana, mask)
        pb, _ = svy_prop(ana["tierB"].to_numpy(), ana, mask)
        pc, _ = svy_prop(ana["tierC"].to_numpy(), ana, mask)
        pl, _ = svy_prop(ana["lab_only"].to_numpy(), ana, mask & ana["eligible"].to_numpy())
        lo, hi = ci(100 * p, 100 * se)
        print(f"{label:16s} n={int(mask.sum()):5d}  {100*p:5.1f}% ({lo:4.1f}–{hi:4.1f})"
              f"  [A {100*pa:4.1f} / B {100*pb:4.1f} / C {100*pc:4.1f}]  claims-invisible {100*pl:4.1f}%")
        srows.append(dict(subgroup=label, n_unweighted=int(mask.sum()), pct=100 * p,
                          pct_lo=lo, pct_hi=hi, tierA=100 * pa, tierB=100 * pb,
                          tierC=100 * pc, claims_invisible_pct=100 * pl))

    sub_row("Overall", dom)
    for v, levels in [("sex", ["Men", "Women"]), ("agegrp", ["Age 65–74", "Age ≥75"]),
                      ("race", ["NH White", "NH Black", "Hispanic", "NH Asian", "Other/Multi"]),
                      ("pir", ["PIR <1.3", "PIR 1.3–3.5", "PIR >3.5"])]:
        for lv in levels:
            sub_row(lv, (ana[v] == lv).to_numpy())
    pd.DataFrame(srows).to_csv(os.path.join(out_dir, "subgroup_estimates.csv"), index=False)

    # ---- MACE-averted scenario grid ----
    base_risk = SELECT_PLACEBO_RATE_YR * BRIDGE_WINDOW_YR
    grid = []
    print("\n== MACE averted over 18 months (ASCVD subgroup) ==")
    for uptake in [0.10, 0.25, 0.50]:
        for adh in [0.30, 0.40, 0.60]:
            avoided = n_ascvd * uptake * adh * base_risk * (1 - SELECT_HR)
            grid.append(dict(uptake=uptake, persistence_12m=adh,
                             treated=int(n_ascvd * uptake * adh), mace_averted=round(avoided)))
    gdf = pd.DataFrame(grid)
    print(gdf.pivot(index="uptake", columns="persistence_12m", values="mace_averted").to_string())
    print(f"Theoretical maximum (100% uptake & persistence): {n_ascvd*base_risk*(1-SELECT_HR):,.0f}")
    gdf.to_csv(os.path.join(out_dir, "mace_scenarios.csv"), index=False)

    return dict(n0=n0, n3=n3, n_lab=n_lab, n_ascvd=n_ascvd, subgroups=pd.DataFrame(srows))


# --------------------------------------------------------------------------
# 7. Figure
# --------------------------------------------------------------------------
def make_figure(ana, res, out_dir):
    sg = res["subgroups"].set_index("subgroup")
    order = ["Overall", "Men", "Women", "Age 65–74", "Age ≥75",
             "NH White", "NH Black", "Hispanic", "NH Asian", "Other/Multi",
             "PIR <1.3", "PIR 1.3–3.5", "PIR >3.5"]
    blocks = [(0, 1), (1, 3), (3, 5), (5, 10), (10, 13)]  # group gaps
    ypos, yy = [], 0
    for lo, hi in blocks:
        ypos.extend(range(int(yy), int(yy) + hi - lo))
        yy += hi - lo + 0.7
    ypos = np.array(ypos, dtype=float)
    A = sg.loc[order, "tierA"].to_numpy()
    B = sg.loc[order, "tierB"].to_numpy()
    C = sg.loc[order, "tierC"].to_numpy()
    T = A + B + C

    cA, cB, cC = "#08519c", "#4292c6", "#c6dbef"
    c_detect, c_lab = "#2171b5", "#e6550d"
    M = 1e6

    stages = [("Medicare beneficiaries ≥65 y", res["n0"] / M, "#6baed6"),
              ("BMI ≥27", svy_total(ana["bmi27plus"].to_numpy(), ana, np.ones(len(ana), bool))[0] / M, "#6baed6"),
              ("Meet clinical criteria\n(before exclusions)", svy_total(ana["crit_pre"].to_numpy(), ana, np.ones(len(ana), bool))[0] / M, "#6baed6"),
              ("Bridge-eligible\n(excl. T2D & current GLP-1)", res["n3"] / M, None),
              ("Uptake @25%", res["n3"] / M * 0.25, "#9ecae1"),
              ("Adherent at 12 mo @40%", res["n3"] / M * 0.10, "#9ecae1")]

    fig = plt.figure(figsize=(14.5, 7.6))
    gs = GridSpec(1, 2, width_ratios=[1.05, 1.0], wspace=0.28)
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1])

    # Panel A: stacked eligibility by subgroup
    axA.barh(ypos, A, color=cA, height=0.72, label="Tier A: BMI ≥35")
    axA.barh(ypos, B, left=A, color=cB, height=0.72,
             label="Tier B: BMI 30–34.9 + HFpEF/unctrl. HTN/CKD 3a+")
    axA.barh(ypos, C, left=A + B, color=cC, height=0.72,
             label="Tier C: BMI 27–29.9 + prediabetes/prior MI/stroke")
    for y, t in zip(ypos, T):
        axA.text(t + 0.4, y, f"{t:.1f}%", va="center", fontsize=9, color="#333333")
    axA.set_yticks(ypos)
    axA.set_yticklabels(order, fontsize=10)
    axA.invert_yaxis()
    axA.set_xlim(0, 33)
    axA.set_xlabel("Bridge-eligible, % of Medicare beneficiaries ≥65 y", fontsize=10)
    axA.spines[["top", "right"]].set_visible(False)
    axA.legend(fontsize=8.4, loc="upper center", bbox_to_anchor=(0.5, -0.085),
               ncol=3, frameon=False, handlelength=1.2, columnspacing=1.0)
    axA.set_title("A  Bridge-eligible population by subgroup (NHANES 2017–Mar 2020)",
                  fontsize=11.5, loc="left", fontweight="bold")

    # Panel B: coverage-to-benefit cascade
    yB = np.arange(len(stages))[::-1] * 1.0
    prev = None
    for (lab, val, col), y in zip(stages, yB):
        if col is not None:
            axB.barh(y, val, color=col, height=0.62)
            axB.text(val + 0.6, y, f"{val:.1f}M", va="center", fontsize=9.5)
        else:
            det = (res["n3"] - res["n_lab"]) / M
            axB.barh(y, det, color=c_detect, height=0.62)
            axB.barh(y, res["n_lab"] / M, left=det, color=c_lab, height=0.62)
            axB.text(val + 0.6, y, f"{val:.1f}M", va="center", fontsize=9.5, fontweight="bold")
        if prev is not None:
            axB.text(0.4, y - 0.42, f"({val/prev*100:.0f}% of prev.)",
                     fontsize=7.2, color="#888888", style="italic")
        prev = val
    axB.set_yticks(yB)
    axB.set_yticklabels([s[0] for s in stages], fontsize=9.5)
    axB.spines[["top", "right"]].set_visible(False)
    axB.set_xlabel("Population, millions", fontsize=10)
    axB.set_xlim(0, 52)
    axB.set_ylim(-0.9, 5.9)
    axB.set_title("B  From coverage to cardiovascular benefit", fontsize=11.5,
                  loc="left", fontweight="bold")
    axB.legend(handles=[mpatches.Patch(color=c_detect, label=f"Claims-detectable: {(res['n3']-res['n_lab'])/M:.1f}M"),
                        mpatches.Patch(color=c_lab, label=f"Claims-invisible (lab/exam only): {res['n_lab']/M:.1f}M")],
               fontsize=8.6, loc="center right", bbox_to_anchor=(1.0, 0.62), frameon=False)
    max_mace = res["n_ascvd"] * SELECT_PLACEBO_RATE_YR * BRIDGE_WINDOW_YR * (1 - SELECT_HR)
    base_mace = max_mace * 0.25 * 0.40
    axB.text(0.40, 0.10,
             f"ASCVD subgroup (prior MI/stroke, no T2D): {res['n_ascvd']/M:.1f}M eligible\n"
             f"SELECT HR 0.80, 18-mo Bridge window:\n"
             f"• Base case (25% uptake, 40% adherence): ~{base_mace:,.0f} MACE averted\n"
             f"• Full coverage & adherence: ~{max_mace:,.0f} MACE averted",
             transform=axB.transAxes, fontsize=8.8, va="bottom",
             bbox=dict(boxstyle="round,pad=0.5", fc="#fff7ec", ec="#e6550d", lw=0.9))
    fig.text(0.5, -0.035,
             "Source: NHANES 2017–March 2020 (MEC sample; design-based weighted estimates). "
             "CMS Bridge criteria operationalized as detailed in Methods.",
             fontsize=7.5, color="#555555", ha="center")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(out_dir, f"figure_bridge_eligibility.{ext}")
        plt.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print("saved:", path)
    plt.close()


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data", help="where to store NHANES .xpt files")
    ap.add_argument("--out-dir", default="outputs", help="where to write results")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Step 1/3  download NHANES files")
    download(args.data_dir)
    print("Step 2/3  load, merge, define variables")
    ana = build_analytic_sample(load_and_merge(args.data_dir))
    print(f"  analytic sample: n = {len(ana)} (Medicare beneficiaries ≥65 y)")
    print("Step 3/3  survey-weighted analysis + figure")
    res = run_analysis(ana, args.out_dir)
    make_figure(ana, res, args.out_dir)
    print("\nDone. All outputs in:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
