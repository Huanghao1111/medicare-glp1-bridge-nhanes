# Medicare GLP-1 Bridge eligibility — a measurement-based analysis (NHANES)

Companion code for the research letter:

> **Half of Medicare Beneficiaries Eligible for the GLP-1 Bridge Program Are Invisible to Claims Data**

Using directly measured data from NHANES (2017–March 2020), we estimate that **21.4% of Medicare beneficiaries aged ≥65 years — 9.7 million (95% CI 8.3–11.1)** — meet the eligibility criteria of the Medicare GLP-1 Bridge program (launched July 1, 2026), **2.6-fold the claims-based projection of 3.8 million**. **51% of eligible beneficiaries qualify only through laboratory or examination findings** (undiagnosed prediabetes or measured uncontrolled hypertension) and would be invisible to claims-based screening.

## Key findings

| Measure | Estimate (95% CI) |
|---|---|
| Bridge-eligible, % of Medicare ≥65 y | **21.4% (19.5–23.3)** |
| Bridge-eligible population | **9.73M (8.31–11.14)** |
| — Tier A (BMI ≥35) | 3.93M |
| — Tier B (BMI 30–34.9 + comorbidity) | 2.34M |
| — Tier C (BMI 27–29.9 + cardiometabolic criteria) | 3.46M |
| Claims-invisible share of the eligible | **51% (4.94M)** |
| Eligible with prior MI/stroke (SELECT-like ASCVD) | 1.62M |
| MACE averted over 18 mo (25% uptake, 40% persistence) | ~1,200 (max ~11,700) |
| Eligibility, NH Black vs NH Asian | 28.5% vs 8.3% |

## Why the 2017–March 2020 cycle?

It is the most recent NHANES cycle whose **public release includes prescription-drug names** (`RXQ_RX`), which are required to exclude current GLP-1 users and to count antihypertensive medication classes. The 2021–2023 public files no longer release drug names.

## Quick start

```bash
pip install -r requirements.txt
python analysis.py            # downloads NHANES automatically (CDC servers)
```

Options: `--data-dir` (default `data/`), `--out-dir` (default `outputs/`).

## Outputs

- `figure_bridge_eligibility.png` / `.pdf` — two-panel main figure (subgroup eligibility; coverage-to-benefit cascade)
- `headline_estimates.csv` — weighted percentages and population counts with 95% CIs
- `subgroup_estimates.csv` — eligibility by sex, age, race/ethnicity, income
- `mace_scenarios.csv` — MACE-averted scenario grid (uptake × persistence)

## Operationalization of CMS Bridge criteria (announced May 6, 2026)

| CMS criterion | NHANES implementation |
|---|---|
| BMI ≥35 | measured `BMXBMI` ≥35 |
| BMI 30–34.9 + HFpEF / uncontrolled hypertension / CKD ≥3a | self-reported CHF (`MCQ160E`) \| on antihypertensive (`BPQ040A`) with measured BP ≥140/90 \| eGFR <60 (CKD-EPI 2021, `LBXSCR`) |
| BMI 27–29.9 + prediabetes / prior MI / prior stroke / symptomatic PAD | HbA1c 5.7–6.4% (`LBXGH`) \| `MCQ160D` \| `MCQ160F`; PAD unavailable → conservative |
| Exclusions: T2D, OSA, MASH, current GLP-1 use | diabetes by self-report/medication/HbA1c ≥6.5%; GLP-1 users via `RXDDRUG`; OSA/MASH unavailable → conservative |

Estimates use MEC examination weights (`WTMECPRP`) with Taylor-linearized design-based SEs (`SDMVSTRA`/`SDMVPSU`). Benefit modeling applies the SELECT hazard ratio (0.80) and placebo-arm MACE rate (2.41%/year) to the eligible subgroup with prior MI/stroke over the 18-month program window.

## Limitations

Non-institutionalized sampling frame; HFpEF approximated by self-reported congestive heart failure (removing this criterion entirely changes eligibility from 21.4% to 20.9%); OSA/MASH/PAD not measurable (estimates biased downward); 2017–2020 data precede further rises in obesity prevalence; SELECT effect extrapolation; Part D enrollment approximated by Medicare coverage.

## Citation

> [Author list]. Half of Medicare Beneficiaries Eligible for the GLP-1 Bridge Program Are Invisible to Claims Data. *Diabetes Care* (under review), 2026.

## License

Code: MIT. NHANES data are public domain (CDC/NCHS); see CDC data-use policies.
