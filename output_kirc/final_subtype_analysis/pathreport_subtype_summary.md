# Final subtype pathology-report summary

## Scope

- Source: `data/TCGA_Reports.csv`, matched by the complete `case ID + report UUID` key.
- Case-level normalized fields and their verbatim source sentences are recorded in `pathreport_case_core_fields.csv`; the same sentences are presented by subtype in `pathreport_case_original_evidence.md`.
- Verbatim evidence preserves the source capitalization, punctuation, wording, and OCR errors. ` || ` separates two adjacent source sentences when one sentence alone does not contain the complete field.
- Included: 76 of 79 final-subtype patients.
- Excluded by request: `TCGA-B0-5706`, `TCGA-B0-5709`, and `TCGA-B0-5712` because they were absent from the CSV corpus.
- This is a descriptive review of the available reports. Counts use explicit report statements; an unmentioned feature is not treated as negative.
- Fuhrman grade and pT are retained as originally reported. Reports span different historical staging conventions.

## Comparative summary

| Subtype | Reports | Tumor size, median (IQR), cm | Fuhrman grade | pT recorded in report | Pathology profile |
|---|---:|---:|---|---|---|
| SUBTYPE01 | 10 | 4.35 (3.63-5.58) | G2 4; G2-3 1; G3 5 | 9/10: pT1* 6, pT2 1, pT3* 2 | Conventional clear-cell tumors of intermediate grade, usually localized, with a minority showing renal-vein or perinephric extension. |
| SUBTYPE02 | 15 | 4.10 (3.25-5.25) | G2 8; G3 6; G4 1 | 15/15: pT1* 11, pT2 2, pT3* 2 | Predominantly localized low/intermediate-grade clear-cell tumors. One distinct G4 sarcomatoid-necrotic outlier and one pT3b renal-vein-invasive case are present. |
| SUBTYPE03 | 11 | 3.50 (2.35-4.50) | G2 5; G3 6 | 10/11: pT1* 8, pT2a 1, pT3a 1 | Smallest and most localized profile overall. Most are pT1 clear-cell tumors; renal-sinus invasion is exceptional rather than typical. |
| SUBTYPE04 | 15 | 6.40 (4.90-8.75) | G2 4; G3 4; G4 7 | Explicit pT was available in only 6/15 reports | Most aggressive profile: largest tumors, frequent G4 morphology, renal-vein/sinus/pericapsular invasion, necrosis, rhabdoid differentiation, and documented metastatic disease in selected cases. |
| SUBTYPE05 | 25 | 5.00 (4.00-6.80) | G2 12; G2 with focal G3 1; G3 9; G4 3 | 24/25: pT1* 15, pT3* 8, pT4 1 | Heterogeneous or bimodal profile: a large localized pT1 group coexists with a substantial advanced group showing renal-vein invasion, nodal/metastatic disease, necrosis, or sarcomatoid differentiation. |

`pT1*` and `pT3*` combine the corresponding unspecified and lettered subcategories exactly as stated in the reports.

## Subtype interpretations

### SUBTYPE01: intermediate-grade, mostly localized clear-cell RCC

- All 10 reports describe conventional/clear-cell RCC.
- Tumors are generally small to medium, although one reaches 9.0 cm.
- Grades cluster between Fuhrman 2 and 3; no grade-4 tumor is present.
- Six of nine explicitly staged tumors are pT1. Two reports document pT3/pT3a disease.
- Most reports describe negative surgical margins and confinement to the kidney.
- The main invasive exceptions are `TCGA-BP-4989` with renal-vein involvement and `TCGA-BP-5183` with extension through the renal capsule/perinephric tissue.

Interpretation: a predominantly localized clear-cell group with intermediate nuclear grade, plus a small invasive tail.

### SUBTYPE02: localized low/intermediate-grade group with an isolated aggressive outlier

- All 15 reports describe conventional/clear-cell RCC.
- Eleven of 15 tumors are recorded as pT1; most grades are Fuhrman 2 or 3.
- `TCGA-BP-4995` shows muscularized renal-vein branch involvement and pT3b disease.
- `TCGA-BP-5200` is a distinct aggressive outlier: Fuhrman grade 4, focal sarcomatoid differentiation, large areas of necrosis, and a 7.5-cm tumor.
- The remaining reports largely describe confined tumors and negative margins.

Interpretation: mainly a localized clear-cell phenotype; the overall subtype should not be labelled sarcomatoid because that feature is concentrated in one case.

### SUBTYPE03: smallest and most localized pathology profile

- All 11 reports describe clear-cell RCC.
- Median tumor size is 3.5 cm, the smallest among the five subtypes.
- No grade-4 tumor is present; grades are split between Fuhrman 2 and 3.
- Eight of 10 explicitly staged tumors are pT1.
- `TCGA-BP-4967` is the major invasive exception, with renal-sinus fat involvement and pT3a disease.
- `TCGA-B8-5549` describes tumor necrosis but no sarcomatoid differentiation.

Interpretation: the most compact, predominantly organ-confined clear-cell group, without a strong high-grade dedifferentiation signal.

### SUBTYPE04: high-grade, large and invasive phenotype

- All 15 included reports describe clear-cell/conventional RCC.
- Seven of 15 tumors are Fuhrman grade 4, and the median size is 6.4 cm.
- Explicit necrosis is described in at least seven reports.
- `TCGA-BP-4992` has focal sarcomatoid differentiation.
- `TCGA-CJ-4637` and `TCGA-CJ-4641` have rhabdoid differentiation; both are grade 4.
- Multiple reports document renal-vein, renal-sinus, hilar/pericapsular vessel, or perinephric tissue involvement.
- `TCGA-CJ-4637` documents adrenal metastatic RCC, while `TCGA-G6-A5PC` records lung metastases.
- Most resection margins remain negative, showing that negative margins do not imply biologically indolent disease.

Interpretation: the clearest aggressive pathological subtype, defined by high grade, larger size, invasive growth and dedifferentiated morphology.

### SUBTYPE05: heterogeneous, with localized and highly aggressive branches

- Twenty-four reports describe clear-cell/conventional RCC. `TCGA-B0-5117` is an explicit chromophobe RCC discordance within the TCGA-KIRC-labelled cohort.
- Twelve tumors are Fuhrman grade 2, but 13 have grade-3/4 morphology or a focal grade-3 component.
- Fifteen of 24 explicitly staged tumors are pT1, whereas nine are pT3/pT4.
- Sarcomatoid differentiation is explicit in `TCGA-BP-4770` and `TCGA-CZ-5468`.
- `TCGA-BP-4770` is the strongest aggressive case: 15-cm, grade-4, predominantly sarcomatoid, pT4, and tumor at an inked resection margin.
- `TCGA-B8-5158` is grade 4, pT3a/pN1, with metastatic RCC in 1/15 hilar lymph nodes.
- Several cases show renal-vein tumor thrombus/invasion, renal-sinus or perinephric extension, and tumor necrosis.
- `TCGA-B0-4821` includes renal-vein thrombus and a report of metastatic RCC in a lung specimen.

Interpretation: not a single uniformly aggressive class. It combines a sizeable localized low-grade component with a clinically important advanced, invasive and occasionally sarcomatoid component.

## Overall ordering suggested by the pathology reports

From the available descriptive evidence:

1. **SUBTYPE04** has the strongest high-grade and invasive phenotype.
2. **SUBTYPE05** is the most heterogeneous and contains a substantial aggressive branch.
3. **SUBTYPE01** and **SUBTYPE02** are mainly localized/intermediate groups with smaller invasive or high-grade subsets.
4. **SUBTYPE03** is the most consistently small and localized group.

This ordering is descriptive and should be compared with the existing molecular, imaging and survival results rather than treated as a new subtype definition.
