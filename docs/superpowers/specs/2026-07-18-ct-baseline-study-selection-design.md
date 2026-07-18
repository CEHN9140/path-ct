# CT Baseline Study Selection

## Objective

Before multimodal matching, keep one baseline CT acquisition date per TCGA-KIRC subject so that downstream CT quality control selects Series from a single time point.

## Selection rule

1. Load and path-filter CT rows as currently implemented in `process_ct`.
2. Parse `Study Date` using the metadata format `%m-%d-%Y`; an invalid date is an error.
3. Group rows by `Subject ID` and find the earliest `Study Date` for each subject.
4. Keep every Series acquired on that earliest date.
5. If several `Study UID` values share the earliest date, keep them all; downstream CT QC will select the Series.
6. Preserve `Study UID`, `Study Date`, and existing Series metadata in the generated multimodal JSON.

This rule selects the earliest TCIA baseline time point. It does not claim direct alignment with the surgery date.

## Verification

- Confirm that the number of subjects is unchanged.
- Confirm that every retained row has its subject's minimum valid `Study Date`.
- Confirm that same-day Study UIDs and all of their Series are retained.
- Confirm that the generated JSON contains no later CT dates for a subject.
