# Dataset

This project uses the public **Credit Card Fraud Detection** dataset (ULB / Worldline).

Do **not** commit `creditcard.csv` (about 150 MB). Download it and place it here:

```text
data/creditcard.csv
```

Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Expected schema

| Column | Description |
| --- | --- |
| `Time` | Seconds elapsed between this transaction and the first transaction in the file |
| `V1` … `V28` | PCA-transformed features |
| `Amount` | Transaction amount |
| `Class` | `1` = fraud, `0` = legitimate |

The file contains 284,807 transactions and 492 frauds (about 0.172% positives).
