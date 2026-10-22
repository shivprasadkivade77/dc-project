# AcousticTap Hybrid Test Report (20261023_005613)

- **Winning Architecture:** `ExtraTrees_V3` (85% ML + 15% Fourier Triangulation)
- **Total Dataset Samples:** 839 (671 train / 168 holdout)
- **5-Fold Cross-Validation Macro F1:** `0.8984`
- **Holdout Test Accuracy:** `92.86%`
- **Holdout Test Macro F1:** `0.9291`

## Holdout Classification Report
```text
              precision    recall  f1-score   support

    left_tap       0.88      0.94      0.91        63
       noise       0.91      0.93      0.92        45
   right_tap       1.00      0.92      0.96        60

    accuracy                           0.93       168
   macro avg       0.93      0.93      0.93       168
weighted avg       0.93      0.93      0.93       168

```

## Confusion Matrix (`['left_tap', 'noise', 'right_tap']`)
```text
[[59  4  0]
 [ 3 42  0]
 [ 5  0 55]]
```
