# AcousticTap Hybrid Test Report (20261019_152054)

- **Winning Architecture:** `Hybrid_SVM_RBF` (50% ML + 50% Fourier Transform Triangulation)
- **Total Dataset Samples:** 314
- **5-Fold Cross-Validation Macro F1:** `0.8549`
- **Holdout Test Accuracy:** `82.54%`

## Holdout Classification Report
```text
              precision    recall  f1-score   support

    left_tap       0.77      0.74      0.76        23
   right_tap       1.00      0.90      0.95        20
       noise       0.74      0.85      0.79        20

    accuracy                           0.83        63
   macro avg       0.84      0.83      0.83        63
weighted avg       0.83      0.83      0.83        63

```

## Confusion Matrix (`['left_tap', 'right_tap', 'noise']`)
```text
[[17  0  6]
 [ 2 18  0]
 [ 3  0 17]]
```
