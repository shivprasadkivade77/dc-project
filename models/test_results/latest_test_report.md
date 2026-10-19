# AcousticTap Hybrid Test Report (20261019_165703)

- **Winning Architecture:** `Hybrid_SVM_RBF` (50% ML + 50% Fourier Transform Triangulation)
- **Best Tuned Hyperparameters:** `{'clf__C': 5.0, 'clf__class_weight': {0: 1.3, 1: 1.0, 2: 1.0}, 'clf__gamma': 'scale'}`
- **Total Dataset Samples:** 589
- **5-Fold Cross-Validation Macro F1:** `0.8771`
- **Holdout Test Accuracy:** `88.98%`

## Holdout Classification Report
```text
              precision    recall  f1-score   support

    left_tap       0.92      0.84      0.88        43
   right_tap       0.89      0.97      0.93        40
       noise       0.86      0.86      0.86        35

    accuracy                           0.89       118
   macro avg       0.89      0.89      0.89       118
weighted avg       0.89      0.89      0.89       118

```

## Confusion Matrix (`['left_tap', 'right_tap', 'noise']`)
```text
[[36  2  5]
 [ 1 39  0]
 [ 2  3 30]]
```
