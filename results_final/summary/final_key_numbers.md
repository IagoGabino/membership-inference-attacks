# Final key numbers

Generated from CSVs under the selected results_final directory.

## CDC best attack AUC

- cdc_vulnerable_5k: mean best attack AUC = 0.6496
- cdc_vulnerable_2k: mean best attack AUC = 0.6465
- cdc_overfitted: mean best attack AUC = 0.5978
- cdc_baseline: mean best attack AUC = 0.5080
- cdc_regularized: mean best attack AUC = 0.5056
- Best: cdc_vulnerable_5k (0.6496)

## CDC DP noise effect

- no DP reference: mean best attack AUC = 0.6497
- noise=0.10: mean best attack AUC = 0.6110
- noise=0.30: mean best attack AUC = 0.5850
- noise=0.60: mean best attack AUC = 0.5296
- noise=1.00: mean best attack AUC = 0.5348

## 20NG best attack AUC

- news20_vulnerable_2k: mean best attack AUC = 0.9404
- news20_overfitted_small: mean best attack AUC = 0.9075
- news20_baseline_small: mean best attack AUC = 0.8525
- news20_vulnerable_5k: mean best attack AUC = 0.8512
- news20_overfitted: mean best attack AUC = 0.8202
- news20_baseline: mean best attack AUC = 0.7332
- news20_regularized: mean best attack AUC = 0.6868
- Best: news20_vulnerable_2k (0.9404)

## 20NG top-k risk

- news20_vulnerable_2k: mean best top-5% precision = 0.8889
- news20_baseline_small: mean best top-5% precision = 0.8593
- news20_overfitted_small: mean best top-5% precision = 0.8296
- news20_vulnerable_5k: mean best top-5% precision = 0.7630
- news20_overfitted: mean best top-5% precision = 0.7284
- news20_baseline: mean best top-5% precision = 0.7037
- news20_regularized: mean best top-5% precision = 0.6519

## 20NG DP noise effect (overfitted baseline)

- no DP reference: mean best attack AUC = 0.8889
- noise=0.10: mean best attack AUC = 0.8683
- noise=0.30: mean best attack AUC = 0.8053
- noise=0.60: mean best attack AUC = 0.7007
- noise=1.00: mean best attack AUC = 0.6518

## 20NG API exposure

- oracle_full: mean best attack AUC = 0.7952
- probs_full: mean best attack AUC = 0.7948
- top1_confidence: mean best attack AUC = 0.7678
- probs_rounded_2: mean best attack AUC = 0.7479
- probs_rounded_1: mean best attack AUC = 0.7076
- label_only: mean best attack AUC = 0.6697
