# Results final

Esta pasta contem as saidas dos experimentos finais. Os resultados antigos em
`results/`, `results_api/`, `resultsdp/` ou `results_20ng/` nao sao usados como
fonte de verdade para o relatorio final.

Nota: se `results_final/20news_dp/` existir, ela corresponde a uma rodada
exploratoria anterior baseada em cenario vulnerable. A referencia final de DP
para 20NG e `results_final/20news_dp_overfitted/`.

## Estrutura

```text
results_final/
|-- cdc/
|   |-- metrics/cdc_results.csv
|   |-- metrics/cdc_best_attack_summary.csv
|   `-- figures/
|-- cdc_dp/
|   |-- metrics/cdc_dp_noise_results.csv
|   |-- metrics/cdc_dp_noise_summary.csv
|   `-- figures/
|-- 20news/
|   |-- metrics/20news_results.csv
|   |-- metrics/20news_best_attack_summary.csv
|   `-- figures/
|-- 20news_topk/
|   |-- metrics/20news_topk_results.csv
|   |-- metrics/20news_topk_best_attack_summary.csv
|   `-- figures/
|-- 20news_dp_overfitted/
|   |-- metrics/20news_dp_noise_results.csv
|   |-- metrics/20news_dp_noise_summary.csv
|   `-- figures/
|-- 20news_api/
|   |-- metrics/20news_api_exposure_results.csv
|   |-- metrics/20news_api_exposure_best_attack_summary.csv
|   |-- metrics/20news_api_exposure_impact_vs_oracle.csv
|   `-- figures/
`-- summary/
    |-- cdc_best_results.csv
    |-- cdc_dp_summary.csv
    |-- 20news_best_results.csv
    |-- 20news_topk_summary.csv
    |-- 20news_dp_summary.csv
    |-- 20news_api_summary.csv
    `-- final_key_numbers.md
```

## CSVs para usar no relatorio

- CDC principal: `summary/cdc_best_results.csv`
- Efeito DP no CDC: `summary/cdc_dp_summary.csv`
- 20NG principal: `summary/20news_best_results.csv`
- Risco top-k 20NG: `summary/20news_topk_summary.csv`
- Efeito DP no 20NG overfitted: `summary/20news_dp_summary.csv`
- API exposure 20NG: `summary/20news_api_summary.csv`
- Numeros prontos para slides: `summary/final_key_numbers.md`

## CDC DP noise - best attack AUC

Medias por seed extraidas de `cdc_dp/metrics/cdc_dp_noise_summary.csv`:

| config | mean best_attack_auc |
| --- | ---: |
| sem DP | 0.6497 |
| noise 0.10 | 0.6110 |
| noise 0.30 | 0.5850 |
| noise 0.60 | 0.5296 |
| noise 1.00 | 0.5348 |

## Comando para reconstruir o sumario

```bash
python -m src.build_final_summary --input-dir results_final --output-dir results_final/summary
```
