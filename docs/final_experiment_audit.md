# Auditoria da branch para experimento final

Data: 2026-06-13

## Resumo executivo

A branch reunia CDC Diabetes, 20 Newsgroups, DP noise, API exposure, top-k
risk e shadow models em runners diferentes. O pipeline final foi separado em
scripts `src/run_final_*`, todos escrevendo em `results_final/`, e shadow/HAR/
Sensorless/Canary ficaram fora dos dicionarios e runners finais.

## Scripts encontrados

- `main_experiment.py`: runner historico com `cdc_main`, `cdc_shadow`,
  `ng_main`, `ng_shadow`, `compare_current_vs_shadow`, `final_report`,
  `smoke_test` e `sanity_check`. Mistura current attacks, 20NG e shadow.
- `src/run_20newsgroups_experiment.py`: runner antigo completo de 20NG. Fitava
  TF-IDF globalmente antes dos splits.
- `src/run_noise_experiment.py`: runner antigo de DP noise para CDC.
- `src/run_api_exposure_experiment.py`: runner antigo de API exposure em CDC.
- `src/shadow_attack.py`: implementacao de shadow models. Mantida como legado
  para resultado negativo/relatorio, mas fora do pipeline final.
- `src/run_final_cdc.py`: novo runner final CDC sem DP.
- `src/run_final_cdc_dp_noise.py`: novo runner final CDC DP noise.
- `src/run_final_20news.py`: novo runner final 20NG sem DP.
- `src/run_final_20news_topk.py`: novo runner final 20NG top-k risk.
- `src/run_final_20news_dp_noise.py`: novo runner final 20NG DP noise.
- `src/run_final_20news_api_exposure.py`: novo runner final 20NG API exposure.
- `src/run_final_all.py`: runner master opcional.
- `src/build_final_summary.py`: agregador final para tabelas e numeros de slides.

## Configs encontradas

- `src/config.py` misturava CDC, DP noise e API exposure com nomes curtos
  (`baseline`, `overfitted`, `dp_sgd`), e `MAIN_EXPERIMENTS` ainda incluia DP
  no fluxo principal CDC.
- `src/config_20newsgroups.py` tinha configs separadas para 20NG, mas com nomes
  curtos duplicados (`baseline`, `overfitted`, etc.).
- A nova organizacao em `src/config.py` define blocos finais:
  `CDC_EXPERIMENTS`, `CDC_DP_EXPERIMENTS`, `NEWS20_EXPERIMENTS`,
  `NEWS20_TOPK_EXPERIMENTS`, `NEWS20_DP_EXPERIMENTS`,
  `NEWS20_API_EXPOSURE_EXPERIMENTS` e `LEGACY_EXPERIMENTS_DO_NOT_RUN`.
- Aliases antigos foram mantidos para compatibilidade com scripts historicos.

## Implementacoes finais

- CDC final sem DP: `src/run_final_cdc.py`.
- CDC DP noise: `src/run_final_cdc_dp_noise.py`, baseado em config vulneravel,
  nao em baseline fraco.
- 20NG final sem DP: `src/run_final_20news.py`.
- 20NG top-k risk: `src/run_final_20news_topk.py`.
- 20NG DP noise: `src/run_final_20news_dp_noise.py`, baseado na rodada final
  overfitted (`NEWS20_DP_NOISE_BASE_CONFIG`, derivada de
  `news20_overfitted_small`). A rodada exploratoria anterior baseada em
  `news20_vulnerable_2k` nao e mais a referencia principal.
- 20NG API exposure: `src/run_final_20news_api_exposure.py`, treina uma vez por
  modelo/seed e reusa modelo/splits para todos os exposure modes.

## Legado ou descartado do pipeline final

- Shadow models: permanecem em `main_experiment.py` e `src/shadow_attack.py`,
  mas nenhum runner final importa ou executa shadow.
- HAR: nao ha loader final nem config final; aparece apenas no historico Git e
  documentacao antiga.
- Sensorless: nao encontrado em runners Python finais.
- Canary: nao encontrado em runners Python finais; aparece apenas como nome de
  branch/historico.

## Riscos metodologicos auditados

- Splits CDC: `src/data.py` usa `fit_members`, `val_selection`, `nonmembers` e
  `holdout`, com scaler/imputer fitados apenas em `X_fit`.
- Splits 20NG antigo: `load_20newsgroups()` fitava TF-IDF antes do split, o que
  podia vazar vocabulario/estatisticas globais. Corrigido para os runners finais
  com `load_20newsgroups_texts()`, `create_text_splits_20ng()` e
  `preprocess_text_splits_20ng()`, fitando TF-IDF apenas em `X_fit`.
- Ataque: `target_val_selection` nao entra no ataque; members vem de `X_fit` e
  nonmembers vem de `X_nonmember`.
- Top-k: integrado em `run_all_attacks` via score continuo; loss usa `-loss`.
  Foi adicionado `lift_at_1/5/10_percent`.
- DP: os runners finais usam cenarios inicialmente atacaveis como referencia
  (CDC vulneravel e 20NG overfitted), registram `epsilon`, `delta`,
  `noise_multiplier` e `max_grad_norm`, e deixam o checkpoint final sem restaurar
  o melhor modelo em cenarios de memorizacao (`restore_best_model=False`).
- API exposure: o novo runner 20NG treina uma vez por modelo/seed e reusa o
  mesmo modelo e os mesmos member/nonmember loaders em todos os modos.
- Resultados antigos: nao foram apagados; resultados finais escrevem apenas em
  `results_final/` ou no diretorio informado por `--output-dir`.

## Arquivos alterados/criados

- Alterados:
  - `src/config.py`
  - `src/data_20newsgroups.py`
  - `src/membership_attack.py`
- Criados:
  - `src/final_runner_utils.py`
  - `src/run_final_cdc.py`
  - `src/run_final_cdc_dp_noise.py`
  - `src/run_final_20news.py`
  - `src/run_final_20news_topk.py`
  - `src/run_final_20news_dp_noise.py`
  - `src/run_final_20news_api_exposure.py`
  - `src/run_final_all.py`
  - `src/build_final_summary.py`
  - `results_final/README.md`
  - `docs/final_experiment_audit.md`

## Checklist antes de rodar o experimento final

- Verificar ambiente PyTorch/CUDA.
- Rodar smoke tests dos runners nao-DP e API.
- Confirmar que `results_final_smoke/` contem CSVs e figuras.
- Confirmar que `20NG` final usa `preprocess_text_splits_20ng`, nao
  `load_20newsgroups()` vetorizado globalmente.
- Confirmar que Opacus esta instalado antes de rodar DP:
  `python -c "import opacus; print(opacus.__version__)"`.
- Rodar primeiro uma seed nos DP runners se houver limitacao de tempo/GPU.
- Nao usar `main_experiment.py --run_mode *_shadow` como experimento final.
- Nao misturar CSVs antigos de `results/`, `results_api/`, `resultsdp/` ou
  `results_20ng/` nas tabelas finais.
