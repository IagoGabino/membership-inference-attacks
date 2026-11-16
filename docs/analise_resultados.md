# Análise dos Resultados — Membership Inference Vulnerability Analysis

> Execução: 04/06/2026 | Dataset: CDC Diabetes Health Indicators (253k amostras, 21 features)
> 3 seeds (42, 123, 7) | 11 configurações de modelo | 3 ataques | 99 experimentos totais

## Visão Geral

Pipeline experimental completo de membership inference attack em dados tabulares sensíveis. Avalia quais fatores aumentam a vulnerabilidade de modelos MLP a ataques que tentam inferir se um dado pertenceu ao conjunto de treinamento.

---

## Fase 1 — MVP (MEDIUM = 40k amostras, ~20k membros)

| Modelo | Train Acc | Test Acc | Gen Gap | Attack AUC (mean ± std) |
|--------|-----------|----------|---------|-------------------------|
| **baseline** | 86.8% | 86.6% | 0.002 ± 0.003 | 0.503 ± 0.009 |
| **overfitted** | 98.5% | 82.7% | 0.159 ± 0.002 | **0.594 ± 0.011** |
| **regularized** | 86.6% | 86.5% | 0.001 ± 0.005 | 0.506 ± 0.009 |

**Observações:**
- O baseline (early stopping ~12 épocas) e o regularized (dropout 0.3 + weight decay + early stopping ~13 épocas) não apresentam overfitting e os ataques não superam random chance.
- O overfitted (80 épocas, sem regularização) memoriza os dados de treino e é o único significativamente vulnerável.
- Resultados estáveis entre seeds (std baixo).

---

## Fase 2 — Efeito do tamanho de treino (SMALL = 10k, ~5k membros)

| Modelo | Train Acc | Test Acc | Gen Gap | Attack AUC (mean ± std) |
|--------|-----------|----------|---------|-------------------------|
| **overfitted_small** | 99.7% | 82.3% | 0.174 ± 0.009 | **0.608 ± 0.032** |
| **baseline_small** | 86.9% | 86.2% | 0.006 ± 0.008 | 0.508 ± 0.020 |

**Observações:**
- overfitted_small tem gen_gap e attack AUC ligeiramente maiores que overfitted MEDIUM, mas a diferença é modesta e a variância entre seeds é maior (std 0.032).
- baseline_small permanece seguro mesmo com 4x menos dados.
- O efeito do tamanho dos dados é secundário ao efeito do overfitting neste cenário.

---

## Fase 3 — DP-SGD

### Arquitetura pequena [64, 32] (mesma do baseline)

| Modelo | ε (epsilon) | Attack AUC (mean ± std) |
|--------|-------------|-------------------------|
| **dp_sgd_nm0.5** | ~20 | 0.506 ± 0.010 |
| **dp_sgd_nm1.0** | ~3 | 0.503 ± 0.007 |
| **dp_sgd_nm1.5** | ~1.5 | 0.507 ± 0.004 |

### Arquitetura overfitted [256, 128, 64] com DP

| Modelo | Train Acc | Test Acc | Gen Gap | Attack AUC (mean ± std) |
|--------|-----------|----------|---------|-------------------------|
| **dp_overfitted_nm0.5** | 86.1% | 86.1% | ~0 | 0.500 ± 0.007 |
| **dp_overfitted_nm1.0** | 86.1% | 86.1% | ~0 | 0.509 ± 0.005 |
| **dp_overfitted_nm1.5** | 86.1% | 86.1% | ~0 | 0.502 ± 0.008 |

**Observações:**
- DP-SGD neutraliza completamente o ataque em ambas as arquiteturas.
- O resultado mais revelador: a **mesma arquitetura** que sem DP atinge AUC 0.594 (overfitted), com DP cai para ~0.50 (random).
- O mecanismo: DP impede a memorização (train acc cai de 98.5% → 86.1%), eliminando o sinal explorado pelo atacante.
- A utilidade de classificação (test acc ~86%) é preservada — o modelo não precisava memorizar para classificar bem.
- O ε diminui com mais ruído (20 → 3 → 1.5), mas neste cenário todos os noise multipliers já bastam para proteção total.

---

## Validação das Hipóteses

| Hipótese | Resultado | Evidência |
|----------|-----------|-----------|
| **H1** Overfitting correlaciona com maior sucesso do ataque | ✅ Confirmada | AUC 0.594±0.011 (overfitted) vs 0.503±0.009 (baseline) |
| **H2** Maior confiança em membros correlaciona com vulnerabilidade | ✅ Confirmada | conf_gap 0.015±0.001 (overfitted) vs ~0 (baseline) |
| **H3** Regularização reduz o ataque | ✅ Confirmada | regularized indistinguível de random (0.506±0.009) |
| **H4** DP-SGD protege, com custo para modelos que overfittariam | ✅ Confirmada | dp_overfitted: train acc 98.5%→86.1%, ataque neutralizado |
| **H5** Menos dados → mais risco | ⚠️ Parcial | Gen gap maior (0.174 vs 0.159), AUC ligeiramente maior mas com variância alta |
| **H6** Features comportamentais explicam o ataque | ✅ Confirmada | 6 features (loss, confiança, entropia) suficientes para AUC 0.60 |

---

## Discussão: Overfitting e Membership Inference

No cenário experimental avaliado, observamos **forte correlação entre overfitting e sucesso do ataque**. Todos os modelos sem overfitting apresentaram attack AUC indistinguível de random chance, enquanto modelos com overfitting foram os únicos vulneráveis.

Contudo, é importante notar que **overfitting não é condição necessária nem suficiente para membership inference**:

- **Não é necessária**: ataques mais sofisticados (LiRA, calibrated attacks) podem explorar diferenças sutis mesmo em modelos bem generalizados, como variações nos logits ou gradientes. Nosso cenário (black-box, 6 features comportamentais) pode não capturar esses sinais mais fracos.

- **Não é suficiente**: um modelo overfitted com saídas restritas (apenas top-1 label, sem probabilidades) ou com mecanismos de calibração pode não fornecer sinal suficiente ao atacante.

A formulação precisa é: **overfitting é um fator de risco que amplifica o sinal disponível ao atacante**, tornando ataques simples (threshold, LR, RF) eficazes. Em nosso cenário experimental, esse fator foi determinante.

Referências:
- Shokri et al. (2017) — correlação entre overfitting e MI
- Yeom et al. (2018) — relação formal entre generalization gap e MI
- Carlini et al. (2022) — ataques eficazes mesmo com baixo overfitting (LiRA)

---

## Estabilidade dos Resultados (Multi-seed)

A execução com 3 seeds confirma que os resultados são estáveis:
- Modelos não vulneráveis: std do attack AUC entre 0.004–0.010
- Modelos vulneráveis: std de 0.011 (overfitted MEDIUM) e 0.032 (overfitted_small)
- A maior variância do overfitted_small é esperada dado o menor volume de dados

Nenhuma conclusão muda entre seeds — as hipóteses são confirmadas ou rejeitadas de forma consistente.

---

## Artefatos Gerados

### Training Curves

O gráfico `training_curves.png` ilustra visualmente a raiz da vulnerabilidade:

- **BASELINE** (12 épocas) — train e val loss convergem juntos para ~0.31. Sem gap, sem memorização. Ataque = random.
- **OVERFITTED** (80 épocas) — train loss cai para ~0.03 enquanto val loss **sobe** de 0.3 para 1.5. Divergência clássica de overfitting. É esse gap que o ataque explora (AUC 0.594).
- **REGULARIZED** (14 épocas) — comportamento idêntico ao baseline. Dropout + weight decay impede a divergência. Ataque = random.

A divergência das curvas no overfitted é o sinal que permite ao atacante distinguir membros de não-membros: o modelo "lembra" dos exemplos de treino (loss baixa) mas erra mais em dados novos (loss alta).

---

## Artefatos Gerados

- `results/metrics/summary_results.csv` — 99 linhas (11 modelos × 3 ataques × 3 seeds)
- `results/metrics/aggregated_results.csv` — mean ± std por (modelo, ataque)
- `results/figures/multi_seed_error_bars.png` — AUC com barras de erro
- `results/figures/generalization_gap_vs_attack_auc.png`
- `results/figures/confidence_gap_vs_attack_auc.png`
- `results/figures/attack_auc_by_model.png`
- `results/figures/utility_vs_privacy_attack.png`
- `results/figures/train_size_vs_attack_auc.png`
- `results/figures/epochs_vs_attack_auc.png`
- `results/figures/epsilon_vs_attack_auc.png`
- `results/figures/epsilon_vs_accuracy.png`

---

## Sobre os Valores de Attack AUC (~0.60)

O AUC máximo observado (~0.60) é modesto comparado a resultados da literatura (0.70–0.90+). Isso não invalida o trabalho — é um resultado **realista** explicado pelas características do cenário:

1. **Dataset binário** — com apenas 2 classes, o vetor de probabilidades carrega pouca informação discriminativa. Datasets com muitas classes (CIFAR-100, ImageNet) fornecem muito mais sinal ao atacante.

2. **Dataset grande e "fácil"** — CDC Diabetes tem 253k amostras disponíveis e o modelo baseline acerta 86% sem memorizar. Há pouco incentivo para o modelo memorizar exemplos individuais.

3. **Features tabulares genéricas** — alta sobreposição entre amostras. Diferente de imagens ou texto, onde exemplos são mais únicos e identificáveis.

4. **Ataques simples** — threshold, Logistic Regression e Random Forest sobre 6 features comportamentais. Ataques mais sofisticados (shadow models, LiRA de Carlini et al. 2022) extrairiam mais sinal.

**O que importa para as hipóteses não é o valor absoluto do AUC, mas:**
- A diferença relativa entre configurações (0.60 vs 0.50) é estatisticamente significativa (não se sobrepõe entre seeds)
- Regularização e DP neutralizam completamente o sinal (AUC → 0.50)
- Os resultados são consistentes com a literatura em cenários comparáveis

Para fortalecer os ataques (se desejado em trabalhos futuros):
- Treinar com menos amostras (2k–5k membros)
- Treinar por mais épocas (200+) para overfitting extremo
- Implementar ataques com shadow models ou LiRA
- Usar datasets com mais classes

---

## Comparação com a Literatura

### Shokri et al. (2017) — "Membership Inference Attacks Against Machine Learning Models"

O paper fundacional de MI avaliou ataques contra modelos treinados em Google Prediction API e Amazon ML. Comparação direta com nossos resultados:

| Aspecto | Shokri et al. | Nosso experimento |
|---------|---------------|-------------------|
| Cenário | Black-box, vetor de probabilidades | Black-box, vetor de probabilidades |
| Ataque | Shadow models (100 shadow models) | Threshold + LR + RF sobre 6 features |
| Dataset comparável | **Adult Census** (binário, 14 features) | **CDC Diabetes** (binário, 21 features) |
| Train acc (Adult/CDC) | 0.848 | 0.868 (baseline) |
| Test acc (Adult/CDC) | 0.842 | 0.866 (baseline) |
| Gap | 0.006 | 0.002 |
| **Attack precision/AUC** | **0.503 (random)** | **0.503 (random)** |

**Nossos resultados no baseline são idênticos aos de Shokri et al. no Adult Census** — ambos datasets binários, sem overfitting, ataque = random chance. Isso valida nosso pipeline experimental.

### Cenário com overfitting

| Aspecto | Shokri et al. (Purchase 100 classes) | Nosso experimento (overfitted) |
|---------|--------------------------------------|-------------------------------|
| Train acc | 0.999 | 0.985 |
| Test acc | 0.656 | 0.827 |
| Gap | 0.343 | 0.159 |
| Attack | **0.935** (precision) | **0.594** (AUC) |

A diferença nos valores absolutos é explicada por:
1. **Número de classes**: Shokri usa 100 classes (100 sinais no vetor de probabilidades) vs nossas 2 classes (1 sinal)
2. **Tipo de ataque**: shadow models são mais poderosos que threshold/LR/RF sobre features
3. **Gap menor**: nosso overfitted ainda generaliza razoavelmente (82.7% test acc)

O paper confirma explicitamente que **classificadores binários são intrinsecamente mais resistentes a MI**:

> "The model is a binary classifier, which means that the attacker has to distinguish members from non-members by observing essentially 1 signal, since the two outputs are complements of each other. This is not enough to extract useful membership information."

### Mitigações — Regularização

Shokri et al. mostram que L2 regularization com λ=1e-2 reduz o ataque de precision 0.87 → 0.53 (próximo de random). Nossos resultados confirmam: modelos com regularização (dropout + weight decay + early stopping) têm attack AUC ~0.50 — consistente com os achados do paper.

### DP-SGD

O paper menciona differential privacy como defesa formal ideal mas **não a avalia empiricamente**. Nosso trabalho vai além ao implementar e medir DP-SGD com Opacus, mostrando que:
- DP neutraliza o ataque (AUC → 0.50) em todas as configurações testadas
- O custo de utilidade depende da arquitetura: modelo simples mantém 86% test acc, modelo complexo perde capacidade de memorização (train acc 98.5% → 86.1%)

### O que nosso trabalho adiciona à literatura

1. **Avaliação empírica de DP-SGD** — o paper original só menciona como defesa teórica
2. **Multi-seed** — confirmação de estabilidade estatística dos resultados (3 seeds, mean ± std)
3. **Comparação direta regularização vs DP** — mostramos que regularização convencional já basta neste cenário
4. **Dataset de saúde binário** — complementa o Texas Hospital (100 classes) de Shokri com um cenário binário (CDC Diabetes)
5. **Features comportamentais explícitas** — em vez de shadow models opacos, usamos 6 features interpretáveis (loss, confiança, entropia)
