# biocomp-cb396-prediction

Predição de estrutura secundária de proteínas (**Q3**: hélice/folha/coil) no
dataset **CB396**, a partir de perfis evolutivos (**PSSM**, via PSI-BLAST) e
modelos clássicos e neurais.

Resultados (Q3, validação cruzada por proteína — `GroupKFold` 7 folds):

| Modelo | Q3 |
|--------|-----|
| Linear SVM | 0,6962 |
| CNN 1D | 0,7344 |
| **CNN + BiLSTM** | **0,7570** |
| CNN + BiLSTM + CRF | 0,7464 |

## Instalação (uv)

```bash
uv sync
```

Isso cria o ambiente virtual e instala as dependências. Para rodar o pipeline
completo (download → PSI-BLAST) é necessário também o **`ncbi-blast+`** instalado
no sistema (`psiblast`, `makeblastdb`, `blastdbcmd`); a etapa de treino/avaliação
**não** precisa de BLAST, mas precisa que as features já tenham sido geradas.

## Reproduzir os experimentos

O pipeline completo baixa o dataset, gera as PSSMs e as features, e treina/avalia
os modelos:

```bash
uv run cb396 dataset     # baixa CB396, parseia, salva CSV/FASTA (392 válidas)
uv run cb396 blastdb     # baixa Swiss-Prot pré-formatado (~213 MB)
uv run cb396 psiblast    # PSI-BLAST por proteína -> PSSMs  (~35 min)
uv run cb396 features    # PSSM + janela 13 -> X (61316, 260)
uv run cb396 evaluate    # validação cruzada (Linear SVM)
uv run cb396 train       # modelo final (.joblib)
uv run cb396 summary     # gera experiment_summary.json

# ou tudo de uma vez:
uv run cb396 all
```

Use `--data-dir <dir>` para trocar o diretório base (padrão: `data`).

Depois que as features (`data/artifacts/X_pssm_window13.npy`, `y_q3.npy`,
`groups.npy`) existem, é possível reavaliar/treinar sem refazer o PSI-BLAST:

```bash
uv run cb396 evaluate            # Linear SVM, GroupKFold(7) -> Q3
uv run cb396 train               # treina e salva o modelo final (.joblib)
uv run cb396 evaluate --model rbf  # SVM RBF (amostrado, predict em lotes)
```

### CNN 1D (opcional, requer PyTorch)

```bash
uv sync --extra cnn              # instala torch (CPU)
uv run cb396 train-cnn            # CNN; GroupKFold(7) + modelo final
uv run cb396 train-cnn-bilstm     # CNN + BiLSTM (melhor resultado)
uv run cb396 train-cnn-bilstm-crf # CNN + BiLSTM + CRF
```

Os modelos reaproveitam as PSSMs (ou as reconstroem de `X` quando os `.pssm`
brutos não estão presentes) e predizem a proteína inteira de uma vez.

## Estrutura

```
src/cb396/          # pacote
  config.py         # caminhos e constantes
  dataset.py        # download + parsing + validação do CB396
  blastdb.py        # download do Swiss-Prot
  pssm.py           # PSI-BLAST + parsing de PSSM
  features.py       # janela deslizante -> X/y/groups
  train.py          # SVMs, GroupKFold, modelo final
  cnn.py            # CNN / CNN+BiLSTM / CNN+BiLSTM+CRF
  cli.py            # interface de linha de comando
tests/              # testes unitários (sem rede nem BLAST)
data/               # dados e artefatos gerados pelo pipeline
```

## Notas

- `scikit-learn` está fixado em `>=1.6,<1.7` para casar com o `.joblib` salvo
  (treinado em 1.6.1) e evitar `InconsistentVersionWarning`.
- O PSI-BLAST é o passo caro (~35 min) e exige `ncbi-blast+` no sistema.
