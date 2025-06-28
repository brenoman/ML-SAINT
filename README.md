# SAINT vs Gradient Boosting & AutoML for Tabular Data

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Repositório oficial do projeto comparando o modelo **SAINT (Self-Attention Transformer)** com modelos tradicionais (LightGBM, XGBoost, CatBoost) e ferramentas AutoML (AutoGluon, auto-sklearn) em 30 datasets tabulares do OpenML-CC18.

## 🔍 Contexto
Trabalho final da disciplina de Aprendizagem de Máquina (2025) com objetivo de:
- Implementar e avaliar o SAINT - modelo Transformer para dados tabulares
- Comparar com benchmarks de Gradient Boosting e AutoML
- Aplicar protocolo estatístico rigoroso (teste de Demšar)

## 🚀 Principais Resultados
- ✅ **AutoGluon** obteve melhor performance geral (acurácia)
- ⚡ **LightGBM** foi o mais eficiente (tempo de execução)
- 🧠 **SAINT** mostrou potencial competitivo, especialmente em AUC-OVO
- 📊 CatBoost teve melhor generalização (menor overfitting)

## 📊 Métricas Analisadas
- Acurácia (Accuracy)
- AUC One-vs-One (AUC-OVO)
- Entropia Cruzada (Cross-Entropy)
- Tempo de execução

## 🛠️ Como Reproduzir
1. Clone o repositório:
   ```bash
   git clone https://github.com/brenoman/ML-SAINT.git
2. Instale as dependências:
    ```bash
    pip install -r requirements.txt   
3. Execute os scripts (veja detalhes na documentação)

## 📄 Documentação Completa
- [Relatório Técnico](/docs/relatorio_final.pdf)
- [Apresentação](/docs/apresentacao.pdf)
