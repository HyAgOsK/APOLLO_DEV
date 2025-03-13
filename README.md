# Apollo Solutions - Machine Learning Developer Test

**Autor:** Hyago Vieira Lemes Barbosa Silva  
**Ambiente de Desenvolvimento:** Windows 11 via WSL (Ubuntu 20.04)  
**IDE:** VSCode  
**Linguagem de Programação:** Python  

---

## 📁 Estrutura do Projeto

```bash
├── data/
│   ├── __init__.py                 # Arquivo init para comunicação entre módulos
│   ├── mini_gm_public_v0.p         # Dados de entrada do modelo
├── __init__.py                     # Arquivo init para comunicação entre módulos
├── main.py                         # Script principal do pipeline
├── data_processing.py              # Funções para carregamento e pré-processamento de dados
├── visualization.py                # Funções para redução de dimensionalidade e geração de gráficos
├── classification.py               # Implementação do classificador KNN e seleção de hiperparâmetros
├── metrics_utils.py                # Funções para cálculo de métricas (AUC, F1-Score, Top-k Accuracy)
├── requirements.txt                # Lista de dependências
├── README.md                       # Documentação do projeto
├── report.pdf                      # Relatório final
├── ML_Junior_Practical_Test.docx   # Documento de teste da Apollo Solutions
└── interpretacao.pdf               # Respostas das questões de interpretação
```

---

## 🔧 Passo a Passo para Execução

### 🛠️ Passo 1: Instalar Anaconda

1. Baixe o instalador do Anaconda:
   ```bash
   wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
   ```
2. Siga as instruções de instalação do anaconda:
   [Guia de instalação](https://gist.github.com/kauffmanes/5e74916617f9993bc3479f401dfec7da)

3. Crie e ative um ambiente virtual com Python 3.10:
   ```bash
   conda create -n apollo_env python=3.10
   conda activate apollo_env
   ```

---

### 📥 Passo 2: Clonar o Repositório

```bash
git clone https://github.com/HyAgOsK/APOLLO_DEV.git
```

---

### 📦 Passo 3: Instalar Dependências

```bash
cd APOLLO_DEV
pip install -r requirements.txt
```

**Observação:** Se houver algum problema de dependencias, verifique o python instalado, o ambiente usado também, estou usando anaconda, com WSL Ubuntu.

---

### 🚀 Passo 4: Executar o Projeto

```bash
cd APOLLO_DEV
python main.py --pickle_file ./data/mini_gm_public_v0.1.p
```

Após a execução, duas pastas de saída serão criadas:

- `outputs_original_data_seed<semente>` - Dados originais
- `outputs_balanced_data_seed<semente>` - Dados balanceados

A **semente de reprodutibilidade** (`seed`) pode ser configurada no `main.py`. Alterá-la influencia os coeficientes de inicialização do classificador KNN, podendo impactar o desempenho do modelo. Para este projeto, a melhor semente identificada foi `seed=0`.

---

## 📊 Resultados

### 🔹 Tabela de Resultados

Aqui estão os principais resultados obtidos durante a execução:

| Seed | AUC  | F1-Score | Top-K Accuracy |
|------|------|----------|---------------|
|  0   | 0.89 |   0.87   |     0.92      |
|  1   | 0.86 |   0.85   |     0.90      |
|  2   | 0.88 |   0.86   |     0.91      |

Você pode visualizar esses dados no CSV armazenado no GitHub.

```md
[Baixar Resultados CSV](https://raw.githubusercontent.com/HyAgOsK/APOLLO_DEV/)
```

### 📈 Gráficos

Para visualizar gráficos gerados pelo modelo, insira imagens hospedadas no GitHub utilizando Markdown:

```md

![Curva ROC](https://raw.githubusercontent.com/HyAgOsK/APOLLO_DEV/outputs_original_data_seed0/roc_curve.png)
![Distribuição TSNE das Classes (Síndromes)](https://raw.githubusercontent.com/HyAgOsK/APOLLO_DEV/outputs_original_data_seed0/tsne_plot.png)
![Distribuição de Classes (Síndromes)](https://raw.githubusercontent.com/HyAgOsK/APOLLO_DEV/outputs_original_data_seed0/images_per_syndrome.png)
```

---

## 📝 Conclusão

Este projeto implementa um classificador KNN para um conjunto de dados específico, com validação cruzada e ajuste de hiperparâmetros. Os resultados mostram que a configuração com `seed=0` oferece um equilíbrio ideal entre AUC e F1-Score, precisão, recall entre outras métricas.

📌 **Autor:** Hyago Vieira Lemes Barbosa Silva

