# Apollo Solutions Machine Learning Developer Test

**Observação**: *Para desenvolvimento do projeto, foi utilizado um disposito com SO Windows 11, através do WSL (Ubuntu 20.04), com desenvolvimento na IDE VsCode com linguagem de programação Python*

Autor : Hyago Vieira Lemes Barbosa Silva

# Estrutura do projeto

```bash
├── data/
├────── __init__.py                 # Arquivo init padrão para comunicação dos dados entre as pastas Python
├────── mini_gm_public_v0.p
├── __init__.py                     # Arquivo init padrão para comunicação dos dados entre as pastas Python
├── main.py                         # Script principal que orquestra o pipeline
├── data_processing.py              # Funções para carregar e preprocessar os dados
├── visualization.py                # Funções para redução de dimensionalidade e geração de gráficos
├── classification.py               # Implementação do classificador KNN, validação cruzada e seleção de k
├── metrics_utils.py                # Funções para calcular AUC, F1-Score, Top‑k Accuracy e gerar tabelas/curvas
├── requirements.txt                # Lista de dependências (ex.: numpy, scikit-learn, matplotlib, etc.)
├── README.md                       # Instruções de execução e explicação do projeto
├── report.pdf                      # Relatório final (a ser gerado com a documentação completa)
├── ML Junior Practical Test.docx   # Documento de teste para empresa APOLLO
└── interpretacao.pdf               # PDF respondendo as questões de interpretação

```

# Passo 1
 
- Instalar o anaconda no site oficial. (https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh)
- Siga as instruções de exemplo deste GitHub (https://gist.github.com/kauffmanes/5e74916617f9993bc3479f401dfec7da)
- Após a instalação é necessário criar um ambiente virtual com anaconda, utilizando python>=3.10 e ativar o ambiente virtual.

`conda create -n <nome_do_ambiente_virtual> python=3.10`

`conda activate <nome_do_ambiente_virtual>`

# Passo 2

- Baixar o projeto

`git clone https://github.com/HyAgOsK/APOLLO_DEV.git`

# Passo 3

- Instalar as dependencias

`cd APOLLO_DEV`
`pip install -r requirements.txt`

**Observação**: Algumas libs já estão instaladas através do conda, quando foi iniciado o ambiente virtual, para ver as demais libs que necessitam para seu ambiente
basta executar o comando `pip freeze` e você vera todas libs que existem e sáo pré existentes na criação do seu ambiente virtual. Após isso, comparando, você instala as demais libs que faltam em seu ambiente virtual. 

# Passo 4

- Executar o projeto, e obter os `outputs.../`

`cd APOLLO_DEV`
`python main.py`

**Observação**: Após executar o projeto, serão criadas 2 pastas, uma pasta é referente aos dados originais.

- O nome da primeira pasta por padrão é `outputs_original_data_seed<semente_de_reprodutibilidade>`
- O nome da segunda pasta, é `outputs_balanced_data_seed<semente_de_reprodutibilidade>`


A `<semente_de_reprodutibilidade>` pode ser configurada na main, dentro do código, onde você pode alterar para conseguir obter reprodutibilidade dos resutados, anteriores.

Estes resultados atuais são para semente `seed=0`. 

Mas caso modifique, irão criar outras pastas, e esta semente de reprodutibilidade esta atrelada ao coeficientes de inicialização para curva de erro que irá ter o aprendizado 
durante o treinamento com knn classification. Caso altere, sempre tera um resultado pior ou melhor, não existe receita mágica qual melhor seed específica. Mas o resultado interessante encontrado foi com seed=0.
