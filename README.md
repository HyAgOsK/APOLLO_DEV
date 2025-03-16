# Apollo Solutions - Machine Learning Developer Test

**Autor:** Hyago Vieira Lemes Barbosa Silva  
**Ambiente de Desenvolvimento:** Windows 11 via WSL (Ubuntu 20.04)  
**IDE:** VSCode  
**Linguagem de Programação:** Python  

---

## 📁 Estrutura do Projeto

```bash
├── data/
│   ├── __init__.py                          # Arquivo init para comunicação entre arquivos python em diferentes repositorios
│   ├── mini_gm_public_v0.p                  # Dados de entrada do modelo
├── __init__.py                              # Arquivo init para comunicação entre arquivos python em diferentes repositorios
├── main.py                                  # Script principal do pipeline
├── data_processing.py                       # Funções para carregamento e pré-processamento de dados
├── visualization.py                         # Funções para redução de dimensionalidade e geração de gráficos
├── classification.py                        # Implementação do classificador KNN e seleção de hiperparâmetros
├── metrics_utils.py                         # Funções para cálculo de métricas (AUC, F1-Score, Top-k Accuracy)
├── multiple_testing_augmentation_dataset.py # Funções para aumento de dados, análise descritiva dos aumentos de dados (NOISE, MIXUP, SMOTE)
├── requirements.txt                         # Lista de dependências
├── README.md                                # Documentação do projeto
├── Report.pdf                               # Relatório final
├── ML_Junior_Practical_Test.docx            # Documento de teste da Apollo Solutions
└── interpretation.pdf                       # Respostas das questões de interpretação
```

---

## 🔧 Passo a Passo para Execução

### 🛠️ Passo 1: Instalar Anaconda

1. Baixe o instalador do Anaconda:
   ```bash
   wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
   ```
2. Siga as instruções de instalação do anaconda
   **Guia de Instalação do WSL e Anaconda no Windows**

   ### 1. Instalar o WSL (Windows Subsystem for Linux)
   Abra o **Prompt de Comando** ou **PowerShell** como administrador e execute:

   ```bash
   wsl --install
   ```

   Se a instalação falhar, instale o Ubuntu manualmente pela **Microsoft Store**. Após a instalação, abra o Ubuntu para concluir a configuração inicial.

   ### 2. Baixar o Anaconda
   Acesse o site oficial do **Anaconda** e faça o download do arquivo `.sh` correspondente ao seu sistema:
   - **Versão**: Linux x64 `.sh`

   ### 3. Atualizar os pacotes do WSL
   Após instalar o WSL e abrir o terminal Ubuntu, execute os seguintes comandos para atualizar os pacotes:

   ```bash
   sudo apt-get update
   sudo apt-get upgrade -y
   ```

   ### 4. Instalar Python3 e Pip
   Para garantir que o Python e o gerenciador de pacotes `pip` estejam instalados, execute:

   ```bash
   sudo apt-get install python3-pip -y

   
   ```
   **Observação**: O pip do projeto que estou usando possui a versão pip 25.0. Caso instale outra versão sugiro alterar para esta.

   ### 5. Executar o WSL no Windows
   Abra um terminal no **PowerShell** e digite:

   ```bash
   wsl
   ```

   Abra uma nova aba e digite novamente `wsl`, ou navegue até o diretório onde o Anaconda foi baixado:

   ```bash
   cd /mnt/c/<seu_usuario>/Downloads
   ```

   Verifique a presença do arquivo do **Anaconda** (`Anaconda3-2024.06-1-Linux-x86_64.sh`). Para instalá-lo, execute:

   ```bash
   bash Anaconda3-2024.06-1-Linux-x86_64.sh -b -u -p ~/anaconda3
   ```

   ### 6. Verificar a instalação e inicializar o Anaconda
   Após a instalação, volte para o diretório **root** e verifique se o Anaconda foi instalado corretamente:

   ```bash
   cd
   ls
   ~/anaconda3/bin/conda init bash
   ~/anaconda3/bin/conda init zsh
   ```

   Agora, edite o arquivo `.bashrc` para confirmar a configuração do Anaconda:

   ```bash
   nano .bashrc
   ```

   Dentro do arquivo, procure pela entrada indicando a instalação do Anaconda. Se a pasta `anaconda3` estiver presente, a instalação foi concluída com sucesso.

   ---


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

**Observação:** Caso tenha o arquivo zip, basta descompacta-lo, e entrar na pasta principal `APOLLO_DEV-main/`
---

### 📦 Passo 3: Instalar Dependências

```bash
cd APOLLO_DEV
pip install -r requirements.txt
```

**Observação:** Se houver algum problema de dependencias, verifique o python e pip instalado, conforme comentei, o ambiente virtual usado é anaconda, com WSL Ubuntu. A versão do pip 25.0. Python 3.10.

---

### 🚀 Passo 4: Executar o Projeto

```bash
cd APOLLO_DEV
python main.py --pickle_file ./data/mini_gm_public_v0.1.p
```

Após a execução, quatro pastas de saída serão criadas:

- `outputs_original_data_seed<semente>` - Dados originais
- `outputs_balanced_data_seed<semente>` - Dados balanceados
- `outputs_original_data_mixup_seed<semente>` - Dados MIXUP (originais + artificiais)
- `outputs_balanced_data_mixup_seed<semente>` - Dados MIXUP (originais + artificiais) balanceados


A **semente de reprodutibilidade** (`seed`) pode ser configurada no `main.py`. Alterá-la influencia os coeficientes de inicialização do classificador KNN, podendo impactar o desempenho do modelo. Para este projeto, a semente utilizada foi `seed=0`. 

---

## 📈 Resultados

- Os resultados, discussões e toda análise do projeto, está no arquivo Report.pdf
- As respostas para as perguntas de interpretação, estão no arquivo Interpretation.pdf

## 📝 Conclusão

Este projeto implementa um classificador KNN para um conjunto de dados específico, com validação cruzada e ajuste de hiperparâmetros. Os resultados mais promissores foram com base de dados balanceadas, com normalização L2 aplicada, aos dados. Mixup também foi grande satisfatória, porém não se sabe sobre as imagens verdadeiras apenas os embeddings das imagens, com isso apenas as características, é complicado ainda mais por se tratar de imagens de síndrome, são delicadas as características, ou seja, qualquer modificação artificial pode gerar uma anomalia, ou seja, artificialmente é melhor fazer isso nas imagens originais, aplicando sim MIXUP, GANS, PIX2PIX, PATHOLOGY GAN, DATA-AGUMENTATION. Enfim.

Estou muito satisfeito com projeto e fico muito feliz em poder participar!

📌 **Autor:** Hyago Vieira Lemes Barbosa Silva

