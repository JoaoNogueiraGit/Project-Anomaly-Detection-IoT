# 🛡️ Plataforma de Detecção de Anomalias em Redes IoT com Aprendizagem Máquina

Este projeto visa o desenvolvimento de uma plataforma prática para deteção de anomalias e padrões suspeitos em tráfego IoT, recorrendo a métodos de aprendizagem automática supervisionada e não supervisionada. A solução recolhe e processa tráfego de rede simulado  para identificar ciberataques contra protocolos críticos e redes tradicionais.

## 📊 Fontes de Dados e Datasets

Para garantir uma cobertura abrangente, o treino e validação dos modelos de inteligência artificial assentam em três pilares distintos de dados:

### 1. IoT / Domótica: Dataset MQTT-IoT-IDS2020
Focado em ataques contra o protocolo de mensagens MQTT, amplamente utilizado em dispositivos IoT. Utilizamos os ficheiros de fluxos bidirecionais (`biflow`) para otimizar a extração de padrões de comunicação completa.
* 🔗 **Link:** [IEEE DataPort - MQTT Dataset](https://ieee-dataport.org/open-access/mqtt-iot-ids2020-mqtt-internet-things-intrusion-detection-dataset)

| Ficheiro | Tamanho | Descrição / Classe |
| :--- | :--- | :--- |
| `biflow_normal.csv` | 25.22 MB | **Tráfego Benigno:** Estabelece a *baseline* de comportamento legítimo da rede. |
| `biflow_mqtt_bruteforce.csv` | 4.46 MB | **Ataque MQTT:** Tentativas de adivinhação de credenciais (*Brute Force*). |
| `biflow_scan_A.csv` | 4.14 MB | **Reconhecimento:** Ataque de *scan* de portas de forma agressiva. |
| `biflow_scan_sU.csv` | 7.83 MB | **Reconhecimento UDP:** *Scanning* focado especificamente em portas UDP. |
| `biflow_sparta.csv` | 26.99 MB | **Ataque Composto:** Tráfego gerado pela ferramenta Sparta (*scanning* + *brute-force*). |

### 2. Tráfego de Rede Geral: Dataset CICIDS2017
Fornece um "vocabulário" diversificado de ciberataques clássicos (ex: DoS, Web Attacks, Infiltração) e tráfego benigno atualizado, essencial para a deteção de anomalias em infraestruturas de rede de suporte à IoT.
* 🔗 **Link:** *(A adicionar)*
* **Ficheiros Selecionados:** *(A definir durante a fase de Análise Exploratória)*

### 3. IoT Industrial (IIoT): Dataset Modbus
Focado em tráfego SCADA para deteção de anomalias em ambientes industriais, onde o protocolo Modbus é o standard.
* 🔗 **Link:** *(A adicionar)*
* **Ficheiros Selecionados:** *(A definir)*

---

## ⚙️ Configuração Local e Organização de Dados

**Aviso de Controlo de Versões:** Devido ao tamanho elevado dos datasets, os ficheiros `.csv` originais **não** estão incluídos neste repositório.

Para executar o projeto localmente e simular o tráfego:
1. Crie uma pasta `data/raw/` na raiz do projeto, com subpastas para cada protocolo (ex: `data/raw/MQTT/`, `data/raw/CICIDS2017/`).
2. Descarregue os ficheiros através dos links oficiais acima.
3. Coloque os ficheiros nas respetivas subpastas.
4. Certifique-se de que a pasta `data/` está incluída no seu ficheiro `.gitignore`.

## 🛠️ Stack Tecnológica Preliminar
* **Linguagem:** Python
* **Processamento e ML:** Pandas, Scikit-learn (k-Means, Isolation Forest, Random Forest) 
* **Interface Web:** *(A definir)*