1. cost sensitive learning
2. baseline con random predictor o altri tipi di baselines
3. kappa statistic con una baseline
4. plottare le curve roc, auc, precision-recall, etc.

5. BERT: aggiungere le varie altre loss per regressione e il correlation coefficient

# Altri possibili modelli classici di machine learning da poter usare

1. K-Nearest Neighbors (KNN), Edited Nearest Neighbors (ENN), Condensed Nearest Neighbors (CNN). Possibili idee:
   1. usare clustering per creare rappresentanti di cluster e diminuire il numero di campioni da usare per l'inferenza e il training
2. Logistic Regression o Multi-Layer Perceptron (MLP) con custom cost loss training

# Altre feature engineering e pre-processing da poter fare

1. (Text -> Pre-Processed Text -> Mean word2vec embedding) (offline) -> Model (online)
2. (Text -> Knowledge Graph -> KG Embeddings) (offline) -> Model (online)

# All Possible Dataset Generation Choices

p1 p2 # persone
b1 b2 # backgrounds
\  /
 st   # storia
 rl   # descrizione della relazione
  |
chat

generare nuove chat in cui non si da una spiegazione ad ogni messaggio ma si da una spiegazione all'intera chat dove vengono spiegate le polarità dei messaggi e le polarità della relazione tra le due persone.

# Idee

1. La visualizzazione della distribuzione dei dati generati (coppie di persone, scenari e chats) può essere un task futuro per meglio investigare la qualità del dataset generato e la fattibilità di continuare in questa direzione. Ne consegue che lo scopo di questo studio è un'investigazione più approfondita della fattibilità di usare LLM per generare dataset di conversazioni tra persone, piuttosto che un dataset di conversazioni tra persone reali. (La visualizzazione può essere fatta chiedendo ad LLMs di creare embeddings delle coppie delle persone, delle persone singole e degli scenari. Altri modi potrebbero consistere nel chiedere ad LLMs di estrarre informazioni in formato strutturato in modo da creare grafici, histogrammi, spider charts ecc. per visualizzare le distribuzioni dei dati generati. Esempio: estrarre età o geolocalizzazione con istogrammi, BIG 5 traits con spider charts ecc.)

2. Ci sono tre possibili modi in cui il modello può essere usato:
   1. L'utente può soltanto spostare e ridimensionare una finestra (sequenza) di messaggi CONSECUTIVI in una chat. (Idealmente si suppone che dataset, modello e codice di training siano abbastanza adatti per supportare questa funzionalità senza modifiche.)
   2. L'utente può scegliere una qualsiasi sequenza ANCHE NON CONSECUTIVA di messaggi in una chat. (In tal caso l'utente può anche scegliere messaggi non sempre correlati fra loro. E' necessario modificare solo il dataset e fare data augmentation ma il modello e il codice di training non devono essere modificati.)
   3. L'utente sceglie un messaggio target e il modello in automatico ritrova i messaggi più correlati per predire e spiegare al meglio la sua polarità. (Questa soluzione implica la presenza di un modello di retrieval che trova i messaggi più correlati in una chat, e quindi la modifica del dataset, del modello e del codice di training per supportare questa funzionalità.) (FUNZIONALITA' FUTURA)