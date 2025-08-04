# All Possible Dataset Generation Choices

p1 p2 # persone
b1 b2 # backgrounds
\  /
 st   # storia
 rl   # descrizione della relazione
  |
chat

generare nuove chat in cui non si da una spiegazione ad ogni messaggio ma si da una spiegazione all'intera chat dove vengono spiegate le polarità dei messaggi e le polarità della relazione tra le due persone.

# All Possible Task Design Choices

## Message Token Separators

1. Only a unified "[SEP]" token
2. "[USR1]" and "[USR2]" tokens to separate messages from different users
**POSSIBLE ADDITIONAL IDEA**: Using `token_type_ids` with additional embeddings (other than token and positional embeddings like the ones in every transformer architecture) to distinguish between user 1 messages, with 0, and user 2 messages, with 1

## Task Output Choices

predictions    explanations   is_feasible
1              1              1/2 x
N              N              1
N              1              0   x
1              N              1/2 x

Siccome la predizione è contestualizzata all'intera chat, molto probabilmente è più conveniente dare una singola spiegazione per tutte le predizioni o in generale per l'intera chat invece che suddividerla per ogni singolo messaggio. Questa supposizione è anche supportata dal fatto che modelli tipo BART sono stati progettati per generare spiegazioni contestualizzate all'intera sequenza di input e lavorerebbero meglio in questi contesti. Si sconsiglia dunque di seguire tutte le strade che prevedono di dare una spiegazione per ogni singolo messaggio. Riteniamo, dunque, che spiegazioni localizzate non abbiano senso.

## Contextualized Prediction for each message in a chat

1. Multiple Predictions at Once
   1. Input: chat with messages separated by separation methods
   2. Output: predictions for each separated message
2. Only One Prediction at Once
   1. Input: chat with messages separated by separation methods and with a target "[CLS]" token positioned at the message to predict
   2. Output: prediction for that message

## Possible Models

1. BERT for only predictions: classic BERT model with an additional regression/classification head. Separation and target tokens are used directly from the loaded vocabulary embeddings and they are not new. They're are simply used and positioned in different ways. Possible checkpoints are:
   1. dbmdz/bert-base-cased
   2. dbmdz/bert-base-italian-xxl-cased
2. BART for only explanations: classic BART model with or without additional separation and target tokens. The target token is used to position the explanation in the output sequence. Possible checkpoints are:
   1. morenolq/bart-it
3. (Multi-Task) BART: pre-trained BART with additional embeddings and with an additional regression/classification head. Possible checkpoints are:
   1. facebook/mbart-large-50
   2. morenolq/bart-it

# Idee

1. La visualizzazione della distribuzione dei dati generati (coppie di persone, scenari e chats) può essere un task futuro per meglio investigare la qualità del dataset generato e la fattibilità di continuare in questa direzione. Ne consegue che lo scopo di questo studio è un'investigazione più approfondita della fattibilità di usare LLM per generare dataset di conversazioni tra persone, piuttosto che un dataset di conversazioni tra persone reali. (La visualizzazione può essere fatta chiedendo ad LLMs di creare embeddings delle coppie delle persone, delle persone singole e degli scenari. Altri modi potrebbero consistere nel chiedere ad LLMs di estrarre informazioni in formato strutturato in modo da creare grafici, histogrammi, spider charts ecc. per visualizzare le distribuzioni dei dati generati. Esempio: estrarre età o geolocalizzazione con istogrammi, BIG 5 traits con spider charts ecc.)

2. Ci sono tre possibili modi in cui il modello può essere usato:
   1. L'utente può soltanto spostare e ridimensionare una finestra (sequenza) di messaggi CONSECUTIVI in una chat. (Idealmente si suppone che dataset, modello e codice di training siano abbastanza adatti per supportare questa funzionalità senza modifiche.)
   2. L'utente può scegliere una qualsiasi sequenza ANCHE NON CONSECUTIVA di messaggi in una chat. (In tal caso l'utente può anche scegliere messaggi non sempre correlati fra loro. E' necessario modificare solo il dataset e fare data augmentation ma il modello e il codice di training non devono essere modificati.)
   3. L'utente sceglie un messaggio target e il modello in automatico ritrova i messaggi più correlati per predire e spiegare al meglio la sua polarità. (Questa soluzione implica la presenza di un modello di retrieval che trova i messaggi più correlati in una chat, e quindi la modifica del dataset, del modello e del codice di training per supportare questa funzionalità.) (FUNZIONALITA' FUTURA)