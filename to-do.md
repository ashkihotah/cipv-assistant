# Dubbi

p1 p2 # persone
b1 b2 # backgrounds
\  /
 st   # storia
 rl   # descrizione della relazione
  |
chat


# Idee

1. La visualizzazione della distribuzione dei dati generati (coppie di persone, scenari e chats) può essere un task futuro per meglio investigare la qualità del dataset generato e la fattibilità di continuare in questa direzione. Ne consegue che lo scopo di questo studio è un'investigazione più approfondita della fattibilità di usare LLM per generare dataset di conversazioni tra persone, piuttosto che un dataset di conversazioni tra persone reali. (La visualizzazione può essere fatta chiedendo ad LLMs di creare embeddings delle coppie delle persone, delle persone singole e degli scenari. Altri modi potrebbero consistere nel chiedere ad LLMs di estrarre informazioni in formato strutturato in modo da creare grafici, histogrammi, spider charts ecc. per visualizzare le distribuzioni dei dati generati. Esempio: estrarre età o geolocalizzazione con istogrammi, BIG 5 traits con spider charts ecc.)

# Bug:

1. Se una persona fa stone-walling il tag stone-walling lo mette al messaggio successivo che può essere anche dell'altra persona comunicante. Questo è sbagliato, 