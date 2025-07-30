# Ruolo e compito

A partire dalle descrizioni di due persone che ti fornirò, ogni volta che ti invierò una configurazione degli iperparametri il tuo compito sarà quello di:
1. scegliere un evento in base gli iperparametri della chat dati in input
2. generare una chat che sia coerente con
   1. le loro personalità fornite
   2. con la fase fornita e l'evento scelto
   3. con gli iperparametri forniti
3. analizzare per ciascun messaggio il sentimento espresso e assegnare una polarità numerica compresa nell'intervallo [-1, 1] che rifletta il sentimento generale del messaggio, dove:
   1. Un valore pari a 1 indica un messaggio pieno di sentimento positivo.
   2. Un valore pari a -1 indica un messaggio pieno di sentimento negativo.
   3. Un valore pari a 0 indica un messaggio neutro.
   Questa polarità rappresenta il sentimento generale del messaggio, tenendo conto delle emozioni e delle dinamiche relazionali tra i personaggi.

# Vincoli di output

1. La chat di output deve essere in formato txt
2. L'intera chat deve essere preceduta dalla seguente stringa formattata:
    ```
    Evento: {evento}\n\n
    ```
dove l'evento deve essere scelto tra quelli descritti nelle linee guida riportate di seguito
3. Ogni messaggio deve seguire l'espressione regolare:
   ```python
   r"""
   (?P<message>(?P<timestamp>\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d) | (?P<name>.+):\n(?P<content>.+)\nPolarity: (?P<polarity>(?:-?|\+?)\d\.?\d?\d?))\n\n
   """
   ```
   dove:
   1. Il timestamp deve essere nel formato “YYYY-MM-DD HH:MM:SS”.
   2. Il nome deve essere uno dei due personaggi forniti.
   3. Il messaggio deve essere realistico e deve riflettere le dinamiche, le emozioni e le interazioni tra i due personaggi.
