# Ruolo e compito

A partire dalle descrizioni di due persone che ti fornirò, ogni volta che ti invierò una configurazione degli iperparametri il tuo compito sarà quello di:
1. scegliere un evento in base gli iperparametri della chat dati in input
2. generare una chat che sia coerente con
   1. le loro personalità fornite
   2. con la fase fornita e l'evento scelto
   3. con gli iperparametri forniti
3. fornire una spiegazione narrativa dettagliata dell'escalation emotiva di entrambi i personaggi, simile a quella che potrebbe fare un esperto di psicologia delle relazioni e delle dinamiche interpersonali, in cui si evidenzia:
   1. il possibile impatto che ciascun messaggio può aver dato sulla conversazione complessiva, sullo stato emotivo e sentimentale dei personaggi facendo riferimento alle dinamiche emerse nella chat.
   2. le possibili cause di un determinato comportamento, evidenziando le motivazioni e le emozioni dei personaggi.
   3. i possibili effetti di un determinato comportamento, evidenziando l'impatto sulla conversazione complessiva e sulle emozioni dei personaggi.

Lo scopo della spiegazione non è quello di analizzare i personaggi in sè, ma piuttosto quello di:
1. Spiegare perchè sono state assegnate determinate polarità ai messaggi.
2. Fornire un'interpretazione qualitativa narrativa dei sentimenti espressi e delle emozioni provate dai personaggi.
3. Riflettere sulle cause e conseguenze dei comportamenti accaduti in chat al fine di analizzare l'impatto che questi comportamenti emersi hanno sulle dinamiche relazionali, emotive e sentimentali dei personaggi.

L'idea è quella di costruire un dataset di chat con polarità e spiegazioni che possano essere utilizzate per addestrare modelli di intelligenza artificiale in grado di:
1. Riconoscere le emozioni assegnando delle polarità ai messaggi.
2. Fornire spiegazioni narrative che possano essere utilizzate per migliorare la comprensione delle interazioni tra i personaggi, le loro emozioni, i loro sentimenti e le dinamiche relazionali.

# Vincoli di output

1. La chat di output deve essere in formato txt
2. L'intera chat deve essere preceduta dalla seguente stringa formattata:
    ```
    Evento: {evento}\n\n
    ```
dove l'evento deve essere scelto tra quelli descritti nelle linee guida riportate di seguito
3. Ogni messaggio deve seguire l'espressione regolare "(?P<message>(?P<timestamp>\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d) | (?P<name>.+):\n(?P<content>.+)\nPolarity: (?P<polarity>(?:-?|\+?)\d\.?\d?\d?))\n\n" dove:
   1. Il timestamp deve essere nel formato “YYYY-MM-DD HH:MM:SS”.
   2. Il nome deve essere uno dei due personaggi forniti.
   3. Il messaggio deve essere realistico e deve riflettere le dinamiche, le emozioni e le interazioni tra i due personaggi.
   4. La polarità deve essere un numero float compreso nell'intervallo [-1, 1] che riflette il sentimento generale del messaggio, dove:
      1. Un valore pari a 1 indica un messaggio pieno di sentimento positivo.
      2. Un valore pari a -1 indica un messaggio pieno di sentimento negativo.
      3. Un valore pari a 0 indica un messaggio neutro.
4. Alla fine della chat, aggiungi la spiegazione formattata nel seguente modo:
    ```
    Spiegazione:
    {spiegazione}
    ```
5. La spiegazione deve:
    1. evitare di iniziare con un riferimento diretto alla fase attuale scelta della relazione. Esempi di inizio da evitare sono "Nella fase X della relazione...", "In questa fase della relazione...", "Questa chat illustra l'inizio di una relazione nella fase di scoperta...".
    2. essere rigorosamente narrativa, simile a quella che un esperto di psicologia potrebbe produrre.
    3. essere dettagliata, approfondita e profonda, evitando di essere generica o superficiale.
    4. evitare di far riferimento agli iperparametri scelti.
    5. evitare di fare un esplicito riferimento alle polarità numeriche dei messaggi, ma piuttosto dare un'interpretazione qualitativa narrativa dei sentimenti espressi dai personaggi.
    6. evitare espressioni come "tipica della sua personalità", "comportamento tipico", ma piuttosto deve preferire espressioni tipiche di un esperto psicologo che non conosce le personalità interne dei personaggi ma cerca di analizzare, studiare e comprendere i loro comportamenti come "in questo caso, X ha mostrato un comportamento che può essere interpretato come...", " Y ha reagito in un modo che può essere visto come..." etc.
 6. Se vuoi citare qualche frase nella spiegazione, non ricopiare l'intera frase, ma estrapola solo la parte più significativa e contestualizzala all'interno della tua analisi.
