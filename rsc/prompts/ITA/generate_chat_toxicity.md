# Ruolo e compito

A partire dalle descrizioni di due persone che ti fornirò, ogni volta che ti invierò una configurazione degli iperparametri il tuo compito sarà quello di:
1. scegliere un evento in base gli iperparametri della chat dati in input
2. generare una chat che sia coerente con
   1. le loro personalità fornite
   2. con la fase fornita e l'evento scelto
   3. con gli iperparametri forniti
3. analizzare per ciascun messaggio il livello di tossicità e assegnare una polarità numerica compresa nell'intervallo [-1, 1] che rifletta quanto il messggio sia tossico, dove:
   1. Un valore pari a 1 indica un messaggio completamente sano.
   2. Un valore pari a -1 indica un messaggio completamente tossico.
   3. Un valore pari a 0 indica un messaggio neutro.
   Questa polarità rappresenta il livello di tossicità generale del messaggio, tenendo conto delle emozioni, dei sentimenti e delle dinamiche relazionali tra i personaggi.

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

# Guida alle Fasi del Ciclo di Vita di una Relazione

Questa guida descrive le tappe evolutive tipiche di una relazione romantica. Ogni relazione è unica, e le fasi possono avere durate diverse, sovrapporsi o, in alcuni casi, essere saltate. Lo scopo è fornire un modello psicologicamente plausibile per generare interazioni e dialoghi realistici in chat.

## Flirt e Conoscenza Iniziale

### 1. Descrizione e Caratteristiche

Questa è la fase embrionale, caratterizzata da **attrazione, curiosità e incertezza**. L'interazione è guidata dal desiderio di conoscere l'altro e, contemporaneamente, di presentare la migliore versione di sé stessi. La comunicazione è spesso leggera, giocosa e focalizzata sulla ricerca di affinità e punti in comune. La vulnerabilità è bassa e si tende a nascondere i propri difetti. L'ansia da prestazione ("gli piacerò?") e l'eccitazione per il potenziale futuro sono le emozioni dominanti.

* **Caratteristiche Principali:**
    * Idealizzazione iniziale.
    * Comunicazione attenta e calibrata.
    * Focus sugli aspetti positivi e sulle somiglianze.
    * Basso livello di conflitto.
    * Incertezza sulla reciprocità dei sentimenti.

### 2. Esempi di Eventi (Adattabili a una Chat)

* **Sani:**
    * **Post-primo appuntamento:** Uno dei due scrive per dire che è stato/a bene e per proporre di rivedersi.
        * `"Ehi, volevo solo dirti che sono stato benissimo ieri sera. Mi piacerebbe molto rivederti presto, se ti va :)"`
    * **Condivisione di interessi:** Inviarsi link, canzoni o meme che rispecchiano una conversazione avuta.
        * `"Ho visto questo video e mi ha fatto subito pensare a quello che mi raccontavi del tuo cane! 😂 [link]"`
    * **Organizzazione del secondo/terzo appuntamento:** La chat diventa il luogo per pianificare logisticamente l'incontro successivo.
        * `"Allora che dici, cinema o aperitivo giovedì? Fammi sapere tu cosa preferisci!"`

* **Tossici:**
    * **Love Bombing prematuro:** Dichiarazioni esagerate e inappropriate per la fase di conoscenza.
        * `"Non ho mai provato una cosa così in vita mia. Penso di essermi già innamorato di te."` (dopo due giorni)
    * **Controllo e gelosia ingiustificati:** Richieste pressanti sulla posizione o sulle attività dell'altro.
        * `"Sei online su WhatsApp ma non mi rispondi? Con chi stai parlando?"`
    * **Ghosting:** Interruzione improvvisa e totale delle comunicazioni dopo alcuni scambi positivi.
        * `(Nessun messaggio dopo una chat apparentemente piacevole la sera prima)`

* **Neutrali:**
    * **Domande conoscitive:** Le classiche domande per scoprire di più sull'altro.
        * `"Domanda a bruciapelo: qual è il viaggio più bello che hai mai fatto?"`
    * **Logistica del primo appuntamento:** Definire ora e luogo dell'incontro.
        * `"Perfetto, allora ci vediamo alle 20:30 davanti al locale. Se faccio tardi ti scrivo!"`
    * **Commenti sulla giornata:** Brevi aggiornamenti per mantenere il contatto.
        * `"Giornata pesantissima al lavoro oggi... non vedo l'ora che arrivi il weekend. Tu tutto bene?"`

## Infatuazione e Romanticismo

### 1. Descrizione e Caratteristiche

Questa è la fase dell'**innamoramento**, dominata da ormoni come dopamina e ossitocina. I partner si sentono una cosa sola, le differenze vengono minimizzate o percepite come affascinanti. L'idealizzazione raggiunge il suo apice: il partner è visto come perfetto. C'è un forte desiderio di fusione, di passare più tempo possibile insieme e di condividere ogni aspetto della propria vita. Il focus si sposta dall' "io" al "noi".

* **Caratteristiche Principali:**
    * Intensa passione e attrazione fisica.
    * Idealizzazione massima del partner.
    * Desiderio di simbiosi e fusione.
    * Comunicazione continua e molto affettuosa.
    * Tendenza a trascurare amici, hobby e altri aspetti della vita individuale.

### 2. Esempi di Eventi (Adattabili a una Chat)

* **Sani:**
    * **Pianificazione della prima fuga romantica:** Organizzare il primo weekend insieme.
        * `"Amore, ho visto un'offerta per un weekend alle terme. Che ne dici se scappiamo via sabato e domenica? Solo io e te."`
    * **Condivisione di vulnerabilità:** Raccontare per la prima volta in chat una paura o un segreto profondo, ricevendo supporto.
        * `"Non l'ho mai detto a nessuno, ma a volte ho una paura matta di fallire... Volevo solo che lo sapessi."`
    * **Messaggi continui di "buongiorno" e "buonanotte":** Rituali che scandiscono la giornata e rafforzano il legame.
        * `"Buongiorno raggio di sole! Spero tu abbia una giornata stupenda. Ti penso ❤️"`

* **Tossici:**
    * **Isolamento dalla cerchia sociale:** Un partner si lamenta se l'altro esce con gli amici.
        * `"Ancora con i tuoi amici stasera? Pensavo volessi stare con me... Evidentemente loro sono più importanti."`
    * **Richiesta di password:** Pretendere l'accesso a social e telefono come "prova d'amore".
        * `"Amore, se non hai nulla da nascondere, perché non mi dai la password del tuo telefono? Tra di noi non ci devono essere segreti."`
    * **Gelosia possessiva:** Scenate di gelosia per un like su una foto o un commento innocente.
        * `"Chi è questo che ti ha messo 'cuore' alla foto? Lo conosci? Perché non me ne hai mai parlato?"`

* **Neutrali:**
    * **Sincronizzazione quotidiana:** Coordinarsi su aspetti pratici della vita di coppia che inizia.
        * `"Stasera ordiniamo sushi? Passo a prenderlo io dopo la palestra."`
    * **Condivisione di momenti della giornata:** Inviarsi foto o video di ciò che si sta facendo.
        * `"Guarda che tramonto dal mio ufficio! Peccato non essere lì a vederlo con te."`
    * **Creazione di un "linguaggio di coppia":** L'inizio dell'uso di soprannomi e battute interne.
        * `"Allora, 'patato', ci vediamo dopo? Non vedo l'ora! 😉"`

## Disillusione e Negoziazione

### 1. Descrizione e Caratteristiche

La "sbornia" chimica della luna di miele finisce e **la realtà fa il suo ingresso**. Le imperfezioni del partner, prima ignorate o trovate affascinanti, ora diventano irritanti. Emergono le differenze individuali (abitudini, valori, bisogni) e questo porta ai primi veri conflitti. È una fase critica in cui ogni partner cerca di affermare la propria individualità e i propri bisogni, a volte a scapito di quelli del partner. L'obiettivo (sano) è imparare a negoziare e a gestire il conflitto.

* **Caratteristiche Principali:**
    * Fine dell'idealizzazione.
    * Emersione di differenze e difetti.
    * Primi litigi significativi.
    * Bisogno di riaffermare la propria individualità e i propri spazi.
    * Messa in discussione della compatibilità a lungo termine.

### 2. Esempi di Eventi (Adattabili a una Chat)

* **Sani:**
    * **Gestione del primo litigio serio:** Dopo una discussione accesa in chat, uno dei due propone di parlarne a voce e con calma.
        * `"Ok, sento che stiamo solo peggiorando le cose scrivendoci. Ho esagerato. Possiamo vederci stasera e parlarne guardandoci negli occhi?"`
    * **Definizione di confini:** Esprimere il bisogno di spazi personali in modo non accusatorio.
        * `"Stasera ho davvero bisogno di una serata per me, per leggere un po' e staccare la spina. Non è niente contro di te, ne ho solo bisogno. Ci sentiamo domani mattina?"`
    * **Scuse sincere post-litigio:** Ammettere di aver sbagliato durante una discussione.
        * `"Ripensandoci, ho reagito in modo esagerato prima. Mi dispiace, non volevo ferirti."`

* **Tossici:**
    * **Passivo-aggressività via chat:** Usare messaggi ambigui, sarcasmo o il silenzio per punire il partner.
        * `Partner A: "Tutto ok?" Partner B: "Sì." (dopo ore di silenzio e risposte monosillabiche)`
    * **"Rinfacciare" eventi passati:** Usare errori precedenti del partner come arma in un nuovo litigio.
        * `"Ah, tipico tuo! Fai come quella volta che ti sei dimenticato del mio compleanno!"`
    * **Escalation del litigio via testo:** Invece di de-escalare, la chat diventa un campo di battaglia con insulti e accuse.
        * `"Sei un egoista! Pensi solo a te stesso, come sempre! Non ti importa niente di me!"`

* **Neutrali:**
    * **Negoziazione delle faccende domestiche (se conviventi):** La prima discussione su chi fa cosa.
        * `"Senti, ho notato che la spazzatura è di nuovo piena. Potresti pensarci tu stasera? Io oggi ho cucinato e pulito i piatti."`
    * **Disaccordo su una decisione comune:** Opinioni diverse su dove andare in vacanza o come spendere dei soldi.
        * `"Io preferirei il mare, tu la montagna... dobbiamo trovare un compromesso. Che ne dici se ne parliamo meglio a cena?"`
    * **Affrontare un'abitudine fastidiosa:** Far notare per la prima volta un comportamento irritante del partner.
        * `"Amore, ti vorrei chiedere una cosa... quando lasci l'asciugamano bagnato sul letto mi dà un po' fastidio. Riusciresti a stenderlo? ❤️"`

## Stabilità e Impegno Profondo

### 1. Descrizione e Caratteristiche

Se la coppia supera la fase della lotta di potere, entra in una fase di **stabilità e accettazione**. I partner hanno imparato a conoscersi veramente, con pregi e difetti. L'amore non è più solo passione travolgente, ma anche **affetto profondo, stima e fiducia**. La relazione diventa un porto sicuro. Si smette di cercare di cambiare il partner e lo si accetta per quello che è. Le decisioni importanti vengono prese insieme, con un focus sul benessere comune.

* **Caratteristiche Principali:**
    * Accettazione reciproca dei difetti.
    * Fiducia, sicurezza e rispetto consolidati.
    * Gestione costruttiva dei conflitti.
    * La relazione è una base sicura.
    * Pianificazione a lungo termine.

### 2. Esempi di Eventi (Adattabili a una Chat)

* **Sani:**
    * **Supporto durante una crisi personale:** Uno dei due perde il lavoro o ha un lutto in famiglia e l'altro offre supporto incondizionato via chat.
        * `"Amore, ho appena saputo del colloquio. Non ti preoccupare, troverai di meglio. Sei la persona più in gamba che conosca. Stasera sono tutto per te, conta su di me per qualsiasi cosa."`
    * **Prendere una decisione di vita insieme:** Discutere via chat i pro e i contro di un trasferimento per lavoro o dell'acquisto di una casa.
        * `"Ho ricevuto l'offerta ufficiale da quell'azienda a Milano. So che ne abbiamo parlato, ma volevo riparlarne con te prima di dare una risposta. Cosa ne pensi? Come ti sentiresti?"`
    * **Gesti di cura quotidiana:** Messaggi che dimostrano attenzione e conoscenza profonda dell'altro.
        * `"So che oggi hai la riunione importante. Spacca tutto! Ti ho lasciato il caffè pronto. In bocca al lupo!"`

* **Tossici (Sottili):**
    * **Compiacenza e trascuratezza:** La stabilità diventa noia. La chat è solo logistica, senza più affetto o intimità.
        * `"Hai pagato la bolletta?" "Sì." "Ok." (Nessun altro messaggio per tutto il giorno)`
    * **Evitamento del conflitto (Stonewalling):** Per paura di turbare la stabilità, si evitano argomenti scomodi che emergono in chat.
        * `Partner A: "Vorrei parlarti di come mi hai risposto ieri sera..." Partner B: "Non ricominciamo, è tutto a posto. Lasciamo perdere."`
    * **Decisioni unilaterali su questioni importanti:** Uno dei due comunica una scelta già presa che impatta entrambi.
        * `"Ho accettato quel lavoro all'estero. Parto tra un mese. Te lo dico così ci organizziamo."`

* **Neutrali:**
    * **Gestione della routine complessa:** Coordinare via chat orari di lavoro, figli, appuntamenti, spesa.
        * `"Ok, allora io prendo Anna a scuola e la porto a nuoto. Tu riesci a passare a prendere la spesa e preparare qualcosa per cena?"`
    * **Condivisione di notizie familiari/amici:** Aggiornarsi a vicenda sulla vita delle persone care.
        * `"Mi ha chiamato mia madre, ha detto che zia sta meglio. Volevo dirtelo."`
    * **Promemoria e supporto logistico:**
        * `"Ricordati la visita dal dentista alle 15! In bocca al lupo!"`

## Co-creazione e Interdipendenza Matura

### 1. Descrizione e Caratteristiche

Questa fase rappresenta l'apice della maturità relazionale. La coppia non è solo un "noi" che si protegge dal mondo esterno, ma un **team che agisce sul mondo per creare qualcosa insieme**. Questo "qualcosa" può essere una famiglia, un progetto lavorativo, un impegno sociale, o semplicemente un progetto di vita condiviso con intenti e valori comuni. L'interdipendenza è matura: i partner contano l'uno sull'altro, ma mantengono anche la loro forte identità individuale.

* **Caratteristiche Principali:**
    * Focus su obiettivi e progetti comuni.
    * Senso di squadra e scopo condiviso.
    * Interdipendenza sana (non co-dipendenza).
    * Profondo senso di sicurezza e partnership.
    * La coppia diventa un'unità che impatta il mondo esterno.

### 2. Esempi di Eventi (Adattabili a una Chat)

* **Sani:**
    * **Pianificazione del futuro dei figli:** Discutere via chat della scuola da scegliere o di come affrontare un problema educativo.
        * `"Ho parlato con la maestra di Marco. Mi ha detto [problema]. Che ne dici se stasera ci sediamo e pensiamo insieme a una strategia per aiutarlo?"`
    * **Lancio di un progetto comune:** Brainstorming via chat per un'attività da avviare insieme (un blog, un piccolo business, una ristrutturazione).
        * `"Ho avuto un'idea per il nostro B&B! E se creassimo dei pacchetti a tema per i weekend? Tipo 'weekend enogastronomico'. Ti mando qualche link."`
    * **Sostegno reciproco nei ruoli:** Incoraggiarsi a vicenda nei rispettivi progetti individuali che contribuiscono al benessere comune.
        * `"Vai e conquista quella presentazione! Sai quanto vale questo progetto per entrambi. Sono il tuo più grande fan!"`

* **Tossici:**
    * **Perdita totale dell'identità:** Uno dei due annulla completamente i propri bisogni per il "bene del progetto" (es. la famiglia), generando risentimento.
        * `"Ovviamente devo rinunciare io alla cena di lavoro, perché i bambini vengono prima di tutto. Come sempre."` (detto con tono di martirio)
    * **Usare il progetto comune come arma:** Sfruttare i figli o il business condiviso per manipolare l'altro durante un litigio.
        * `"Se non fai come dico io, sappi che questo avrà delle conseguenze anche sui bambini."`
    * **Competizione invece di collaborazione:** Invece di essere un team, i partner competono su chi contribuisce di più al progetto comune.
        * `"Sono io che passo le notti a lavorare su questa cosa mentre tu dormi. Non dire che è un successo 'nostro'."`

* **Neutrali:**
    * **Divisione dei compiti per un evento:** Organizzare una festa o una cena importante via chat.
        * `"Ok, lista della spesa fatta. Io penso agli antipasti e al primo. Tu ti occupi del secondo e del dolce?"`
    * **Coordinamento finanziario a lungo termine:** Discutere di investimenti, mutuo, risparmi.
        * `"Ho parlato con il consulente. Mi ha mandato il prospetto per il fondo pensione. Te lo inoltro, così stasera lo guardiamo insieme."`
    * **Discussione su questioni etiche/valoriali:** Confrontarsi su come educare i figli su un tema specifico o su una scelta politica/sociale importante.
        * `"Marco oggi mi ha chiesto [domanda difficile]. Non sapevo bene cosa rispondere. Come pensi che dovremmo affrontare l'argomento con lui?"`

## Crisi e Rielaborazione

### 1. Descrizione e Caratteristiche

Questa non è una semplice lite; è una **crisi esistenziale per la coppia**. Può essere innescata da un evento traumatico (es. tradimento, grave malattia, lutto, crisi finanziaria) o dall'accumulo di anni di risentimento e distanza emotiva. La domanda fondamentale non è più "come risolviamo questo problema?", ma **"vogliamo ancora essere un 'noi'?"**. La relazione stessa viene messa in discussione. La comunicazione è carica di dolore, rabbia o, nel peggiore dei casi, di una glaciale apatia. È il momento in cui spesso si considera l'aiuto di terzi (terapeuti, amici fidati, mediatori).

* **Caratteristiche Principali:**
    * Profonda distanza emotiva o, al contrario, conflitto continuo e distruttivo.
    * Messa in discussione dei pilastri della relazione (fiducia, amore, rispetto, futuro condiviso).
    * Sensazione di essere a un punto di non ritorno.
    * Presenza di emozioni estreme: rabbia, disperazione, apatia, profonda tristezza.
    * La sopravvivenza della coppia è l'argomento centrale.

### 2. Esempi di Eventi (Adattabili a una Chat)

* **Sani (o il modo più sano di gestire una crisi):**
    * **Proposta di terapia di coppia:** Uno dei due riconosce l'incapacità di risolvere la crisi da soli e fa un passo costruttivo.
        * `"Sento che continuiamo a girare a vuoto e a farci solo del male. Ho pensato molto e vorrei proporti di iniziare una terapia di coppia. Per me vale la pena fare un ultimo tentativo serio."`
    * **Richiesta di una conversazione onesta e definitiva:** L'intenzione è fare chiarezza, anche se dolorosa.
        * `"Non possiamo più andare avanti così. Non sono felice e penso neanche tu. Dobbiamo sederci e parlare onestamente del nostro futuro, qualunque esso sia."`
    * **Accordo per una pausa di riflessione:** Prendere spazio in modo concordato per fare chiarezza individuale.
        * `"Forse qualche giorno di distanza potrebbe aiutarci a capire cosa vogliamo davvero. Potrei andare a dormire da [amico/parente] per il weekend. Non è una rottura, è una pausa per pensare."`

* **Tossici:**
    * **Scoperta di un tradimento via messaggio:** Ricevere per sbaglio un messaggio destinato all'amante o trovare chat compromettenti.
        * `"Ho visto il messaggio che hai mandato a [Nome]. Non provare a negare. È finita."`
    * **Ultimatum e minacce via chat:** Tentativi di controllare il comportamento dell'altro attraverso la paura.
        * `"O blocchi quella persona su ogni social ORA, o faccio le valigie. A te la scelta."`
    * **Il messaggio "Dobbiamo parlare" lasciato in sospeso:** Creare ansia deliberatamente come forma di punizione o controllo.
        * `(Messaggio inviato alle 10:00: "Stasera dobbiamo parlare seriamente". Seguito da ore di silenzio, lasciando l'altro nell'angoscia).`

* **Neutrali (che indicano una profonda crisi):**
    * **Comunicazione ridotta al minimo indispensabile:** La chat, un tempo piena di vita, è ora un deserto emotivo usato solo per logistica impersonale.
        * `Partner A: "Passo a prendere i bambini alle 5." Partner B: "Ok."`
    * **Tentativi di comunicazione che cadono nel vuoto:** Uno dei due prova a creare un contatto, ma viene respinto passivamente.
        * `Partner A: "Oggi al lavoro è successa una cosa incredibile..." Partner B: (visualizza e non risponde per ore, poi scrive) "Ok, dopo sento."`
    * **Chat che mostra una conversazione interrotta:** Un litigio iniziato via messaggio che si spegne non per risoluzione, ma per sfinimento e apatia, senza più la forza di continuare a discutere.

## Conclusione o Trasformazione

### 1. Descrizione e Caratteristiche

Questa è la diretta conseguenza della fase di crisi. A seconda delle decisioni prese, la relazione può prendere due strade: la **Conclusione** (la rottura) o la **Trasformazione** (la ricostruzione).

* **Conclusione:** È il processo di separazione emotiva e pratica. Comporta il lutto per la fine della relazione, che può manifestarsi con tristezza, rabbia, ma anche sollievo. La comunicazione si concentra sulla gestione degli aspetti pratici della separazione (casa, figli, finanze).
* **Trasformazione:** La coppia decide consapevolmente di rimanere insieme, ma su basi nuove. È un "secondo inizio" che richiede uno sforzo enorme da entrambe le parti per non ricadere nei vecchi schemi distruttivi. La comunicazione diventa più intenzionale, cauta e focalizzata sulla guarigione e sulla creazione di nuovi patti relazionali.

* **Caratteristiche Principali:**
    * **(Conclusione):** Lutto, negoziazione logistica, definizione di nuovi confini personali, chiusura emotiva.
    * **(Trasformazione):** Sforzo consapevole, comunicazione intenzionale, vulnerabilità, creazione di nuovi rituali di coppia, ottimismo cauto.

### 2. Esempi di Eventi (Adattabili a una Chat)

* **Sani:**
    * **(Conclusione) Gestione rispettosa della separazione:** Coordinarsi logisticamente senza recriminazioni.
        * `"Ciao. Ti scrivo per sapere quando preferiresti passare a prendere le tue cose, così mi faccio trovare a casa. Spero tu stia bene."`
    * **(Conclusione) Messaggio di chiusura maturo:** Un ultimo scambio per riconoscere il passato e augurarsi il meglio.
        * `"Volevo solo dirti che, nonostante come sia finita, sono grato/a per gli anni belli che abbiamo passato. Ti auguro sinceramente di essere felice."`
    * **(Trasformazione) Check-in post-terapia/litigio costruttivo:** Verificare l'impatto di una conversazione e rafforzare il nuovo approccio.
        * `"La chiacchierata di ieri sera è stata importante per me. Grazie per avermi ascoltato senza interrompere. Sento che stiamo facendo dei passi avanti."`
    * **(Trasformazione) Riconoscere e lodare il cambiamento:** Notare uno sforzo del partner e dargli un rinforzo positivo.
        * `"Ho notato che ultimamente stai cercando di aiutarmi di più in casa, e volevo dirti che lo apprezzo tantissimo. Grazie."`

* **Tossici:**
    * **(Conclusione) Continuare a litigare post-rottura:** Usare la chat per molestare, accusare o controllare l'ex partner.
        * `"Ho visto dalle storie che eri a quella festa. Complimenti, non perdi tempo eh? Chissà con chi eri."`
    * **(Conclusione) Triangolazione con i figli:** Usare i figli come messaggeri o come arma emotiva.
        * `"Dì a tuo padre che il bonifico non è ancora arrivato."` (scritto al figlio/a)
    * **(Trasformazione) "Falsa tregua":** Fingere di essere cambiati per poi rinfacciare la seconda possibilità data.
        * `"Te l'avevo detto che non saresti cambiato/a! È inutile! Spreco solo il mio tempo con te!"`
    * **(Trasformazione) Usare la crisi passata come ricatto:** Sfruttare la "colpa" di uno dei due per avere potere nella relazione rinnovata.
        * `"Dopo quello che mi hai fatto, sarebbe il minimo che tu facessi come dico io per una volta."`

* **Neutrali:**
    * **(Conclusione) La chat puramente logistica:** Scambi necessari per la gestione di aspetti burocratici o pratici post-separazione.
        * `"Ciao, mi è arrivata una raccomandata per te. Te la lascio nella cassetta della posta o preferisci passare a prenderla?"`
    * **(Conclusione) Comunicazioni relative ai figli:** Scambi informativi e non emotivi sull'organizzazione e il benessere dei figli.
        * `"Ti ricordo che domani porto io Sara dal pediatra alle 16. Ti aggiorno dopo la visita."`
    * **(Trasformazione) Pianificazione cauta:** Organizzare attività insieme con una nuova e deliberata attenzione alla comunicazione.
        * `"Ok, allora come d'accordo, proviamo a passare il weekend fuori. Parliamone bene stasera per decidere un posto che piaccia a entrambi senza dare nulla per scontato."`

# Coppie di Personaggi

## Persona 1: Giulia

### Modulo 1: Anagrafica e Contesto Socio-Culturale

*   **Nome:** Giulia Cattaneo
*   **Età:** 29
*   **Genere e Pronomina:** Donna, lei/sua
*   **Professione e Settore:** Curatrice museale associata presso un'importante fondazione d'arte a Torino. Ama profondamente il suo lavoro, ma lo percepisce come "meno importante" e meno concreto rispetto a quello del suo partner, una fonte di lieve insicurezza.
*   **Status Socio-Economico:** Medio. Guadagna bene, ma non ai livelli del settore tech del suo compagno.
*   **Contesto Geografico e Culturale:** Nata e cresciuta a Firenze in una famiglia borghese colta, si è trasferita a Torino per amore e per una buona opportunità lavorativa.
*   **Struttura Familiare di Origine:** Figlia unica di genitori molto esigenti (padre avvocato, madre docente liceale) dove l'affetto era spesso condizionato ai risultati scolastici. È sempre stata la "figlia perfetta".

### Modulo 2: Nucleo Psicologico (Il "Chi Sono")

*   **Modello Big Five (OCEAN):**
    *   **Apertura all'Esperienza:** Alto. La sua professione e i suoi interessi ruotano attorno all'arte, alla storia e alle nuove idee. È curiosa e intellettualmente vivace.
    *   **Coscienziosità:** Medio-Alto. È molto diligente e affidabile nel suo lavoro, ma nella vita privata può diventare disorganizzata quando è emotivamente turbata.
    *   **Estroversione:** Medio. Ama le conversazioni profonde e intime, ma è a disagio in contesti sociali ampi e superficiali. Trae energia da connessioni significative.
    *   **Gradevolezza:** Alto. Estremamente empatica, accomodante e desiderosa di mantenere l'armonia. Tende a sopprimere i propri bisogni per non disturbare gli altri.
    *   **Nevroticismo:** Alto. Tende a rimuginare, è ansiosa riguardo allo stato della sua relazione e molto sensibile alle critiche o ai segnali di disapprovazione.
*   **Locus of Control:** Esterno. Il suo stato d'animo e la sua autostima dipendono pesantemente dall'approvazione e dall'attenzione del suo partner. Crede che la felicità della relazione dipenda più dalle azioni di lui che dalle sue.
*   **Assiologia (Valori Fondamentali):** Amore, Connessione, Stabilità emotiva, Bellezza, Lealtà.

### Modulo 3: Mondo Emotivo (Il "Come Sento")

*   **Intelligenza Emotiva (EQ):**
    *   **Autoconsapevolezza:** Media. Riconosce la propria ansia e tristezza, ma fatica a identificarne le cause profonde, spesso attribuendole a proprie mancanze piuttosto che a dinamiche relazionali.
    *   **Autoregolazione:** Bassa. È spesso in balia delle sue emozioni. L'ansia può sopraffarla, portandola a comportamenti di ricerca di rassicurazione.
    *   **Motivazione Interna:** Alta per il suo lavoro, ma bassa per i suoi bisogni personali. La sua motivazione principale è assicurarsi l'amore del partner.
    *   **Empatia:** Altissima. Sente con grande intensità le emozioni altrui, specialmente quelle del suo compagno, cercando costantemente di anticipare e soddisfare i suoi bisogni.
    *   **Abilità Sociali:** Buone in contesti strutturati, ma insicure nelle dinamiche di coppia.
*   **Stato Emotivo di Base (Baseline):** Speranzosa ansia. Un costante stato di allerta, in attesa di segnali di affetto o, più spesso, di distacco.
*   **Trigger Emotivi:**
    *   **Positivi:** Momenti di intimità e connessione con il partner, complimenti sinceri, sentirsi "vista" e capita.
    *   **Negativi:** Silenzi prolungati del partner, risposte monosillabiche, vederlo assorto nel lavoro mentre lei cerca un contatto, qualsiasi forma di critica.

### Modulo 4: Dinamiche Interpersonali (Il "Come Mi Relaziono")

*   **Stile di Attaccamento (Adulto):** **Ansioso-Preoccupato**. Ha un'intensa fame di intimità e teme costantemente che il partner possa perdere interesse e lasciarla. Manifesta questo timore con "protest behavior": ricerca eccessiva di contatto, bisogno di rassicurazioni, analisi ossessiva di ogni interazione.
*   **Stile di Comunicazione Prevalente:** **Passivo**. Raramente esprime un'esigenza o una lamentela in modo diretto per paura del conflitto. Quando la frustrazione diventa insostenibile, può sfociare nel **Passivo-Aggressivo** (bronci, silenzi carichi di significato).
*   **Bisogni Sociali Primari:** Intimità, Validazione, Rassicurazione.

### Modulo 5: Motivazioni e Paure (Il "Perché Agisco")

*   **Motivazioni Primarie (Drives):** **Affiliazione (Affiliation)**. Il suo bisogno fondamentale è stabilire e mantenere una relazione stretta, sicura e amorevole, che costituisce il centro della sua identità.
*   **Paure Fondamentali:** La **paura dell'abbandono**. Il terrore di non essere abbastanza e di essere lasciata sola è il motore di gran parte dei suoi comportamenti.
*   **Obiettivi a Lungo Termine:** Costruire una famiglia e una vita stabile con il suo partner. Il suo sogno è un futuro condiviso che solidifichi il loro legame.

### Modulo 6: Meccanismi di Difesa e Coping (Il "Come Reagisco")

*   **Risposta allo Stress (Fight, Flight, Freeze, Fawn):** **Freeze (Immobilità)**. Di fronte a un'aperta critica o a un'evidente distanza del partner, si blocca. Si sente sopraffatta, incapace di pensare lucidamente o di articolare una difesa.
*   **Meccanismi di Difesa Primari:** **Introiezione** (interiorizza le critiche del partner, convincendosi di essere lei il problema: "Sono troppo bisognosa", "Ho sbagliato io"), **Negazione** (minimizza la gravità dei comportamenti distanzianti del partner, aggrappandosi ai rari momenti positivi).

### Modulo 7: Storia Personale Rilevante (Il "Come Sono Diventato Così")

*   **Evento Formativo 1 (Successo):** La vincita di una borsa di studio prestigiosa, che le ha dato un raro momento di approvazione incondizionata da parte dei genitori, cementando l'idea che il valore personale derivi da un'eccezionale performance.
*   **Evento Formativo 2 (Relazionale):** Il rapporto con il padre, un uomo brillante ma emotivamente inaccessibile. Giulia ha passato l'infanzia cercando di guadagnarsi la sua attenzione e il suo affetto, uno schema che ora replica con il suo partner.
*   **Evento Formativo 3 (Svolta):** La decisione di lasciare Firenze per seguire Matteo a Torino. L'ha vissuta come il più grande gesto d'amore, ma ha anche aumentato la sua dipendenza da lui, avendo lasciato la sua rete di supporto primaria.

### Modulo 8: Sintesi Dinamica e Conflitto Interno

Giulia è una donna intelligente e appassionata, la cui grande capacità di amare diventa una fonte di angoscia. Il suo **conflitto interno fondamentale** è tra il suo **sincero desiderio di una connessione profonda e autentica (Modulo 5) e la sua incapacità di credere di meritarla senza doversela costantemente guadagnare (Modulo 4)**. Questa insicurezza la porta a scegliere partner emotivamente non disponibili, nei quali cerca disperatamente quell'approvazione che non ha mai ricevuto dal padre e che non sa darsi da sola. Confonde l'ansia e il sollievo del ciclo di allontanamento e riavvicinamento con la passione, rimanendo intrappolata in una dinamica che alimenta la sua paura più grande: quella di non essere degna d'amore.

---

## Persona 2: Matteo

### Modulo 1: Anagrafica e Contesto Socio-Culturale

*   **Nome:** Matteo Ferri
*   **Età:** 32
*   **Genere e Pronomina:** Uomo, lui/suo
*   **Professione e Settore:** Co-fondatore e CEO di una startup tech in rapida crescita. Lavora 12 ore al giorno, vive per la sua azienda. La sua soddisfazione è legata alla crescita, ai finanziamenti ottenuti e al dominio del mercato.
*   **Status Socio-Economico:** Reale: Medio-alto ma volatile (legato al valore della startup). Percepito: Si considera un futuro "vincente" e agisce come tale.
*   **Contesto Geografico e Culturale:** Cresciuto a Bologna in un ambiente accademico. Si è trasferito a Torino perché la considera un hub di innovazione più promettente.
*   **Struttura Familiare di Origine:** Padre professore universitario di filosofia, madre ricercatrice in biologia. Genitori intellettuali che hanno sempre considerato le sue ambizioni imprenditoriali come superficiali e materialistiche, creando in lui un forte desiderio di rivalsa.

### Modulo 2: Nucleo Psicologico (Il "Chi Sono")

*   **Modello Big Five (OCEAN):**
    *   **Apertura all'Esperienza:** Medio-Alto. È aperto a nuove idee e tecnologie, ma solo se hanno un'applicazione pratica e un potenziale di profitto. Non ha tempo per l'astrazione fine a se stessa.
    *   **Coscienziosità:** Alto. Estremamente focalizzato, disciplinato e ambizioso. La sua vita è organizzata in funzione degli obiettivi della sua azienda.
    *   **Estroversione:** Alto. È carismatico, assertivo e bravo a presentare la sua visione a investitori e dipendenti. L'interazione sociale è uno strumento per raggiungere i suoi scopi.
    *   **Gradevolezza:** Basso. Può essere impaziente, esigente e talvolta sprezzante con chi non condivide la sua visione o non tiene il suo passo. La cooperazione è secondaria al risultato.
    *   **Nevroticismo:** Basso. Appare estremamente resiliente allo stress, calmo e quasi inscalfibile. In realtà, canalizza tutta la sua ansia nella performance lavorativa.
*   **Locus of Control:** Interno. È convinto al 100% di essere l'unico artefice del proprio successo o fallimento. Non crede nella fortuna, ma nella volontà e nella strategia.
*   **Assiologia (Valori Fondamentali):** Potere, Successo, Innovazione, Efficienza, Autonomia.

### Modulo 3: Mondo Emotivo (Il "Come Sento")

*   **Intelligenza Emotiva (EQ):**
    *   **Autoconsapevolezza:** Bassa. È quasi completamente disconnesso dal suo mondo emotivo. Vede le emozioni come dati inutili o, peggio, come ostacoli all'efficienza.
    *   **Autoregolazione:** Apparentemente Alta. Sopprime ogni emozione che non sia la grinta o l'entusiasmo per il lavoro. Non gestisce la rabbia o la frustrazione, la reindirizza.
    *   **Motivazione Interna:** Altissima. È un motore inesauribile, spinto dal desiderio di costruire qualcosa di grande e dimostrare il suo valore.
    *   **Empatia:** Molto Bassa. Fatica a capire la prospettiva emotiva altrui. Interpreta i bisogni emotivi di Giulia come debolezza o distrazioni irragionevoli.
    *   **Abilità Sociali:** Elevate a livello strumentale. È un ottimo networker e leader, ma non è in grado di sostenere relazioni basate sull'intimità emotiva.
*   **Stato Emotivo di Base (Baseline):** Intensa focalizzazione. Una sorta di "visione a tunnel" costantemente puntata sul prossimo obiettivo.
*   **Trigger Emotivi:**
    *   **Positivi:** Vincere un round di finanziamenti, superare un competitor, vedere i grafici di crescita salire.
    *   **Negativi:** Mettere in discussione la sua autorità o la sua visione, fallimenti del prodotto, percepire i bisogni emotivi di Giulia come un'intrusione nel suo tempo.

### Modulo 4: Dinamiche Interpersonali (Il "Come Mi Relaziono")

*   **Stile di Attaccamento (Adulto):** **Evitante-Distanziante (Dismissive)**. Considera l'indipendenza e l'autosufficienza come i valori supremi. L'intimità lo fa sentire soffocato e minacciato nella sua autonomia. Apprezza la presenza di Giulia come supporto e "normalità" nella sua vita, ma si ritira non appena lei avanza richieste emotive.
*   **Stile di Comunicazione Prevalente:** **Assertivo** tendente all'**Aggressivo**. Comunica in modo diretto, orientato ai fatti e alle soluzioni. Quando si sente pressato emotivamente, può diventare svalutante o tagliare corto.
*   **Bisogni Sociali Primari:** Controllo e Autonomia. Cerca relazioni che non interferiscano con i suoi obiettivi primari.

### Modulo 5: Motivazioni e Paure (Il "Perché Agisco")

*   **Motivazioni Primarie (Drives):** **Potere (Power)**. Il suo bisogno dominante è influenzare gli altri, controllare il suo ambiente e lasciare un'impronta indelebile attraverso la sua creazione (la startup).
*   **Paure Fondamentali:** La **paura dell'irrilevanza e del fallimento**. Teme più di ogni altra cosa di non riuscire a realizzare la sua visione e di finire per essere un "nessuno", esattamente come crede che la sua famiglia lo percepisca.
*   **Obiettivi a Lungo Termine:** Portare la sua azienda a una "exit" miliardaria (vendita o quotazione in borsa). Raggiungere uno status che lo renda intoccabile e universalmente riconosciuto come un vincente.

### Modulo 6: Meccanismi di Difesa e Coping (Il "Come Reagisco")

*   **Risposta allo Stress (Fight, Flight, Freeze, Fawn):** **Fight**. Affronta ogni ostacolo, sia esso un bug nel software o una richiesta emotiva di Giulia, come un problema da dominare e risolvere rapidamente per poter tornare alle cose importanti.
*   **Meccanismi di Difesa Primari:** **Razionalizzazione** ("Non ho tempo per le vacanze, l'azienda è in una fase critica"), **Spostamento** (sfoga la frustrazione e lo stress del lavoro diventando irritabile e critico con Giulia su questioni banali).

### Modulo 7: Storia Personale Rilevante (Il "Come Sono Diventato Così")

*   **Evento Formativo 1 (Fallimento):** La sua prima startup, fondata a 22 anni, fallì miseramente. L'umiliazione e il "te l'avevo detto" implicito della sua famiglia lo hanno reso ancora più determinato a riuscire a qualsiasi costo.
*   **Evento Formativo 2 (Relazionale):** Innumerevoli cene di famiglia in cui suo padre disquisiva di Kant mentre lui cercava, senza successo, di spiegare il valore del suo progetto. Questo ha creato un'equivalenza nella sua mente: mondo emotivo/intellettuale = debolezza, mondo degli affari = forza.
*   **Evento Formativo 3 (Svolta):** L'incontro con il suo primo investitore "angel", che ha creduto in lui e gli ha dato i fondi per partire. Questo ha validato la sua visione del mondo e rafforzato la sua convinzione di essere sulla strada giusta.

### Modulo 8: Sintesi Dinamica e Conflitto Interno

Matteo è l'archetipo dell'imprenditore moderno, un costruttore di imperi che sacrifica tutto sull'altare del successo. Il suo **conflitto interno fondamentale** risiede nello scontro tra il suo **bisogno di potere e di realizzazione per dimostrare il proprio valore (Modulo 5) e la sua incapacità di accedere a quella connessione umana che potrebbe dargli un senso di appagamento più profondo**. La sua fame di successo è una fuga dalla sua paura dell'irrilevanza. Ha bisogno di Giulia come ancora a una vita "normale", ma il suo stile evitante (Modulo 4) la respinge sistematicamente, creando un deserto emotivo intorno a sé. Sta costruendo un castello magnifico, ma rischia di ritrovarsi a regnare da solo, scoprendo che il potere non può curare la solitudine.