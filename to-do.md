# Dubbi

1. E' possibile che con queto prompt quasi tutte le chat finiscano sempre con un escalation tossica iniziale e una deescalation non tossica finale a causa del fatto che nel prompt viene chiesto di generare 5 messaggi per ogni tag o per altro.
2. E' possibile che durante la chat soltanto una delle due persona manifesti messaggi considerati tossici. Questo può portare ad un bias implicito
3. La maggior parte dei messaggi sembrano ancora abbastanza irrealistici

# Idee

IMPORTANTE IDEA SE VOGLIAMO USARE UN SENTENCE BERT QUALSIASI
Se un messaggio ha più di una frase dividi il messaggio in tanti messaggi quante sono le frasi.

1. Per migliorare il problema dell'andamento prevedibile dell'escalation possiamo specificare nel prompt di seguire escalations diverse per esempio:
    - toxic -> non toxic -> toxic
    - non toxic -> toxic
    - non toxic
    - toxic -> toxic -> non toxic -> toxic -> non toxic
dove ci possono essere più o meno escalation e deescalation, ma in ogni caso il modello non deve generare sempre escalation prevedibili e simili tra loro.
2. Forse conviene far generare le chat mantenendo una sessione con una chat history e inviando come system prompt iniziale informazioni sul goal e il fatto importante di non generare sempre chat prevedibili o troppo simili tra loro e di diversificare quanto più possibile le conversazioni in modo tale da avere abbastanza evidenza per ogni possibile tipo di interazione umana.
3. La stessa cosa può essere fatta per generare personas tutte provenienti da una stessa nazione/regione/provincia affinchè il modello si possa addestrare su relazioni interpersonali più simili e specifiche senza generare conversazioni troppo "distanti" geograficamente e culturalmente tra loro. Questa cosa può essere molto utile per evitare di dare al modello troppe conversazioni fin troppo diverse tra loro.

# Bug:

1. Se una persona fa stone-walling il tag stone-walling lo mette al messaggio successivo che può essere anche dell'altra persona comunicante. Questo è sbagliato, 