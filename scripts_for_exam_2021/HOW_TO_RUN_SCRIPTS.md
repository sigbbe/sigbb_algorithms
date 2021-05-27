# HowTo-Guide for å kjøre kok-script

## Apriori
Her, og i alle andre scripts er argumentet bak $-f$ flagget den relative veien til .csv filene som inneholder datapunktene. Tallet bak $-s$ den prosentvise minimum support counten, denne står som oftest i oppgaveteksten. Tallet bak $-c$ er minimum confidence for generering av assosiasjonsregler. Dette er også prosentvis.  
```
py apriori.py -f ./data/<file_name.csv> -s 0.5 -c 0.5
```

## DBSCAN
Her er tallet bak $-p$ minimum points, altså det minste antallet punkter et datapunkt må ha i sitt nabolag for å telles som et kjernepunkt. Tallet bak $-e$ er epsilon, altså hvor stort nabolaget til hvert datapunkt skal være. Argumentet for $-d$ er metoden som skal brukes for å kalkulere distansene mellom datapunkter. Gyldige verdier for $-d$ argumentet er følgende: *manhattan* (*city-block*), *euclidean*, *cosine*, *minkowski*.   
```
py dbscan.py -f ./data/<file_name.csv> -p 4 -e 3 -d manhattan
--dimensions

```

## Decision tree induction
Gyldige arguemnter for $-c$ flagget er: *entropy* og *gini*. Dette er urenhetsmålet (*impurity measure*) som brukes for å generere beslutningstreet. 
```
py decision_tree_induction.py -f ./data/<file_name.csv> -c entropy
```

Formatet på dataene for scriptene kan finnes i ./data/ mappen. 

NB: De andre scriptene som ligger i denne mappen funker ikke!