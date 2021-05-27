

# Apriori

How to run the script:

```
py apriori.py -f <Path to file> -s <Minimum support count> -c <Minimum confidence>  
py apriori.py -f ./data/v_2019.csv -s 0.33 -c 0.6  
```


Structure of the data:

Each line represents a single transaction with $n > 0$ number of items

```
H,B,K  
H,B  
H,C,I  
C,I  
I,K  
H,C,I,U  
```