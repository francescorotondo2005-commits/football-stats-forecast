# Risultati sul test (2024-25, 2025-26, partite mai viste in training)

## Errore medio assoluto (MAE) del valore atteso

| Statistica | Media campionato | Media mobile | GLM Poisson | Gradient Boosting | Scelto (in validazione) | Miglioramento vs media mobile |
|---|---|---|---|---|---|---|
| Gol | 0.985 | 0.932 | 0.904 | 0.905 | GLM Poisson | 3.0% |
| Tiri | 4.115 | 3.768 | 3.555 | 3.573 | Gradient Boosting | 5.2% |
| Tiri in porta | 1.939 | 1.795 | 1.729 | 1.737 | GLM Poisson | 3.7% |
| Falli | 3.194 | 2.851 | 2.827 | 2.821 | Gradient Boosting | 1.1% |
| Corner | 2.255 | 2.147 | 2.083 | 2.087 | GLM Poisson | 3.0% |
| Cartellini | 1.162 | 1.121 | 1.111 | 1.105 | Gradient Boosting | 1.4% |

## Intervalli di previsione al 80% (modello scelto)

Con statistiche intere l'intervallo non può contenere esattamente l'80% di probabilità:
la colonna *Probabilità dichiarata* è quella che il modello assegna davvero all'intervallo.
Il modello è ben calibrato se la *Copertura reale* è vicina alla probabilità dichiarata.

| Statistica | Ampiezza media | Probabilità dichiarata | Copertura reale | Distribuzione |
|---|---|---|---|---|
| Gol | 2.9 | 93.8% | 94.1% | Poisson |
| Tiri | 11.3 | 83.8% | 83.5% | Binomiale Negativa (k = 22) |
| Tiri in porta | 5.5 | 87.3% | 87.2% | Binomiale Negativa (k = 46) |
| Falli | 8.8 | 84.6% | 83.5% | Binomiale Negativa (k = 588) |
| Corner | 6.6 | 86.5% | 86.8% | Binomiale Negativa (k = 12) |
| Cartellini | 3.7 | 92.0% | 92.5% | Poisson |
