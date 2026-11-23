# Risultati sul test (2024-25, 2025-26, partite mai viste in training)

## Errore medio assoluto (MAE) del valore atteso

| Statistica | Media campionato | Media mobile | GLM Poisson | Gradient Boosting | Ensemble | Scelto (in validazione) | Miglioramento vs media mobile |
|---|---|---|---|---|---|---|---|
| Gol | 0.986 | 0.932 | 0.903 | 0.904 | 0.902 | Ensemble | 3.2% |
| Tiri | 4.136 | 3.768 | 3.539 | 3.565 | 3.544 | Ensemble | 5.9% |
| Tiri in porta | 1.930 | 1.796 | 1.730 | 1.735 | 1.729 | Ensemble | 3.7% |
| Falli | 3.056 | 2.851 | 2.819 | 2.809 | 2.807 | Ensemble | 1.5% |
| Corner | 2.243 | 2.147 | 2.082 | 2.084 | 2.080 | GLM Poisson | 3.1% |
| Cartellini | 1.132 | 1.121 | 1.109 | 1.103 | 1.104 | Gradient Boosting | 1.6% |

## Intervalli di previsione al 80% (modello scelto)

Con statistiche intere l'intervallo non può contenere esattamente l'80% di probabilità:
la colonna *Probabilità dichiarata* è quella che il modello assegna davvero all'intervallo.
Il modello è ben calibrato se la *Copertura reale* è vicina alla probabilità dichiarata.

| Statistica | Ampiezza media | Probabilità dichiarata | Copertura reale | Distribuzione |
|---|---|---|---|---|
| Gol | 2.9 | 93.8% | 94.4% | Poisson |
| Tiri | 11.2 | 83.9% | 83.7% | Binomiale Negativa (k = 22) |
| Tiri in porta | 5.5 | 87.2% | 87.0% | Binomiale Negativa (k = 50) |
| Falli | 8.8 | 84.6% | 83.8% | Binomiale Negativa (k = 796) |
| Corner | 6.6 | 86.5% | 86.8% | Binomiale Negativa (k = 12) |
| Cartellini | 3.7 | 91.8% | 92.5% | Poisson |
