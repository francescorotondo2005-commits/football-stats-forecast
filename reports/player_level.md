# Dalle squadre ai giocatori: esperimento sui dati aperti Wyscout 2017-18

Domanda: sapere **chi scende in campo** migliora la previsione delle statistiche di una squadra,
rispetto alle sole medie di squadra? Stesso tipo di modello di footstats (GLM di Poisson),
addestrato sulle partite fino al 31/12/2017 (1330 righe squadra-partita, dalla 6ª giornata) e valutato sul resto della stagione (1834 righe).

- **A**: solo medie di squadra (forma, livello, casa/trasferta, avversario, campionato).
- **B1**: A + rendimento dei giocatori dell'undici della partita precedente ("formazione abituale").
- **B2**: A + rendimento dei giocatori dell'undici ufficiale (noto circa un'ora prima della partita).

Il rendimento di ogni giocatore è la sua media ogni 90 minuti nelle partite precedenti, ristretta
verso la media del suo ruolo quando ha giocato poco (peso pari a 5 partite intere).

## Errore medio assoluto (MAE) sul girone di ritorno

| Statistica | A: squadra | B1: + formazione abituale | B2: + formazione ufficiale | B2 vs A | Intervallo 95% di B2 vs A |
|---|---|---|---|---|---|
| Gol | 0,893 | 0,897 | 0,896 | +0,4% | da -0,4% a +1,3% |
| Tiri | 3,215 | 3,212 | 3,210 | -0,2% | da -1,2% a +0,8% |
| Tiri in porta | 1,784 | 1,787 | 1,785 | +0,0% | da -0,9% a +1,0% |
| Falli | 3,150 | 3,145 | 3,168 | +0,6% | da -0,1% a +1,3% |
| Corner | 2,064 | 2,066 | 2,072 | +0,4% | da -0,1% a +0,9% |
| Cartellini | 1,140 | 1,132 | 1,144 | +0,4% | da -0,1% a +0,8% |

L'intervallo viene da un bootstrap sulle partite del test: se contiene lo zero, la differenza
non è distinguibile dal caso con una sola stagione di dati.

## Quando la formazione cambia molto

In media una squadra cambia 2,8 titolari rispetto alla partita precedente
e nel 31% delle partite ne cambia almeno 4. Analisi diagnostica, non usata per
scegliere il modello: MAE di A e B2 separando le partite con pochi o molti cambi, e correlazione
tra lo scostamento dell'undici dal solito (rendimento atteso dei titolari di oggi rispetto a
quelli della partita precedente) e l'errore del modello di squadra.

| Statistica | ≤ 3 cambi: A | ≤ 3 cambi: B2 | ≥ 4 cambi: A | ≥ 4 cambi: B2 | Correlazione scostamento-errore |
|---|---|---|---|---|---|
| Gol | 0,870 | 0,875 (+0,6%) | 0,942 | 0,944 (+0,1%) | +0,044 |
| Tiri | 3,242 | 3,250 (+0,3%) | 3,155 | 3,121 (-1,1%) | +0,084 |
| Tiri in porta | 1,795 | 1,801 (+0,3%) | 1,760 | 1,751 (-0,5%) | +0,047 |
| Falli | 3,153 | 3,163 (+0,3%) | 3,146 | 3,180 (+1,1%) | +0,023 |
| Corner | 2,080 | 2,091 (+0,5%) | 2,028 | 2,032 (+0,2%) | – |
| Cartellini | 1,132 | 1,139 (+0,7%) | 1,159 | 1,156 (-0,2%) | +0,010 |

## Controllo dei dati

Medie per squadra e partita nel 2017-18: eventi Wyscout contro football-data.co.uk.

| Statistica | Wyscout | football-data |
|---|---|---|
| Tiri | 11,785 | 12,396 |
| Tiri in porta | 4,253 | 4,323 |
| Falli | 12,762 | 12,653 |
| Corner | 4,978 | 4,982 |
| Cartellini | 2,015 | 2,035 |
| Gol | 1,355 | 1,355 |

Dati: Pappalardo et al. (2019), *A public data set of spatio-temporal match events in soccer
competitions*, Scientific Data 6:236, licenza CC BY 4.0
([figshare](https://figshare.com/collections/Soccer_match_event_dataset/4415000)).
