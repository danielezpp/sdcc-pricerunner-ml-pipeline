# sdcc-pricerunner-ml-pipeline
Cloud-native, serverless Machine Learning pipeline using AWS services, featuring event-driven preprocessing, asynchronous training, and real-time/batch inference.

Progetto sviluppato per il corso di Sistemi Distribuiti e Cloud Computing.

Il repository contiene il codice sorgente del backend serverless
basato su AWS (S3, Lambda, API Gateway).

## Struttura
- src/common: configurazioni e utility condivise
- src/preprocess: preprocessing event-driven
- src/train: training asincrono dei modelli
- src/inference: inferenza sincrona e batch

## Note
Le funzioni Lambda sono invocate tramite eventi S3 e API Gateway.

