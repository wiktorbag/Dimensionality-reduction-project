

## Cele projektu

- Porównanie metod redukcji wymiarowości pod względem jakości reprezentacji danych  
- Analiza wpływu redukcji na strukturę danych i separowalność klas  
- Ocena przydatności metod w zadaniach uczenia maszynowego  
- Zbadanie kompromisu między interpretowalnością a jakością odwzorowania  


## BASELINE

Jako baseline potraktujemy model lda.

## Metryki do ewaluacji 

- NMI  
- Silhouette Score  
- Calinski-Harabsz
- Generalised Dunn
- trustworthiness — czy lokalni sąsiedzi zostali zachowani
-kNN accuracy w przestrzeni po redukcji


##  Wizualizacje

- Wykresy rozrzutu (2D, 3D)  
- Porównania między metodami  
- Analiza struktury danych po redukcji, np. clustering  


## Dane

Wykorzystamy kilka zbiorów danych m.in.

- Iris  https://www.kaggle.com/datasets/uciml/iris 
- Fashon- MNIST  https://www.tensorflow.org/datasets/catalog/fashion_mnist
- dane syntetyczne (blobs, moons)  

- CLIP embeddings
Możliwe rozszerzenie:
- https://huggingface.co/embedding-data

## 3 eksperymenty
  1. Iris redukcja -> klasyfikacja logistic regression+k-NN
  2. MNIST --||--
  3. Ekmbedingi redukcja -> kalstrowanie + sprawdzamy co się dzieje tu pomysły raczej w trakcie ekperymentu moxe wyjść a moźe nie wyjść
     
  
## Kamienie milowe

- 5.05. Przygotowanie danych 
- 19.05. Implementacja metod
- 26.05 Porównanie i wizualizacja

## Podział obowiązków 

- Preprocessing i porównanie + Kernel PCA - Michał Krajewski
- PCA i LDA - Zuzanna Skałka
- t-SNE + Autoencoder - Alicja Berent
- UMAP i Isomap - Wiktor Bagiński
