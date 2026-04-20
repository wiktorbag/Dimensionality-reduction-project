

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


##  Wizualizacje

- Wykresy rozrzutu (2D, 3D)  
- Porównania między metodami  
- Analiza struktury danych po redukcji, np. clustering  


## Dane

W projekcie wykorzystamy dane numeryczne, np.:

- Iris  https://www.kaggle.com/datasets/uciml/iris
- MNIST (po przekształceniu do wektorów)  https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html
- dane syntetyczne (blobs, moons)  

- geospatial embedding google
- https://link.springer.com/article/10.1007/s40747-021-00512-9
Możliwe rozszerzenie:
- https://huggingface.co/embedding-data
  
## Kamienie milowe

- 5.05. Przygotowanie danych 
- 19.05. Implementacja metod
- 26.05 Porównanie i wizualizacja

## Podział obowiązków 

- Preprocessing i porównanie - Michał Krajewski
- PCA i LDA - Zuzanna Skałka
- t-SNE - Alicja Berent
- UMAP i Isomap - Wiktor Bagiński
