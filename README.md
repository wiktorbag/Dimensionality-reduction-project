# Porównanie technik redukcji wielowymiarowości

## Opis projektu
Celem projektu jest analiza i porównanie różnych metod redukcji wymiarowości na danych numerycznych. Skupiamy się na **ekstrakcji nowych zmiennych (feature extraction)** oraz ocenie jakości odwzorowania danych w przestrzeni o niższym wymiarze.

Porównujemy zarówno metody klasyczne, jak i nowoczesne podejścia nieliniowe.

---

## Cele projektu

- Porównanie metod redukcji wymiarowości pod względem jakości reprezentacji danych  
- Analiza wpływu redukcji na strukturę danych i separowalność klas  
- Ocena przydatności metod w zadaniach uczenia maszynowego  
- Zbadanie kompromisu między interpretowalnością a jakością odwzorowania  

---

##  Porównywane metody

### 🔹 Klasyczne
- PCA (Principal Component Analysis)  
- LDA (Linear Discriminant Analysis)  

### 🔹 Nowoczesne / nieliniowe
- t-SNE (t-distributed Stochastic Neighbor Embedding)  
- UMAP (Uniform Manifold Approximation and Projection)  
- Isomap  

---

## 📂 Dane

Projekt wykorzystuje dane numeryczne, np.:

- Iris  
- MNIST (po przekształceniu do wektorów)  
- dane syntetyczne (blobs, moons)  

Możliwe rozszerzenie:
- rzeczywiste dane (np. finansowe, medyczne)

---

## Metodologia

1. Wstępne przetwarzanie danych (standaryzacja)  
2. Redukcja wymiarowości (do 2D / 3D)  
3. Wizualizacja wyników  
4. Ewaluacja jakości odwzorowania  
5. (opcjonalnie) zastosowanie klasyfikacji  

---

## Metryki ewaluacji

- Trustworthiness  
- Silhouette Score  
- Accuracy klasyfikatora po redukcji  
- Czas obliczeń  

---

##  Wizualizacje

- Wykresy rozrzutu (2D, 3D)  
- Porównania między metodami  
- Analiza struktury danych po redukcji  
