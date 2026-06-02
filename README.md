# Porównanie technik redukcji wielowymiarowości

## Setup
      mkdir dim_red
      cd dim_red
      python3.14 -m venv .venv
      source .venv/bin/activate
      git init
      git clone https://github.com/wiktorbag/Dimensionality-reduction-project/
      pip install -r requirements.txt


## Opis projektu
Celem projektu jest analiza i porównanie różnych metod redukcji wymiarowości na danych numerycznych.
Skupiamy się na **ekstrakcji nowych zmiennych (feature extraction)** oraz ocenie jakości odwzorowania danych w przestrzeni o niższym wymiarze.

Chcemy porównać klasyczne metody dimensionality reduction LDA i PCA z metodami nowoczesnymi/ nieliniowymi t_SNE, UMAP i Isomap.
Dokonamy redukcji wymiarów za ich pomocą na tradycyjnych zbiorach danych MNIST i Iris oraz na embedingach tesktowych celem dokonania analizy skupień na nowo powstałych zmiennych. Porównanie obejmie zestawienie metryk jak i wizualizację. 
