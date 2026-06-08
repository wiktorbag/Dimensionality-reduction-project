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

Algorytmy zostaną porównane na 3 rodzajach zbiorów danych 

1. klasyczny zbiór tabelaryczny iris
2. zbiór obrazów fashion-mnist
3. zbiór embeddingów tekstowych
