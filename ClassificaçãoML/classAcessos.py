
#  Importa as bibliotecas necessárias

import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Link do dataset
uri = 'https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv'
# Faz a leitura do dataset
dados = pd.read_csv(uri)

# Dict para renomear as colunas
mapa = {
    'home': 'principal',
    'how_it_works': 'como_funciona',
    'contact': 'contato',
    'bought': 'comprou'
}

# Renomeia as colunas
dados.rename(columns=mapa, inplace=True)

# Separa os dados
x = dados[["principal", "como_funciona", "contato"]]
y = dados[["comprou"]]

print('-----------------------------------------------------')
print(x.head())
print('-----------------------------------------------------')
print(y.head())
SEED = 20

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, random_state = SEED, test_size = 0.25, stratify = y)
print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)
