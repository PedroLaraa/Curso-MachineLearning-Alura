import pandas as pd

import seaborn as sns

uri = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"

dados = pd.read_csv(uri)

map = {
    'expected_hours': 'horas_esperadas',
    'price': 'preco',
    'unfinished': 'nao_finalizado'
}

dados.rename(columns=map, inplace=True)

print(dados.head())

sns.scatterplot(x='horas_esperadas', y='preco', data=dados)

