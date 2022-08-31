import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from numpy import random
from sklearn.dummy import DummyClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

resultados_dos_exames = pd.read_csv('exames.csv')

SEED = 123143
random.seed(SEED)

valores_exames = resultados_dos_exames.drop(columns=['id', 'diagnostico'])
diagnostico = resultados_dos_exames.diagnostico

valores_exames_v1 = valores_exames.drop(columns= "exame_33")

treino_x, teste_x, treino_y, teste_y = train_test_split(valores_exames_v1, diagnostico, test_size=0.3)

classificador = RandomForestClassifier(n_estimators= 100)
classificador.fit(treino_x, treino_y)
print('Resultado da classificação: %.2f%% ' %(classificador.score(teste_x, teste_y) *100))

classificador_bobo = DummyClassifier(strategy='most_frequent')
classificador_bobo.fit(treino_x, treino_y)

# print('Resultado da classificação boba: %.2f%%' %(classificador_bobo.score(teste_x, teste_y) *100))

padronizador = StandardScaler()
padronizador.fit(valores_exames_v1)
valores_exames_v2 = padronizador.transform(valores_exames_v1)
valores_exames_v2 = pd.DataFrame(data= valores_exames_v2, columns=valores_exames_v1.keys())

valores_exames_v3 = valores_exames_v2.drop(columns=['exame_29', 'exame_4'])

def classificar(valores):

    treino_x, teste_x, treino_y, teste_y = train_test_split(valores, diagnostico, test_size=0.3)

    classificador = RandomForestClassifier(n_estimators= 100)
    classificador.fit(treino_x, treino_y)
    print('Resultado da classificação: %.2f%% ' %(classificador.score(teste_x, teste_y) *100))

def grafico_violino(valores, inicio, fim):

    dados_plot = pd.concat([diagnostico, valores.iloc[:,inicio:fim]], axis = 1)
    dados_plot = pd.melt(dados_plot, id_vars='diagnostico', var_name='exames', value_name='valores')

    plt.figure(figsize=(10, 10))
    sns.violinplot(x='exames', y='valores', hue='diagnostico' ,data=dados_plot, split=True)

    plt.xticks(rotation=90)

    # EXIBE O GRÁFICO
    plt.show()

grafico_violino(valores_exames_v2, 10, 21)

classificar(valores_exames_v3)
