import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from numpy import random

from sklearn.dummy import DummyClassifier

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

from sklearn.feature_selection import RFECV

from sklearn.feature_selection import RFE

import seaborn as sns

resultados_dos_exames = pd.read_csv('exames.csv')

SEED = 1234
random.seed(SEED)

valores_exames = resultados_dos_exames.drop(columns=['id', 'diagnostico'])
diagnostico = resultados_dos_exames.diagnostico

valores_exames_v1 = valores_exames.drop(columns= "exame_33")

valores_exames_v6 = valores_exames_v1.drop(columns=(["exame_4", "exame_29", "exame_3", "exame_24"]))

treino_x, teste_x, treino_y, teste_y = train_test_split(valores_exames_v6, diagnostico, test_size=0.3)

classificador = RandomForestClassifier(n_estimators= 100)
classificador.fit(treino_x, treino_y)
# print('Resultado da classificação: %.2f%% ' %(classificador.score(teste_x, teste_y) *100))

classificador_bobo = DummyClassifier(strategy='most_frequent')
classificador_bobo.fit(treino_x, treino_y)

# print('Resultado da classificação boba: %.2f%%' %(classificador_bobo.score(teste_x, teste_y) *100))

padronizador = StandardScaler()
padronizador.fit(valores_exames_v1)
valores_exames_v2 = padronizador.transform(valores_exames_v1)
valores_exames_v2 = pd.DataFrame(data= valores_exames_v2, columns=valores_exames_v1.keys())

valores_exames_v3 = valores_exames_v2.drop(columns=['exame_29', 'exame_4'])

# def classificar(valores):

#     treino_x, teste_x, treino_y, teste_y = train_test_split(valores, diagnostico, test_size=0.3)

#     classificador = RandomForestClassifier(n_estimators= 100)
#     classificador.fit(treino_x, treino_y)
#     print('Resultado da classificação: %.2f%% ' %(classificador.score(teste_x, teste_y) *100))

# def grafico_violino(valores, inicio, fim):

#     dados_plot = pd.concat([diagnostico, valores.iloc[:,inicio:fim]], axis = 1)
#     dados_plot = pd.melt(dados_plot, id_vars='diagnostico', var_name='exames', value_name='valores')

#     plt.figure(figsize=(10, 10))
#     sns.violinplot(x='exames', y='valores', hue='diagnostico' ,data=dados_plot, split=True)

#     plt.xticks(rotation=90)

#     # EXIBE O GRÁFICO
#     plt.show()

# grafico_violino(valores_exames_v2, 10, 21)

# classificar(valores_exames_v3)

treino_x, teste_x, treino_y, teste_y = train_test_split(valores_exames_v6, diagnostico, test_size = 0.3)

classificador = RandomForestClassifier(n_estimators=100, random_state = 1234)
classificador.fit(treino_x, treino_y)

# ALGORITMO COM RFE

selecionador_rfe = RFE(estimator = classificador, n_features_to_select = 2, step = 1)
selecionador_rfe.fit(treino_x, treino_y)
treino_rfe = selecionador_rfe.transform(treino_x)
teste_rfe = selecionador_rfe.transform(teste_x)
classificador.fit(treino_rfe, treino_y)

valores_exames_v7 = selecionador_rfe.transform(valores_exames_v6)
valores_exames_v7.shape

plt.figure(figsize=(14, 8))
sns.scatterplot(x = valores_exames_v7[:,0], y = valores_exames_v7[:,1], hue = diagnostico)

# ALGORITMO COM RFECV

selecionador_rfecv = RFECV(estimator = classificador, step = 1, scoring='accuracy')
selecionador_rfecv.fit(treino_x, treino_y)
treino_rfecv = selecionador_rfecv.transform(treino_x)
teste_rfecv = selecionador_rfecv.transform(teste_x)
classificador.fit(treino_rfecv, treino_y)

# matriz_confusao = confusion_matrix(teste_y, classificador.predict(teste_rfecv))
# plt.figure(figsize = (10, 8))
# sns.set(font_scale = 2)
# sns.heatmap(matriz_confusao, annot = True, fmt = "d").set(xlabel = "Predição", ylabel = "Real")

print("Resultado da classificação %.2f%%" % (classificador.score(teste_rfecv, teste_y)* 100))

plt.figure(figsize=(14, 8))
plt.xlabel('Número de exames')
plt.ylabel('Acurácia')

plt.plot(range(1, len(selecionador_rfecv.grid_scores_) +1), selecionador_rfecv.grid_scores_)

# EXIBE O GRÁFICO
plt.show()
