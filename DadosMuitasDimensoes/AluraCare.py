import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from numpy import random
from sklearn.dummy import DummyClassifier

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
