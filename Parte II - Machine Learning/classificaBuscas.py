from tkinter import Y
import pandas as pd

dados = pd.read_csv('cursos.csv')
print(dados.head())

X = dados[['home','busca','logado']]
Y = dados['comprou']
