from dados import carregar_acessos

X, Y = carregar_acessos()

x_treino = X[:90]
y_treino = Y[:90]

x_teste = X[-9:]
y_teste = Y[-9:]

from sklearn.naive_bayes import MultinomialNB
modelo = MultinomialNB()
modelo.fit(x_treino, y_treino)

resultado = modelo.predict(x_teste)

diferencas = resultado - y_teste

acertos = [d for d in diferencas if d == 0]

total_de_acertos = len(acertos)

total_de_elementos = len(x_teste)

taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

print('-------------------------------------------')
print('Testando com', len(x_teste), 'elementos')
print('-------------------------------------------')
print('Taxa de acerto: ',taxa_de_acerto, '%')
print('-------------------------------------------')
print('Total de acertos: ',total_de_acertos)
print('-------------------------------------------')
