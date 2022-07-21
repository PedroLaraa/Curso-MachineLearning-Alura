from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score

# Features
# 1. Pelo longo
# 2. Perna curta
# 3. Faz auau

porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1, 1, 0]

cachorro1 = [0, 1, 1]
cachorro2 = [1, 0, 1]
cachorro3 = [1, 1, 1]

treino_x = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
treino_y = [1, 1, 1, 0, 0, 0]

model = LinearSVC()

model.fit(treino_x, treino_y)
animal_misterioso = [1, 1, 1]

print(model.predict([animal_misterioso]))

misterio1 = [1,1,1]

misterio2 = [0,0,1]

misterio3 = [1,1,0]

teste_x = [misterio1, misterio2, misterio3]

previsoes = model.predict(teste_x)

print(previsoes)

teste_y = [0, 1, 1]

corretos = (previsoes == teste_y).sum()
total = len(teste_x)
taxa_de_acerto = corretos / total
print("Taxa de acerto: %.2f" % (taxa_de_acerto * 100))

taxa_de_acerto = accuracy_score(teste_y, previsoes)
print("Taxa de acerto: %.2f" % (taxa_de_acerto * 100))
