#importar os pacotes
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import randint
from sklearn.model_selection import train_test_split 

#definindo os valores de diâmetro e de preços
dic1 = {"diametro": [x for x in np.random.randint(1,50,1000)] , "preco": []}

for x in dic1["diametro"]:
  if 0 <= x < 10:
    dic1["preco"].append(randint(2,11))

  if 10 <= x < 20:
    dic1["preco"].append(randint(11,26))

  if 20 <= x < 30:
    dic1["preco"].append(randint(26,50))

  if 30 <= x < 40:
    dic1["preco"].append(randint(50,80))

  if 40 <= x < 50:
    dic1["preco"].append(randint(80,100))
    
 #transforma o dic em dataframe
 data = pd.DataFrame(dic1)
 
 #divide o banco de dados de forma randômica para as variáveis de treino e teste
x_treino, x_teste, y_treino, y_teste = train_test_split(data["diametro"],data["preco"], test_size = 0.3)

x_treino = np.array(x_treino).reshape(-1,1)
y_treino = np.array(y_treino).reshape(-1,1)


#treina o modelo
model.fit(x_treino, y_treino)

#testa o algoritmo
preco_pred = model.predict(np.array(x_teste).reshape(-1,1))

#mostra o coeficiente angular
model.coef_

#verifica a variancia
r2_score(y_teste, preco_pred)

#plota o gráfico
plt.scatter(x_teste,y_teste, color='black')
plt.plot(x_teste,preco_pred,color='blue', linewidth=3)
plt.title("Preço das Pizzas por Diâmetro(cm)")
plt.ylabel("Diâmetro(cm)")
plt.xlabel("Preço")
plt.show()
 

