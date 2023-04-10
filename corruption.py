# UNIVERSIDADE DE SÃO PAULO
# INTRODUÇÃO AO PYTHON E MACHINE LEARNING
# GLM - REGRESSÃO SIMPLES E MÚLTIPLA
# Prof. Dr. Luiz Paulo Fávero

#!/usr/bin/env python
# coding: utf-8


# In[ ]: Importação dos pacotes necessários
    
import pandas as pd # manipulação de dado em formato de dataframe
import seaborn as sns # biblioteca de visualização de informações estatísticas
import matplotlib.pyplot as plt # biblioteca de visualização de dados
import statsmodels.api as sm # biblioteca de modelagem estatística
import numpy as np # biblioteca para operações matemáticas multidimensionais
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
import plotly.graph_objs as go # gráfico 3D
from scipy.stats import pearsonr # correlações de Pearson
from sklearn.preprocessing import LabelEncoder # transformação de dados

# In[ ]:
#############################################################################
#         REGRESSÃO COM UMA VARIÁVEL EXPLICATIVA (X) QUALITATIVA            #
#             EXEMPLO 03 - CARREGAMENTO DA BASE DE DADOS                    #
#############################################################################

df_corrupcao = pd.read_csv('corrupcao.csv',delimiter=',',encoding='utf-8')
df_corrupcao

#Características das variáveis do dataset
df_corrupcao.info()

#Estatísticas univariadas
df_corrupcao.describe()

# Estatísticas univariadas por região
df_corrupcao.groupby('regiao').describe()

#Tabela de frequências da variável 'regiao'
#Função 'value_counts' do pacote 'pandas' sem e com o argumento 'normalize'
#para gerar, respectivamente, as contagens e os percentuais
contagem = df_corrupcao['regiao'].value_counts(dropna=False)
percent = df_corrupcao['regiao'].value_counts(dropna=False, normalize=True)
pd.concat([contagem, percent], axis=1, keys=['contagem', '%'], sort=False)


# In[ ]: Conversão dos dados de 'regiao' para dados numéricos, a fim de
#se mostrar a estimação de modelo com o problema da ponderação arbitrária

label_encoder = LabelEncoder()
df_corrupcao['regiao_numerico'] = label_encoder.fit_transform(df_corrupcao['regiao'])
df_corrupcao['regiao_numerico'] = df_corrupcao['regiao_numerico'] + 1
df_corrupcao.head(10)

#A nova variável 'regiao_numerico' é quantitativa (ERRO!), fato que
#caracteriza a ponderação arbitrária!
df_corrupcao['regiao_numerico'].info()
df_corrupcao.describe()


# In[ ]: Modelando com a variável preditora numérica, resultando na
#estimação ERRADA dos parâmetros
#PONDERAÇÃO ARBITRÁRIA!
modelo_corrupcao_errado = sm.OLS.from_formula("cpi ~ regiao_numerico",
                                              df_corrupcao).fit()

#Parâmetros do modelo
modelo_corrupcao_errado.summary()

#Calculando os intervalos de confiança com nível de significância de 5%
modelo_corrupcao_errado.conf_int(alpha=0.05)


# In[ ]: Plotando os fitted values do modelo_corrupcao_errado considerando,
#PROPOSITALMENTE, a ponderação arbitrária, ou seja, assumindo que as regiões
#representam valores numéricos (América do Sul = 1; Ásia = 2; EUA e Canadá = 3;
#Europa = 4; Oceania = 5).

ax =sns.lmplot(
    data=df_corrupcao,
    x="regiao_numerico", y="cpi",
    height=10
)
def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(point['val']) + " " +
                str(point['y']))
plt.title('Resultado da Ponderação Arbitrária', fontsize=16)
plt.xlabel('Região', fontsize=14)
plt.ylabel('Corruption Perception Index', fontsize=14)
label_point(x = df_corrupcao['regiao_numerico'],
            y = df_corrupcao['cpi'],
            val = df_corrupcao['pais'],
            ax = plt.gca()) 


# In[ ]: Dummizando a variável 'regiao'. O código abaixo automaticamente fará: 
# a)o estabelecimento de dummies que representarão cada uma das regiões do dataset; 
# b)removerá a variável original a partir da qual houve a dummização; 
# c)estabelecerá como categoria de referência a primeira categoria, ou seja,
# a categoria 'America_do_sul' por meio do argumento 'drop_first=True'.

df_corrupcao_dummies = pd.get_dummies(df_corrupcao, columns=['regiao'],
                                      drop_first=True)

df_corrupcao_dummies.head(10)

#A variável 'regiao' está inicialmente definida como 'object' no dataset
df_corrupcao.info()
#O procedimento atual também poderia ter sido realizado em uma variável
#dos tipos 'category' ou 'string'. Para fins de exemplo, podemos transformar a
#variável 'regiao' para 'category' ou 'string' e comandar o código anterior:
df_corrupcao['regiao'] = df_corrupcao['regiao'].astype("category")
df_corrupcao.info()
df_corrupcao['regiao'] = df_corrupcao['regiao'].astype("string")
df_corrupcao.info()


# In[ ]: Estimação do modelo de regressão múltipla com n-1 dummies

modelo_corrupcao_dummies = sm.OLS.from_formula("cpi ~ regiao_Asia + \
                                              regiao_EUA_e_Canada + \
                                              regiao_Europa + \
                                              regiao_Oceania",
                                              df_corrupcao_dummies).fit()

#Parâmetros do modelo
modelo_corrupcao_dummies.summary()

#Outro método de estimação (sugestão de uso para muitas dummies no dataset)
# Definição da fórmula utilizada no modelo
lista_colunas = list(df_corrupcao_dummies.drop(columns=['cpi','pais','regiao_numerico']).columns)
formula_dummies_modelo = ' + '.join(lista_colunas)
formula_dummies_modelo = "cpi ~ " + formula_dummies_modelo
print("Fórmula utilizada: ",formula_dummies_modelo)

modelo_corrupcao_dummies = sm.OLS.from_formula(formula_dummies_modelo,
                                               df_corrupcao_dummies).fit()

#Parâmetros do modelo
modelo_corrupcao_dummies.summary()


# In[ ]: Plotando o modelo_corrupcao_dummies de forma interpolada

#Fitted values do 'modelo_corrupcao_dummies' no dataset 'df_corrupcao_dummies'
df_corrupcao_dummies['fitted'] = modelo_corrupcao_dummies.fittedvalues
df_corrupcao_dummies.head()


# In[ ]: Gráfico propriamente dito

from scipy import interpolate

plt.figure(figsize=(10,10))

df2 = df_corrupcao_dummies[['regiao_numerico','fitted']].groupby(['regiao_numerico']).median().reset_index()
x = df2['regiao_numerico']
y = df2['fitted']

tck = interpolate.splrep(x, y, k=2)
xnew = np.arange(1,5,0.1) 
ynew = interpolate.splev(xnew, tck, der=0)

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x']+.02, point['y'], str(point['val']) + " " + str(point['y']))

plt.scatter(df_corrupcao_dummies['regiao_numerico'], df_corrupcao_dummies['cpi'])
plt.scatter(df_corrupcao_dummies['regiao_numerico'], df_corrupcao_dummies['fitted'])
plt.plot(xnew, ynew)
plt.title('Ajuste Não Linear do Modelo com Variáveis Dummy', fontsize=16)
plt.xlabel('Região', fontsize=14)
plt.ylabel('Corruption Perception Index', fontsize=14)
label_point(x = df_corrupcao['regiao_numerico'],
            y = df_corrupcao['cpi'],
            val = df_corrupcao['pais'],
            ax = plt.gca())
