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
#                          REGRESSÃO LINEAR SIMPLES                         #
#                  EXEMPLO 01 - CARREGAMENTO DA BASE DE DADOS               #
#############################################################################
    
df = pd.read_csv('tempodist.csv', delimiter=',')
df

#Características das variáveis do dataset
df.info()

#Estatísticas univariadas
df.describe()


# In[ ]: Gráfico de dispersão

#Regressão linear que melhor se adequa às obeservações: função 'sns.lmplot'

plt.figure(figsize=(20,10))
sns.lmplot(data=df, x='distancia', y='tempo', ci=False)
plt.xlabel('Distância', fontsize=20)
plt.ylabel('Tempo', fontsize=20)
plt.legend(['Valores Reais', 'Fitted Values'], fontsize=14)
plt.show


# In[ ]: Gráfico de dispersão

#Regressão linear que melhor se adequa às obeservações: função 'sns.regplot'

plt.figure(figsize=(20,10))
sns.regplot(data=df, x='distancia', y='tempo', ci=False, color='purple')
plt.xlabel('Distância', fontsize=20)
plt.ylabel('Tempo', fontsize=20)
plt.legend(['Valores Reais', 'Fitted Values'], fontsize=24)
plt.show


# In[ ]: Estimação do modelo de regressão linear simples

#Estimação do modelo
modelo = sm.OLS.from_formula('tempo ~ distancia', df).fit()

#Observação dos parâmetros resultantes da estimação
modelo.summary()


# In[ ]: Salvando fitted values (variável yhat) 
# e residuals (variável erro) no dataset

df['yhat'] = modelo.fittedvalues
df['erro'] = modelo.resid
df


# In[ ]: Gráfico didático para visualizar o conceito de R²

y = df['tempo']
yhat = df['yhat']
x = df['distancia']
mean = np.full(x.shape[0] , y.mean(), dtype=int)

for i in range(len(x)-1):
    plt.plot([x[i],x[i]], [yhat[i],y[i]],'--', color='#2ecc71')
    plt.plot([x[i],x[i]], [yhat[i],mean[i]], ':', color='#9b59b6')
    plt.plot(x, y, 'o', color='#2c3e50')
    plt.axhline(y = y.mean(), color = '#bdc3c7', linestyle = '-')
    plt.plot(x,yhat, color='#34495e')
    plt.title('R2: ' + str(round(modelo.rsquared,4)))
    plt.xlabel("Distância")
    plt.ylabel("Tempo")
    plt.legend(['Erro = Y - Yhat', 'Yhat - Ymédio'], fontsize=10)
plt.show()


# In[ ]: Cálculo manual do R²

R2 = ((df['yhat']-
       df['tempo'].mean())**2).sum()/(((df['yhat']-
                                        df['tempo'].mean())**2).sum()+
                                        (df['erro']**2).sum())

round(R2,4)


# In[ ]: Coeficiente de ajuste (R²) é a correlação ao quadrado

#Correlação de Pearson
df[['tempo','distancia']].corr()

#R²
(df[['tempo','distancia']].corr())**2

#R² de maneira direta
modelo.rsquared


# In[ ]: Modelo auxiliar para mostrar R² igual a 100% (para fins didáticos)

#Estimação do modelo com yhat como variável dependente,
#resultará em uma modelo com R² igual a 100%
modelo_auxiliar = sm.OLS.from_formula('yhat ~ distancia', df).fit()

#Parâmetros resultantes da estimação
modelo_auxiliar.summary()


# In[ ]:Gráfico mostrando o perfect fit

plt.figure(figsize=(20,10))
sns.regplot(data=df, x='distancia', y='yhat', ci=False, color='purple')
plt.xlabel('Distância', fontsize=20)
plt.ylabel('Tempo', fontsize=20)
plt.legend(['Valores Reais', 'Fitted Values'], fontsize=24)
plt.show


# In[ ]: Voltando ao nosso modelo original


#Plotando o intervalo de confiança de 90%
plt.figure(figsize=(20,10))
sns.regplot(data=df, x='distancia', y='tempo', ci=90, color='purple')
plt.xlabel('Distância', fontsize=20)
plt.ylabel('Tempo', fontsize=20)
plt.legend(['Valores Reais', 'Fitted Values'], fontsize=24)
plt.show

#%%
#Plotando o intervalo de confiança de 95%
plt.figure(figsize=(20,10))
sns.regplot(data=df, x='distancia', y='tempo', ci=95, color='purple')
plt.xlabel('Distância', fontsize=20)
plt.ylabel('Tempo', fontsize=20)
plt.legend(['Valores Reais', 'Fitted Values'], fontsize=24)
plt.show

#%%
#Plotando o intervalo de confiança de 99%
plt.figure(figsize=(20,10))
sns.regplot(data=df, x='distancia', y='tempo', ci=99, color='purple')
plt.xlabel('Distância', fontsize=20)
plt.ylabel('Tempo', fontsize=20)
plt.legend(['Valores Reais', 'Fitted Values'], fontsize=24)
plt.show

#%%
#Plotando o intervalo de confiança de 99,999%
plt.figure(figsize=(20,10))
sns.regplot(data=df, x='distancia', y='tempo', ci=99.999, color='purple')
plt.xlabel('Distância', fontsize=20)
plt.ylabel('Tempo', fontsize=20)
plt.legend(['Valores Reais', 'Fitted Values'], fontsize=24)
plt.show


# In[ ]: Calculando os intervalos de confiança

#Nível de significância de 10% / Nível de confiança de 90%
modelo.conf_int(alpha=0.1)

#Nível de significância de 5% / Nível de confiança de 95%
modelo.conf_int(alpha=0.05)

#Nível de significância de 1% / Nível de confiança de 99%
modelo.conf_int(alpha=0.01)

#Nível de significância de 0,001% / Nível de confiança de 99,999%
modelo.conf_int(alpha=0.00001)


# In[ ]: Fazendo predições em modelos OLS
#Ex.: Qual seria o tempo gasto, em média, para percorrer a distância de 25km?

modelo.predict(pd.DataFrame({'distancia':[25]}))

#Cálculo manual - mesmo valor encontrado
5.8784 + 1.4189*(25)


# In[ ]: Nova modelagem para o mesmo exemplo, com novo dataset que
#contém replicações

#Quantas replicações de cada linha você quer? -> função 'np.repeat'
df_replicado = pd.DataFrame(np.repeat(df.values, 3, axis=0))
df_replicado.columns = df.columns
df_replicado


# In[ ]: Estimação do modelo com valores replicados

modelo_replicado = sm.OLS.from_formula('tempo ~ distancia',
                                       df_replicado).fit()

#Parâmetros do modelo
modelo_replicado.summary()


# In[ ]: Calculando os novos intervalos de confiança

#Nível de significância de 5% / Nível de confiança de 95%
modelo_replicado.conf_int(alpha=0.05)


# In[ ]: Plotando o novo gráfico com intervalo de confiança de 95%
#Note o estreitamento da amplitude dos intervalos de confiança!

plt.figure(figsize=(20,10))
sns.regplot(data=df_replicado, x='distancia', y='tempo', ci=95, color='purple')
plt.xlabel('Distância', fontsize=20)
plt.ylabel('Tempo', fontsize=20)
plt.legend(['Valores Reais', 'Fitted Values'], fontsize=24)
plt.show


# In[ ]: PROCEDIMENTO ERRADO: ELIMINAR O INTERCEPTO QUANDO ESTE NÃO SE MOSTRAR
#ESTATISTICAMENTE SIGNIFICANTE

modelo_errado = sm.OLS.from_formula('tempo ~ 0 + distancia', df).fit()

#Parâmetros do modelo
modelo_errado.summary()


# In[ ]: Comparando os parâmetros do modelo inicial (objeto 'modelo')
#com o 'modelo_errado' pela função 'summary_col' do pacote
#'statsmodels.iolib.summary2'

summary_col([modelo, modelo_errado])

#Outro modo mais completo também pela função 'summary_col'
summary_col([modelo, modelo_errado],
            model_names=["MODELO INICIAL","MODELO ERRADO"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs))
        })


# In[ ]: Gráfico didático para visualizar o viés decorrente de se eliminar
# erroneamente o intercepto em modelos regressivos

x = df['distancia']
y = df['tempo']

yhat = df['yhat']
yhat_errado = modelo_errado.fittedvalues

plt.plot(x, y, 'o', color='dimgray')
plt.plot(x, yhat, color='limegreen')
plt.plot(x, yhat_errado, color='red')
plt.xlabel("Distância")
plt.ylabel("Tempo")
plt.legend(['Valores Observados','Fitted Values - Modelo Coreto',
            'Fitted Values - Modelo Errado'], fontsize=9)
plt.show()