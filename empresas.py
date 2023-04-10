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
#                        REGRESSÃO NÃO LINEAR MÚLTIPLA                      #
#                  EXEMPLO 05 - CARREGAMENTO DA BASE DE DADOS               #
#############################################################################

df_empresas = pd.read_csv('empresas.csv', delimiter=',')
df_empresas

#Características das variáveis do dataset
df_empresas.info()

#Estatísticas univariadas
df_empresas.describe()


# In[ ]: Matriz de correlações

#Maneira simples pela função 'corr'
corr = df_empresas.corr()
corr

#Maneira mais elaborada pela função 'rcorr' do pacote 'pingouin'
import pingouin as pg

corr2 = pg.rcorr(df_empresas, method='pearson',
                 upper='pval', decimals=4,
                 pval_stars={0.01: '***',
                             0.05: '**',
                             0.10: '*'})
corr2


# In[ ]: Mapa de calor da matriz de correlações

plt.figure(figsize=(15,10))
sns.heatmap(df_empresas.corr(), annot=True, cmap = plt.cm.viridis,
            annot_kws={'size':15})
plt.show()


# In[ ]: Distribuições das variáveis, scatters, valores das correlações e suas
#respectivas significâncias

def corrfunc(x, y, **kws):
    (r, p) = pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f} ".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)
    ax.annotate("p = {:.3f}".format(p),
                xy=(.4, .9), xycoords=ax.transAxes)

plt.figure(figsize=(15,10))
graph = sns.pairplot(df_empresas, diag_kind="kde")
graph.map(corrfunc)
plt.show()


# In[ ]: Estimando a Regressão Múltipla
modelo_empresas = sm.OLS.from_formula('retorno ~ disclosure +\
                                      endividamento + ativos +\
                                          liquidez', df_empresas).fit()

# Parâmetros do modelo
modelo_empresas.summary()

#Note que o parâmetro da variável 'endividamento' não é estatisticamente
#significante ao nível de significância de 5% (nível de confiança de 95%).

# Cálculo do R² ajustado (slide 15 da apostila)
r2_ajust = 1-((len(df_empresas.index)-1)/(len(df_empresas.index)-\
                                          modelo_empresas.params.count()))*\
    (1-modelo_empresas.rsquared)
r2_ajust # modo direto: modelo_empresas.rsquared_adj


# In[ ]: Procedimento Stepwise

# Instalação e carregamento da função 'stepwise' do pacote
#'statstests.process'
# Autores do pacote: Helder Prado Santos e Luiz Paulo Fávero
# https://stats-tests.github.io/statstests/
# pip install statstests
from statstests.process import stepwise

# Estimação do modelo por meio do procedimento Stepwise
modelo_step_empresas = stepwise(modelo_empresas, pvalue_limit=0.05)


# In[ ]: Teste de verificação da aderência dos resíduos à normalidade

# Teste de Shapiro-Wilk (n < 30)
#from scipy.stats import shapiro
#shapiro(modelo_step_empresas.resid)

# Teste de Shapiro-Francia (n >= 30)
# Instalação e carregamento da função 'shapiro_francia' do pacote
#'statstests.tests'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/
# pip install statstests
from statstests.tests import shapiro_francia
shapiro_francia(modelo_step_empresas.resid)

# Interpretação
teste_sf = shapiro_francia(modelo_step_empresas.resid) #criação do objeto 'teste_sf'
teste_sf = teste_sf.items() #retorna o grupo de pares de valores-chave no dicionário
method, statistics_W, statistics_z, p = teste_sf #definição dos elementos da lista (tupla)
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 #nível de significância
if p[1] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')


# In[ ]: Plotando os resíduos do 'modelo_step_empresas' e acrescentando
#uma curva normal teórica para comparação entre as distribuições

from scipy.stats import norm

plt.figure(figsize=(15,10))
sns.distplot(modelo_step_empresas.resid, fit=norm, kde=True, bins=20,
             color='goldenrod')
plt.xlabel('Resíduos do Modelo Linear', fontsize=16)
plt.ylabel('Frequência', fontsize=16)
plt.show()


# In[ ]: Transformação de Box-Cox

#Para o cálculo do lambda de Box-Cox
from scipy.stats import boxcox

#'x' é uma variável que traz os valores transformados (Y*)
#'lmbda' é o lambda de Box-Cox
x, lmbda = boxcox(df_empresas['retorno'])

print("Lambda: ",lmbda)


# In[ ]: Inserindo o lambda de Box-Cox no dataset para a estimação de um
#novo modelo

df_empresas['bc_retorno'] = x
df_empresas

#Verificação do cálculo, apenas para fins didáticos
df_empresas['bc_retorno2'] = ((df_empresas['retorno'])**(lmbda) - 1) / (lmbda)
df_empresas

del df_empresas['bc_retorno2']


# In[ ]: Estimando um novo modelo múltiplo com variável dependente
#transformada por Box-Cox

modelo_bc = sm.OLS.from_formula('bc_retorno ~ disclosure +\
                                endividamento + ativos +\
                                    liquidez', df_empresas).fit()

# Parâmetros do modelo
modelo_bc.summary()


# In[ ]: Aplicando o procedimento Stepwise no 'modelo_bc"

modelo_step_empresas_bc = stepwise(modelo_bc, pvalue_limit=0.05)

#Note que a variável 'disclosure' retorna ao modelo na forma funcional
#não linear!


# In[ ]: Verificando a normalidade dos resíduos do 'modelo_step_empresas_bc'

# Teste de Shapiro-Francia
shapiro_francia(modelo_step_empresas_bc.resid)

# Interpretação
teste_sf = shapiro_francia(modelo_step_empresas_bc.resid) #criação do objeto 'teste_sf'
teste_sf = teste_sf.items() #retorna o grupo de pares de valores-chave no dicionário
method, statistics_W, statistics_z, p = teste_sf #definição dos elementos da lista (tupla)
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 #nível de significância
if p[1] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')


# In[ ]: Plotando os novos resíduos do 'modelo_step_empresas_bc'

from scipy.stats import norm

plt.figure(figsize=(15,10))
sns.distplot(modelo_step_empresas_bc.resid, fit=norm, kde=True, bins=20,
             color='red')
plt.xlabel('Resíduos do Modelo Box-Cox', fontsize=16)
plt.ylabel('Frequência', fontsize=16)
plt.show()


# In[ ]: Resumo dos dois modelos obtidos pelo procedimento Stepwise
#(linear e com Box-Cox)

summary_col([modelo_step_empresas, modelo_step_empresas_bc],
            model_names=["STEPWISE","STEPWISE BOX-COX"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs))
        })

#CUIDADO!!! OS PARÂMETROS NÃO SÃO DIRETAMENTE COMPARÁVEIS!


# In[ ]: Fazendo predições com o 'modelo_step_empresas_bc'
# Qual é o valor do retorno, em média, para 'disclosure' igual a 50,
#'liquidez' igual a 14 e 'ativos' igual a 4000, ceteris paribus?

modelo_step_empresas_bc.predict(pd.DataFrame({'const':[1],
                                              'disclosure':[50],
                                              'ativos':[4000],
                                              'liquidez':[14]}))


# In[ ]: Não podemos nos esquecer de fazer o cálculo para a obtenção do
#fitted value de Y (variável 'retorno')

(3.702016 * lmbda + 1) ** (1 / lmbda)


# In[ ]: Salvando os fitted values de 'modelo_step_empresas' e
#'modelo_step_empresas_bc'

df_empresas['yhat_step_empresas'] = modelo_step_empresas.fittedvalues
df_empresas['yhat_step_empresas_bc'] = (modelo_step_empresas_bc.fittedvalues
                                        * lmbda + 1) ** (1 / lmbda)

#Visualizando os dois fitted values no dataset
#modelos 'modelo_step_empresas e modelo_step_empresas_bc
df_empresas[['empresa','retorno','yhat_step_empresas','yhat_step_empresas_bc']]


# In[ ]: Ajustes dos modelos: valores previstos (fitted values) X valores reais

from scipy.optimize import curve_fit

def objective(x, a, b, c, d, e, f):
    return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + f

xdata = df_empresas['retorno']
ydata_linear = df_empresas['yhat_step_empresas']
ydata_bc = df_empresas['yhat_step_empresas_bc']

plt.figure(figsize=(10,10))

popt, _ = curve_fit(objective, xdata, ydata_linear)
a, b, c, d, e, f = popt
x_line = np.arange(min(xdata), max(xdata), 1)
y_line = objective(x_line, a, b, c, d, e, f)
plt.plot(x_line, y_line, '--', color='goldenrod', linewidth=3)

popt, _ = curve_fit(objective, xdata, ydata_bc)
a, b, c, d, e, f = popt
x_line = np.arange(min(xdata), max(xdata), 1)
y_line = objective(x_line, a, b, c, d, e, f)
plt.plot(x_line, y_line, '--', color='red', linewidth=3)

plt.plot(xdata,xdata, color='gray', linestyle='-')
plt.scatter(xdata,ydata_linear, alpha=0.5, s=100, color='goldenrod')
plt.scatter(xdata,ydata_bc, alpha=0.5, s=100, color='red')
plt.title('Dispersão e Fitted Values dos Modelos Linear e Box-Cox',
          fontsize=17)
plt.xlabel('Valores Reais de Retorno', fontsize=16)
plt.ylabel('Fitted Values', fontsize=16)
plt.legend(['Stepwise','Stepwise com Box-Cox','45º graus'], fontsize=17)
plt.show()