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
#      DIAGNÓSTICO DE HETEROCEDASTICIDADE EM MODELOS DE REGRESSÃO           #
#              EXEMPLO 06 - CARREGAMENTO DA BASE DE DADOS                   #
#############################################################################
    
df_saeb_rend = pd.read_csv("saeb_rend.csv", delimiter=',')
df_saeb_rend

#Características das variáveis do dataset
df_saeb_rend.info()

#Estatísticas univariadas
df_saeb_rend.describe()


# In[ ]: Tabela de frequências absolutas das variáveis 'uf' e rede'

df_saeb_rend['uf'].value_counts()
df_saeb_rend['rede'].value_counts()


# In[ ]: Plotando 'saeb' em função de 'rendimento', com linear fit

plt.figure(figsize=(15,10))
sns.regplot(x='rendimento', y='saeb', data=df_saeb_rend, marker='o',
            fit_reg=True, color='green', ci=False,
            scatter_kws={"color":'gold', 'alpha':0.5, 's':150})
plt.title('Gráfico de Dispersão com Ajuste Linear', fontsize=20)
plt.xlabel('rendimento', fontsize=17)
plt.ylabel('saeb', fontsize=17)
plt.show()


# In[ ]: Plotando 'saeb' em função de 'rendimento', com destaque para a
#'rede' escolar

plt.figure(figsize=(15,10))
sns.scatterplot(x='rendimento', y='saeb', data=df_saeb_rend,
                hue='rede', alpha=0.5, s=120, palette = 'viridis')
plt.title('Gráfico de Dispersão com Ajuste Linear', fontsize=20)
plt.xlabel('rendimento', fontsize=17)
plt.ylabel('saeb', fontsize=17)
plt.legend(loc='upper left', fontsize=17)
plt.show()


# In[ ]: Plotando 'saeb' em função de 'rendimento', com destaque para a
#'rede' escolar e linear fits - Gráfico pela função 'lmplot' do 'seaborn' com
#estratificação de 'rede' pelo argumento 'hue'

plt.figure(figsize=(15,10))
sns.lmplot(x='rendimento', y='saeb', data=df_saeb_rend,
           hue='rede', ci=None, palette='viridis')
plt.title('Gráfico de Dispersão com Ajuste Linear por Rede', fontsize=14)
plt.xlabel('rendimento', fontsize=12)
plt.ylabel('saeb', fontsize=12)
plt.show()


# In[ ]: Plotando 'saeb' em função de 'rendimento', com destaque para a
#'rede' escolar e linear fits - Gráfico pela função 'regplot' do 'seaborn'

plt.figure(figsize=(15,10))
df1 = df_saeb_rend[df_saeb_rend['rede'] == 'Municipal']
df2 = df_saeb_rend[df_saeb_rend['rede'] == 'Estadual']
df3 = df_saeb_rend[df_saeb_rend['rede'] == 'Federal']
sns.regplot(x='rendimento', y='saeb', data=df1, ci=False, marker='o',
            scatter_kws={"color":'darkorange', 'alpha':0.3, 's':150},
            label='Municipal')
sns.regplot(x='rendimento', y='saeb', data=df2, ci=False, marker='o',
            scatter_kws={"color":'darkviolet', 'alpha':0.3, 's':150},
            label='Estadual')
sns.regplot(x='rendimento', y='saeb', data=df3, ci=False, marker='o',
            scatter_kws={"color":'darkgreen', 'alpha':0.8, 's':150},
            label='Federal')
plt.title('Gráfico de Dispersão com Ajuste Linear por Rede', fontsize=20)
plt.xlabel('rendimento', fontsize=17)
plt.ylabel('saeb', fontsize=17)
plt.legend(fontsize=17)
plt.show()


# In[ ]: Estimação do modelo de regressão e diagnóstico de heterocedasticidade

# Estimando o modelo
modelo_saeb = sm.OLS.from_formula('saeb ~ rendimento', df_saeb_rend).fit()

# Parâmetros do modelo
modelo_saeb.summary()


# In[ ]: Adicionando fitted values e resíduos do 'modelo_saeb'
#no dataset 'df_saeb_rend'

df_saeb_rend['fitted'] = modelo_saeb.fittedvalues
df_saeb_rend['residuos'] = modelo_saeb.resid
df_saeb_rend


# In[ ]: Gráfico que relaciona resíduos e fitted values do
#'modelo_saeb'

plt.figure(figsize=(15,10))
sns.regplot(x='fitted', y='residuos', data=df_saeb_rend,
            marker='o', fit_reg=False,
            scatter_kws={"color":'red', 'alpha':0.2, 's':150})
plt.title('Gráfico de Dispersão entre Resíduos e Fitted Values', fontsize=20)
plt.xlabel('Fitted Values do Modelo', fontsize=17)
plt.ylabel('Resíduos do Modelo', fontsize=17)
plt.legend(fontsize=17)
plt.show()


# In[ ]: Plotando os resíduos do 'modelo_saeb' e acrescentando
#uma curva normal teórica para comparação entre as distribuições
#Kernel density estimation (KDE) - forma não-paramétrica para estimar
#a função densidade de probabilidade de uma variável aleatória

from scipy.stats import norm

plt.figure(figsize=(15,10))
sns.distplot(modelo_saeb.resid, fit=norm, kde=True, bins=15,
             color='red')
sns.kdeplot(data=modelo_saeb.resid, multiple="stack", alpha=0.4,
            color='red')
plt.xlabel('Resíduos do Modelo', fontsize=16)
plt.ylabel('Densidade', fontsize=16)
plt.show()


# In[ ]: Função para o teste de Breusch-Pagan para a elaboração
# de diagnóstico de heterocedasticidade

# Criação da função 'breusch_pagan_test'

from scipy import stats

def breusch_pagan_test(modelo):

    df = pd.DataFrame({'yhat':modelo.fittedvalues,
                       'resid':modelo.resid})
   
    df['up'] = (np.square(df.resid))/np.sum(((np.square(df.resid))/df.shape[0]))
   
    modelo_aux = sm.OLS.from_formula('up ~ yhat', df).fit()
   
    anova_table = sm.stats.anova_lm(modelo_aux, typ=2)
   
    anova_table['sum_sq'] = anova_table['sum_sq']/2
    
    chisq = anova_table['sum_sq'].iloc[0]
   
    p_value = stats.chi2.pdf(chisq, 1)*2
    
    print(f"chisq: {chisq}")
    
    print(f"p-value: {p_value}")
    
    return chisq, p_value


# In[ ]: Teste de Breusch-Pagan propriamente dito

breusch_pagan_test(modelo_saeb)
#Presença de heterocedasticidade -> omissão de variável(is) explicativa(s)
#relevante(s)

#H0 do teste: ausência de heterocedasticidade.
#H1 do teste: heterocedasticidade, ou seja, correlação entre resíduos e
#uma ou mais variáveis explicativas, o que indica omissão de
#variável relevante!

# Interpretação
teste_bp = breusch_pagan_test(modelo_saeb) #criação do objeto 'teste_bp'
chisq, p = teste_bp #definição dos elementos contidos no objeto 'teste_bp'
alpha = 0.05 #nível de significância
if p > alpha:
    print('Não se rejeita H0 - Ausência de Heterocedasticidade')
else:
	print('Rejeita-se H0 - Existência de Heterocedasticidade')


# In[ ]: Dummizando a variável 'uf'

df_saeb_rend_dummies = pd.get_dummies(df_saeb_rend, columns=['uf'],
                                      drop_first=True)

df_saeb_rend_dummies


# In[ ]: Estimação do modelo de regressão múltipla com n-1 dummies

# Definição da fórmula utilizada no modelo
lista_colunas = list(df_saeb_rend_dummies.drop(columns=['municipio',
                                                        'codigo',
                                                        'escola',
                                                        'rede',
                                                        'saeb',
                                                        'fitted',
                                                        'residuos']).columns)
formula_dummies_modelo = ' + '.join(lista_colunas)
formula_dummies_modelo = "saeb ~ " + formula_dummies_modelo

modelo_saeb_dummies_uf = sm.OLS.from_formula(formula_dummies_modelo,
                                               df_saeb_rend_dummies).fit()

#Parâmetros do modelo
modelo_saeb_dummies_uf.summary()

#Estimação do modelo por meio do procedimento Stepwise
from statstests.process import stepwise
modelo_saeb_dummies_uf_step = stepwise(modelo_saeb_dummies_uf, pvalue_limit=0.05)


# In[ ]: Teste de Breusch-Pagan para diagnóstico de heterocedasticidade
#no 'modelo_saeb_dummies_uf_step'

breusch_pagan_test(modelo_saeb_dummies_uf_step)

# Interpretação
teste_bp = breusch_pagan_test(modelo_saeb_dummies_uf_step) #criação do objeto 'teste_bp'
chisq, p = teste_bp #definição dos elementos contidos no objeto 'teste_bp'
alpha = 0.05 #nível de significância
if p > alpha:
    print('Não se rejeita H0 - Ausência de Heterocedasticidade')
else:
	print('Rejeita-se H0 - Existência de Heterocedasticidade')


# In[ ]: Adicionando fitted values e resíduos do 'modelo_saeb_dummies_uf_step'
#no dataset 'df_saeb_rend'

df_saeb_rend['fitted_step'] = modelo_saeb_dummies_uf_step.fittedvalues
df_saeb_rend['residuos_step'] = modelo_saeb_dummies_uf_step.resid
df_saeb_rend


# In[ ]: Gráfico que relaciona resíduos e fitted values do
#'modelo_saeb_dummies_uf_step'

plt.figure(figsize=(15,10))
sns.regplot(x='fitted_step', y='residuos_step', data=df_saeb_rend,
            marker='o', fit_reg=False,
            scatter_kws={"color":'dodgerblue', 'alpha':0.2, 's':150})
plt.title('Gráfico de Dispersão entre Resíduos e Fitted Values', fontsize=20)
plt.xlabel('Fitted Values do Modelo Stepwise com Dummies', fontsize=17)
plt.ylabel('Resíduos do Modelo Stepwise com Dummies', fontsize=17)
plt.legend(fontsize=17)
plt.show()


# In[ ]: Plotando os resíduos do 'modelo_saeb_dummies_uf_step' e acrescentando
#uma curva normal teórica para comparação entre as distribuições
#Kernel density estimation (KDE) - forma não-paramétrica para estimar
#a função densidade de probabilidade de uma variável aleatória

from scipy.stats import norm

plt.figure(figsize=(15,10))
sns.distplot(modelo_saeb_dummies_uf_step.resid, fit=norm, kde=True, bins=15,
             color='dodgerblue')
sns.kdeplot(data=modelo_saeb_dummies_uf_step.resid, multiple="stack", alpha=0.4,
            color='dodgerblue')
plt.xlabel('Resíduos do Modelo', fontsize=16)
plt.ylabel('Densidade', fontsize=16)
plt.show()


# In[ ]: Plotando 'saeb' em função de 'rendimento', com destaque para a
#'uf' e linear fits - Gráfico pela função 'lmplot' do pacote 'seaborn', com
#estratificação de 'uf' pelo argumento 'hue'

plt.figure(figsize=(15,10))
sns.lmplot(x='rendimento', y='saeb', data=df_saeb_rend,
           hue='uf', ci=None, palette='viridis')
plt.title('Gráfico de Dispersão com Ajuste Linear por UF', fontsize=14)
plt.xlabel('rendimento', fontsize=12)
plt.ylabel('saeb', fontsize=12)
plt.show()