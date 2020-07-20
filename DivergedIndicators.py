
#import relevant packages 
import pandas as pd 
import matplotlib.pyplot as plt
import talib
from talib import abstract
import statsmodels.api as sm

import pandas_datareader as pdr 
sp = pdr.get_data_yahoo('AAPL')
sp = sp[sp['Volume']>0]
data_source = r'C:\Users\user\Desktop\Grad\PY Modules\AAPL.xlsx'
sp.to_excel(data_source)
import pandas as pd 
import numpy as np
sp= pd.read_excel(r'C:\Users\user\Desktop\Grad\PY Modules\AAPL.xlsx')

#label data and target columns
x = df[['One','Two','Three','Four']]
y = df ['Target']


#feature selection using iterative back elimination
cols = list(x.columns)
pmax = 1
while (len(cols)>0):
    p= []
    x_1 = x[cols]
    x_1 = sm.add_constant(x_1)
    model = sm.OLS(y,x_1).fit() #assumes OLS model
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print("The selected columns are: :" , selected_features_BE)

#use of LassoCV for feature selection
from sklearn.linear_model import LassoCV
reg = LassoCV()
reg.fit(x, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(x,y))
coef = pd.Series(reg.coef_, index = x.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other "
      +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")

#technical indicator 1 - SMA 
sma = abstract.SMA
sma = abstract.Function('sma')
sma200 = talib.SMA(df['Target'], 200)
plt.plot(sma200, label =200)
plt.plot(df['Target'], label ='Price')
plt.legend(loc='best')

#technical indicator 2 - RSI 
rsi = abstract.RSI
rsi = abstract.Function('rsi')
real= talib.RSI(df['Target'], 21)
plt.plot(real, label =14)
plt.plot(df['Target'], label ='RSI')
plt.legend(loc='best')
