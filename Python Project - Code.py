import os 

os.chdir(r'C:\Users\Madi\Desktop\Lanbide Bootcamp\Python\16- OUTLIERS\EJERCICIOS')

import pandas as pd
import numpy as np

train = pd.read_csv("trainmod.csv", sep=",", encoding="latin-1")

# 1. CLEANING DATA 
    
# all the columns with that have over 30% of missing data, will be removed

perdidos = train.isnull().sum()
perdidos.sum()
train_new = train.loc[:, train.isnull().sum() < 0.3*train.shape[0]]

v_perdidos = train_new.isnull().sum()
v_perdidos
v_perdidos.sum()

    # double checking of the typographical errors

    # XXXXXXX

    # which type of data there is
dtypes=train_new.dtypes

    # 1.1. NUMERICAL DATA cleaning:
    
    # a stadistical summary of the numerical data of the DataFrame
describe=train_new.describe()

    # a new DataFrame only with the numerical data of the original table
trainnum= train_new.select_dtypes(include=['int','float'])
v_perdidosnum = trainnum.isnull().sum()
v_perdidosnum
v_perdidosnum.sum()


for i in train_new:
    print(train_new[i].value_counts())
    
        # it does not seem to have typographical errors that require to be treated
   
    # lost values checking
v_perdidos = train_new.isnull().sum()
v_perdidos
v_perdidos.sum()

    # In order to complete/fill the numerical missing data, the k-nearest neighbors (KNN) algorithm will be used. 
    # This algotherim uses proximity, it groups the data according to their similarity, used afterwards to make predictions 
    # about grouping of an individual data point, which will be used to complete the missing data. 

from fancyimpute import KNN 

trainnum_5 = trainnum
my_imputer_5 =  KNN(k=5) 

    # the k=3 refers to the number of nearest neighbors
    # considered when making a prediction or classication

trainnum_5 = my_imputer_5.fit_transform(trainnum_5)
trainnum_5 = pd.DataFrame(trainnum_5)

    # as when completing the transformation the new dataframe has lost the column names,
    # the column names of the numerical dataframe will be copied to the new one without missing data
trainnum_5.columns = trainnum.columns


    # 1.2. CATEGORICAL DATA cleaning:
        
   # create a new table which will have all the columns from train_new that do not appear in trainnum_5
traincat= train_new.select_dtypes(include=['object'])

    # get the names of the variables/columns
variable_names = traincat.columns.tolist()


    # Calculate the most frequent value of each variable
    # Replace NaN values with the most frequent value

for variable in variable_names:
    most_fr = traincat[variable].mode().values[0]
    traincat[variable].fillna(value=most_fr, inplace=True)
    
    # check that there are no lost values
v_perdidos2 = traincat.isnull().sum()

    # create a new table using the categorical data dataframe and the numerical data dataframe 
join=pd.concat([trainnum_5,traincat],axis=1,join="inner")

    # check that there are no lost values at all in the final DataFrame
v_perdidos3 = join.isnull().sum()


# 2.ANALISING DATA 

import matplotlib.pyplot as plt
import seaborn as sns

    # 2.1. Violin Plot of the year the houses were built & the year it was remodeled  
ax = sns.violinplot(data=trainnum_5[['YearBuilt','YearRemodAdd','GarageYrBlt']])
plt.xlabel('Variables')
plt.ylabel('Year')
plt.title('Years of Building, Remodeled & GarageYrBlt')
ax.set_xlabel('Variables')
ax.set_ylabel('Year')
ax.set_title('Violin Plot House Built, Remodeled & GarageYrBlt Year')
legend_labels = ['House Built Year', 'Remodeled Year', 'Garage Built Year']
ax.legend(legend_labels, loc='lower center')
plt.show()
    
        # It can be seen that the amount of houses that have been built started increading after 
        # 1900 and there have been 3 main periods when there has been an increase of this fact. Also,
        # that the largest amount of houses were built between 1900 & 2010. 

        # It can also be seen that until 1950 there was not data in regard to 
        # year the houses were remodeled. As well as the fact that most of the houses 
        # were remodeled between 1990 & 2010, following the pattern seen previously.
        
        # Finally, regarding the years the garages were built, it shows a similar data comparing to
        # the when the houses were built and the periods.
        
        # To summarise, the highest periods have been almost always similar in all 3 variables. 
        # A further look into violing plot data would be very interesting and useful, as a deeperç
        # analysis could provide information.  
        
des_var = trainnum_5[['YearBuilt','YearRemodAdd', 'GarageYrBlt']].describe()

var_box = trainnum_5[['YearBuilt','YearRemodAdd', 'GarageYrBlt']] 
var_box.plot(kind = 'box')
plt.ylabel('Year')
plt.title('BoxPlot House Built, Remodeled and Garage Year Built Year Boxplot')
ax.set_xlabel('Variables')
ax.set_ylabel('Year')
ax.set_title('BoxPlot House Built, Remodeled and Garage Year Built Year Boxplot')
ax.legend(legend_labels, loc='lower center')
plt.show()

    
    # 2.2. BoxPlot of the the sale price of the houses
    
var1 = trainnum_5.loc[:,'SalePrice']
var1.plot(kind = 'box')
plt.ylabel('Price')
plt.title('Finding Sale Price Outliers')
ax.set_xlabel('Variables')
ax.set_ylabel('Year')
ax.set_title('Finding Sale Price Outliers')
ax.legend(legend_labels, loc='lower center')
plt.show()

sum = var1.describe()

    # The outliers will be taken for further analysis
    
from scipy import stats

    # Create DataFrame of the Sale Price, Outliers of Sale Price. 
SalePrice = trainnum_5['SalePrice']
outliers = SalePrice[np.abs(stats.zscore(SalePrice)) > 3]
    
    # Create a DataFrame, when the index of the outliers dataframe appears in the main final join DataFrame
Join_outliers=pd.merge(join,outliers,left_index=True,right_index=True, how='inner')

    # remove SalePrice_y as it is duplicated
del Join_outliers['SalePrice_y']

    # new DataFrame is created only with data of the SalePrice outliers and their LotArea data. To check whether there is a correlation, a pairplot has been created. 
corr_data=Join_outliers[['SalePrice_x','LotArea']].copy()

g = sns.pairplot(corr_data,plot_kws = {'color': 'green', 'marker': 's'}, diag_kws = {'color': 'orange'})
g.fig.suptitle("Correlation Pairplot: Sale Price & Lot Area", y=1.08)

    # 2.2.1 Boxplot of Outliers 

outliers.plot(kind = 'box')
plt.ylabel('Year')
plt.title('BoxPlot House Built, Remodeled and Garage Year Built Year Boxplot')
ax.set_xlabel('Variables')
ax.set_ylabel('Year')
ax.set_title('BoxPlot House Built, Remodeled and Garage Year Built Year Boxplot')
ax.legend(legend_labels, loc='lower center')
plt.show()
    