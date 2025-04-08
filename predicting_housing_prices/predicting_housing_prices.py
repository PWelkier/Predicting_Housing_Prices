#!/usr/bin/env python
# coding: utf-8

# # Prognozowanie cen nieruchomości w Kalifornii 

# ## Autor:
# ##### Patryk Welkier 217409

# In[6]:


# Importujemy biblioteki potrzebne do analizy danych
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor


# # Streszczenie:

# W noim projekcie metodą regresji liniowej postarałem się oszacować jaki jest koszt zakupu mieszkania w Kaliforni. Do przeprowadzenia tej analizy posłużyłem się takimi parametrami (dotyczących dystryktów) jak długość geograficzna, szerokośś geograficzna, średni wiek budynku, ilość pokoi, ilość sypialni, populacja, ilość gospodarstw domowych, średnie przychody, odleglość od oceanu. Każda z tych cech w mniejszy lub wiekszym stopniu wpływa na cenę mieszkań w tym pięknym miejscu, co udało nam się pokazać w naszej pracy.
# 

# # Słowa kluczowe
# * Regresja liniowa
# * p-value
# * Współczynik determinancji(R^2)
# * Średni błąd kwadratowy
# * Średni błąd procentowy absolutny

# # Założenia regresji liniowej:

# * liniowa zależność - zależność liniowej relacji między zmienną wyjaśniającą, a zmienną wyjaśniną
# * homoskedastyczność - występowania stałej wariancji reszt dla poszczególnych wartości zmiennej niezależnej, co sprowadza sie do tego, że dla poszczególnych wartości / przedziałów wartości przewidywujących rozproszenie błędów jest podobne 
# * brak wspóliniowości predyktów - brak silnej korelacji między zmiennymi objaśniającymi (ponieważ w przeciwnym przypadku najczęściej jeden z nich jest okazuję się nie istotny)
# * liczba obserwacji musi być większa bądź równa liczbie parametrów wyprowadzonych z analizy regresji - warunek ten jest niezbędny do wyliczenia współczynników regresji, im wiecej mamy obserwacji tym z większą precyzja możemy oszacować parametr 
# * brak wystąpienia autokorelacji reszt, składnika losowego - błędy przewidywania rzeczywistej wartości zmiennej zależnej na podstawie utworzonego przez nas modelu regresji są niezależne od siebie
# * reszty mają rozkład zbliżony do rozkładu normalnego - składnik losowy ma rozkład zbliżony do rozkładu normalnego N(0,σ)
# * analiza regresji nie powinna być ekstrapolowana - wyliczony model regresji, który został opracowany dla danego zakresu danych nie powinien być ekstrapolowany na dane spoza zakresu, na którym został zbudowany
# 

# # Regresja liniowa i interpretowalność wyników
# Regresja liniowa generuje funkcje liniowe, dlatego nie jest tak "elastyczna" jak inne modele statystyczne, jednak zapewnia kompromis pomiędzy interpretowalnością wyników, a precyzją modelu. 
# Wykres poniżej, zaczęrpnięty z książki "An Introduction to Statistical Learning", przedstawia zależność elastyczności różnych modeli statystycznych i ich interpretowalności.
# 
# ![Zrzut ekranu 2024-04-10 171241.png](attachment:4f5eabcc-094c-4e99-835e-7a3901066b50.png)

# # Wzory matematyczne

# ![Zrzut ekranu 2024-04-10 163112.png](attachment:aa7b150c-07ce-40f7-a5b6-c8f7d38c5ae9.png)
# ![Zrzut ekranu 2024-04-10 163302.png](attachment:55dcd91b-6147-437d-b41a-37502ba2ab3e.png)
# ![Zrzut ekranu 2024-04-10 163421.png](attachment:92927c67-10e5-4aaf-a805-ffa31249e8c3.png)
# 
# 
# 

# ![image.png](attachment:e4afae30-bba9-4bc8-ad96-fedbc9f28dbf.png)

# ![image.png](attachment:488c5907-d2f2-4842-908d-5b1a66c61a68.png)
# ![image.png](attachment:a4ff093c-352b-4d46-aefc-255417a2c3a1.png)

# ![image.png](attachment:4c0f7bcf-48c5-425f-a58e-66165094cdc4.png)

# # Zmienne:

# * longitude - długość geograficzna dystryktów
# * latitude - szerokość geograficzna dystryktów
# * housing_median_age - średni wiek budynków mieszkalnych w danym obszarze
# * total_rooms - ilość pokoi w danym obszarze
# * total_bedrooms - ilość sypialni mieszczących się w danym obszarze	
# * population - populacja w danym dystrykcie
# * households - ilość gospodarstw domowych w danym dystrykcie
# * median_income - średnie przychody w danym obszarze	
# * median_house_value -średnia wartość domów w danym obszarze
# * ocean_proximity - określenie czy dany obszar znajduję sie daleko od oceanu
# * room_household_ratio - stosunek liczby pokoi do liczby gospodarstw domowych 
# * population_household_ratio - stosunek liczby ludności do liczby gospodarstw domowych (wskaźnik demograficzny)

# In[2]:


# Wczytujemy dane z pliku csv
data = pd.read_csv('housing.csv')
data.head(10)


# # Analiza danych 
#  Dane zostały łącznie pobrane z 20640 dysktryktów w Kalifornii, przy czym brakuje danych o ilości sypialni w 207 dystryktach, które zostały uzupełnione dominantą. Do transformacji danych użyta została standaryzacja. 
# 
#  Odchylenie standardowe jest duże w przypadku median wieku gospodarstw domowych, liczby pokoi, liczby sypialni, liczby gospodarstw domowych, median zarobków i median wartości domów w dystrykcie. Oznacza to, że te dane będą znacząco rozproszone. W przypadku współrzędnych geograficznych odchylenie standardowe jest bardzo niskie, ze względu na to, że wszystkie dystrykty znajdują się w tym samym stanie, w stanie Kalifornia.  
# 
#  W przypadku liczby pokoi, liczby sypialni, populacji, liczby gospodarstw odmowych, median zarobków i median wartości domu w dystryktach mediany tych wartości są niższe od ich wartości średnich, co oznacza, że dużo danych jest znacząco odstających od ich wartości średniej. W przypadku współrzędnych geograficznych i median wieku gospodarstwa domowego różnica jest niewielka i dane nie powinny znacząco odstawać od średniej. 
# 
#  W głębi lądu znajduję się 6551(31,74%) dystryktów. Najwięcej, bo w odległości mniejszej niż 1 godzina od oceanu, znajduję się 9136(44,26%) dystryktów. W pobliżu oceanu znajduję się 2658(12,88%) dystryktów, a w pobliżu zatoki 2290(11,1%) dystryktów. Najmniej, czyli tylko 5(0,02%) dystryktów znajduję się na wyspie. 

# In[3]:


# Sprawdzamy wymiar danych
data.shape


# In[4]:


# Wyświetlamy podsumowanie informacji o danych
data.info()


# In[5]:


# Liczymy wystapienia unikalnych wartości w kolumnie "ocean_proximity"
data["ocean_proximity"].value_counts()


# In[6]:


# Obliczamy i wyświetlamy procentowy udział brakujących wartości w kaśdej kolumnie
for col in data.columns :
    missingSum = data[col].isnull().sum()
    pr = (missingSum/data.shape[0]) *100
    print("{} : {} ({}%)".format(col, missingSum, round(pr, 2)))


# In[7]:


# Przetwarzamy kolumne "ocean_proximity" na kodowanie one-hot
enc = OneHotEncoder(handle_unknown="ignore")
data_1hot = enc.fit_transform(data[["ocean_proximity"]])
data_1hot.toarray()


# In[8]:


# Tworzymy histogramy dla wszystkich kolumn
data.hist(figsize = (20,15))


# # Korelacje
# Liczba gospodarstw domowych, pokoi, sypialni i populacji jest ze sobą ściśle powiązana. Im większa liczba gospodarstw domowych, tym większa będzie liczba pokoi, sypialni i tym samym większa populacja w dystrykcie. Większa mediana wieku gospodarstw w dystrykcie oznacza mniejszą liczbę pokoi, sypialni i populację w tym dystrykcie.

# In[9]:


# Generujemy podsumowanie statystyczne dla danych
data.describe()


# In[10]:


# Tworzymy wykres heatmapy korelacji miedzy kolumnami numerycznymi
sns.heatmap(data.corr(numeric_only= True))


# # Tworzenie zbioru testowego
# Ważnym elementem tworzenia modelu statystycznego jest możliwość jego późniejszej ewaluacji. Dlatego niezbędne jest utworzenie zbioru testowego, który będzie miał podobny rozkład danych co zbiór treningowy. Dzięki temu, będziemy mogli dokładnie ocenić wydajność naszego modelu, nie pomijając części przypadków. 

# In[11]:


# Dzielimy dane na zbiór treningowy (20%) i testowy (80%)
data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)


# In[12]:


# Tworzymy kolumne "income_c" w zbiorze treningowym i testowym, ktora zawiera zaklasyfikowane wartosci z kolumny "median_income"
# na podstawie przedzialow: [0, 1.5), [1.5, 3.0), [3.0, 4.5), [4.5, 6.0), [6.0, inf) i wyświetlamy histogram kolumny "income_c"
data["income_c"] = pd.cut(data["median_income"],
 bins=[0, 1.5, 3.0, 4.5, 6, np.inf],
 labels=[1, 2, 3, 4, 5])
data_test["income_c"] = pd.cut(data_test["median_income"],
 bins=[0, 1.5, 3.0, 4.5, 6, np.inf],
 labels=[1, 2, 3, 4, 5])
data["income_c"].hist()


# In[13]:


# Dzielimy dane na zbiory treningowy i testowy z zachowaniem proporcji dla zmiennej "income_c",
# aby zarowno zbior treningowy jak i testowy mialy podobny rozklad danych
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(data, data["income_c"]):
    train_sss_set = data.iloc[train_index]
    test_sss_set = data.iloc[test_index]
test_sss_set["income_c"].value_counts()/test_sss_set.shape[0]


# In[14]:


# Obliczamy udziały poszczególnych kategorii zmiennej "income_c" w zbiorze treningowym,
# normalizujac liczbe wystapień przez liczbe wszystkich danych
data["income_c"].value_counts()/data.shape[0]


# In[15]:


# Obliczamy udziały poszczególnych kategorii zmiennej "income_c" w zbiorze testowym,
# normalizujac liczbe wystapień przez liczbe wszystkich danych testowych.
data_test["income_c"].value_counts()/data_test.shape[0]


# In[16]:


# Usuwamy kolumne "income_c"
train_sss_set = train_sss_set.copy()
test_sss_set = test_sss_set.copy()
train_sss_set.drop("income_c", axis = 1,inplace=True)
test_sss_set.drop("income_c", axis = 1, inplace=True)
data.drop("income_c", axis=1, inplace=True)
train_sss_set


# In[17]:


# Tworzymy nowy zbiór danych, który zawiera jednynie dane numeryczne.
# Nie zawiera kolumn "ocean_proximity" ani "median_house_value"
data_num = data.drop("ocean_proximity", axis=1)
data_num.drop("median_house_value", axis = 1, inplace = True)
data_num


# # Efekt interakcji
# W trakcie przetwarzania danych, które będą wykorzystywane w regresji liniowej, ważne jest, aby wziąć pod uwagę efekt interakcji (określany w marketingu  jako efekt synergii).
# > The standard linear regression model provides interpretable results
# and works quite well on many real-world problems. However, it makes several highly restrictive assumptions that are often violated in practice. Two
# of the most important assumptions state that the relationship between the
# predictors and response are additive and linear. The additivity assumption means that the association between a predictor Xj and the response Y does
# not depend on the values of the other predictors.

# In[18]:


# Stworzenie transformatora danych, który dodaje kolumny takie jak "room_household_ratio", "population_household_ratio", 
# czy "bedroom_room_ratio" w zależnosci od podanych parametrów true/false

col_names = "total_rooms", "total_bedrooms", "population", "households"
rooms_ix, bedrooms_ix, population_ix, households_ix = [
    data.columns.get_loc(c) for c in col_names]


class XAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedroom_room_ratio=True, add_population_household_ratio=True, add_room_household_ratio=True):
        self.add_bedroom_room_ratio = add_bedroom_room_ratio
        self.add_room_household_ratio = add_room_household_ratio
        self.add_population_household_ratio = add_population_household_ratio
    
    def fit(self, X, y=None):
        return self 
    
    def transform(self, X, y=None):
        if self.add_room_household_ratio:
            add_room_household_ratio = X[:, rooms_ix] / X[:, households_ix]
            X = np.concatenate((X, add_room_household_ratio.reshape(-1, 1)), axis=1)
        if self.add_population_household_ratio:
            add_population_household_ratio = X[:, population_ix] / X[:, households_ix]
            X = np.concatenate((X, add_population_household_ratio.reshape(-1, 1)), axis=1)
        if self.add_bedroom_room_ratio:
            add_bedroom_room_ratio = X[:, bedrooms_ix] / X[:, rooms_ix]
            X = np.concatenate((X, add_bedroom_room_ratio.reshape(-1, 1)), axis=1)
        return X


# In[19]:


# Dodajemy te cechy jako kolumny do zbioru danych
adder = XAdder(True, True, False)
data_added = adder.transform(data.values)
data_added = pd.DataFrame(
    data_added,
    columns=list(data.columns)+["room_household_ratio", "population_household_ratio"],
    index=data.index)
data_added


# In[20]:


# Tworzymy pipeline dla danych numerycznych. Najpierw uzupełniamy brakujace
# wartosci najczesciej wystepującymi wartościami. Nastepnie dodajemy nowe 
# cechy numeryczne a na koniec wszystko standaryzujemy
pipeline_num = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("x_adder", XAdder()),
    ("std_scaler", StandardScaler())
])


# In[21]:


data


# In[22]:


# OneHotEncoding kolumny "ocean_proximity"
data_1hot_df = pd.DataFrame.sparse.from_spmatrix(data_1hot)
data_col = data.join(data_1hot_df)
data_col.drop("ocean_proximity", inplace=True, axis = 1)
data_col = pd.get_dummies(data.astype(str),dtype=int,columns=['ocean_proximity'], prefix='', prefix_sep='')
data_col


# In[23]:


# Przygotowanie zbioru treningowego
train_sss_x = train_sss_set.drop("median_house_value", axis = 1)
train_sss_y = train_sss_set["median_house_value"]
train_x = train_sss_x.copy()
train_sss_x


# In[24]:


# Stworzenie Pipeline-u do transformacji danych, z podziałem na dane numeryczne i kategoryczne
x_num = list(data_num)
x_c = ["ocean_proximity"]

self_pipeline = ColumnTransformer([
 ("num", pipeline_num, x_num),
 ("c", OneHotEncoder(handle_unknown='ignore'), x_c)
 ])
train_sss_pipe = self_pipeline.fit_transform(train_sss_x)
train_prepared = self_pipeline.fit_transform(train_x)


# In[25]:


# Transformacja danych za pomocą pipeline-u
pipe_test_x = data.drop("median_house_value", axis = 1).iloc[:10]
pipe_test_lab = data["median_house_value"].iloc[:10]
pipe_test_x


# In[26]:


pipe_test_lab


# In[27]:


pipe_test_done = self_pipeline.transform(pipe_test_x)
pipe_test_done


# In[28]:


# Sprawdzenie regresji liniowej
regressor = LinearRegression()
regressor.fit(train_sss_pipe, train_sss_y)
prediction_test_10 = regressor.predict(pipe_test_done)
prediction_test_10


# In[29]:


pipe_test_lab


# In[30]:


# Obliczenie RMSE i MAPE na zbiorze treningowym
predictions = regressor.predict(train_sss_pipe)
mse = mean_squared_error(train_sss_y, predictions, squared=False)
print(f"RMSE: {round(mse,3)}")
mape = mean_absolute_percentage_error(y_true=train_sss_y, y_pred=predictions)
print(f"MAPE: {round(mape,3)}")


# # Ocena modelu
# Obliczone poniżej parametry pozwalają na oszacowanie wydajności modelu. Wynika z nich, że największy wpływ na wartość domu ma cecha niezależna x14, czyli liczba pokoi przypadająca na jeden dom. Parametr p-value jest bardzo mały, co sugeruje, że model jest znaczący. Parametr R^2, który przedstawia proporcję wariancji zmiennej zależnej, która jest wyjaśniana przez model regresji w stosunku do całkowitej zmienności tej zmiennej, wynosi 65%. Wartość t-studenta dla cech niezależnych przyjmuje wartości z zakresu 0,030 do 233,544. Minimalną wartość przyjmuje dla parametru x5, czyli liczby sypialni, wynoszącą 0,030. Ponieważ wykorzystujemy tę cechę niezależną jako "interaction term", nie usuwamy jej z modelu.  Poniższy cytat z książki "An Introduction to Statistical Learning" uzasadnia to rozumowanie. Wszystkie pozostałe parametry uznajemy za znaczące.
# 
# >However, it is sometimes the case that an interaction term has a very small p-value, but
# the associated main efects do not. The hierarchical principle states that if we include an interaction in a model, we should also include the main efects, even if the p-values associated with their coefcients are not signifcant. In other words, if the interaction between X1 and X2 seems important, then we should include both X1 and X2 in the model even if their coefcient estimates have large p-values. The rationale for this principle is that if X1 × X2 is related to the response,
# then whether or not the coefcients of X1 or X2 are exactly zero is of little interest. Also X1 × X2 is typically correlated with X1 and X2, and so
# leaving them out tends to alter the meaning of the interaction.

# In[31]:


# Obliczenie parametrów regresji liniowej
result = sm.OLS(train_sss_y, train_sss_pipe).fit()

result.summary()


# In[32]:


# Użycie GridSearchCV do optymalizacji hyper-parametrów
param_grid = {
    'columntransformer__add_bedroom_room_ratio': [True, False],
    'columntransformer__add_population_household_ratio': [True, False],
    'columntransformer__add_room_household_ratio': [True, False]
}
grid_search = GridSearchCV(
    estimator=Pipeline([
        ("columntransformer",XAdder() ),
        ("linear_regression", LinearRegression())
    ]),
    param_grid=param_grid,
    cv=5,  
    scoring='neg_mean_squared_error', 
    verbose=2,  
    n_jobs=-1  
)


# In[33]:


grid_search.fit(train_prepared, train_sss_y)


# In[34]:


# Uzyskanie najlepszego estymatora, z ustawionymi hyper-parametrami
grid_search.best_estimator_


# In[35]:


grid_search.best_params_


# In[36]:


test_sss_x = test_sss_set.drop("median_house_value", axis = 1)
prepared_test = self_pipeline.fit_transform(test_sss_x)
test_sss_y = test_sss_set["median_house_value"]
model = grid_search.best_estimator_
predictions_grid_test = model.predict(prepared_test)
predictions_grid_test


# In[45]:


# Obliczenie błędów RMSE i MAPE na zbiorze testowym
mse_grid_test = mean_squared_error(test_sss_y, predictions_grid_test, squared=False)
print(f"RMSE: {round(mse_grid_test,3)}")
mape_test = mean_absolute_percentage_error(y_true=test_sss_y, y_pred=predictions_grid_test)
print(f"MAPE: {round(mape_test,3)}")


# In[46]:


# Przetestowanie modelu RandomForestRegressor i obliczenie błędów na zbiorze testowym w celu porównania do regresji liniowej
regressor = RandomForestRegressor(random_state = 42)
regressor.fit(train_prepared, train_sss_y)
RGR_test = regressor.predict(prepared_test)
mse = mean_squared_error(test_sss_y, RGR_test, squared=False)
print(f"RMSE: {round(mse,3)}")
mape = mean_absolute_percentage_error(y_true=test_sss_y, y_pred=RGR_test)
print(f"MAPE: {round(mape,3)}")


# In[47]:


# Obliczenie błędów na zbiorze treningowym w celu porównania do regresji liniowej
predicted_train = regressor.predict(train_prepared)
mse = mean_squared_error(train_sss_y, predicted_train, squared=False)
print(f"RMSE: {round(mse,3)}")
mape = mean_absolute_percentage_error(y_true=train_sss_y, y_pred=predicted_train)
print(f"MAPE: {round(mape,3)}")


# # Cytowania
#  Wzory do obliczania Root Mean Squared Error (RSE), Residual Sum of Squares (RSS), współczynnika determinacji (R^2), wykres porównujący modele oraz metody do poprawienia wydajności modelu statystycznego zostały zaczerpnięte z książki "An Introduction to Statistical Learning" autorstwa Gareth James, Daniela Witten, Trevor Hastie i Rob Tibshirani. Książka ta jest uznawanym źródłem w dziedzinie nauki statystycznej i uczenia maszynowego, dostarczając zarówno teoretycznych podstaw, jak i praktycznych przykładów zastosowań. 
#  
# "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems" autorstwa Aurelien'a Geron'a dostarczyła niezbędnych informacji o przetwarzaniu danych przed użyciem ich w algorytmach oraz wiedzy jak ocenić wydajność stworzonego modelu.

# # Bibliografia
# <ol>
#     <li>
#         Bhagat, N., Mohokar, A., & Mane, S. (2016). House Price Forecasting using Data Mining. International Journal of Computer Applications, 152(2),
#         23–26.
#     </li>
#     <li>
#        M. Bhuiyan and M. A. Hasan, ”Waiting to Be Sold: Prediction of TimeDependent House Selling Probability,” 2016 IEEE International Conference on Data Science and Advanced Analytics (DSAA), Montreal, QC,
# 2016, pp. 468-477, doi: 10.1109/DSAA.2016.58.
#     </li>
#     <li>
#         "An Introduction to Statistical Learning" Gareth James, Daniela Witten, Trevor Hastie i Rob Tibshirani 2021
#     </li>
#     <li>
#         "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems" Aurelien  Geron 2020
#     </li>
# </ol>
# 
# 
# ..

# # Podsumowanie
# Projekt badawczy skupiał się na wykorzystaniu regresji liniowej do oszacowania cen mieszkań w Kalifornii, biorąc pod uwagę różnorodne parametry dystryktów. Analiza opierała się na modelu regresji liniowej, w którym badaliśmy zależność między takimi czynnikami jak długość i szerokość geograficzna, wiek budynków, liczba pokoi i sypialni, populacja, dochody oraz odległość od oceanu, a ceną nieruchomości. Wyniki analizy wykazały, że każda z tych cech miała istotny wpływ na cenę mieszkań w Kalifornii.
# 
# Dodatkowo, po przeprowadzeniu analizy regresji liniowej, zastosowałem technikę GridSearchCV w celu optymalizacji modelu. Wyniki tego procesu wskazały na znaczną poprawę modelu.
# 
# Przed zastosowaniem techniki GridSearchCV wartości błędów wynosiły:
# 
# * Średni błąd kwadratowy (RMSE): 68434.686
# * Średni błąd procentowy absolutny (MAPE):0.284
# 
# Po zaimplementowaniu GridSearchCV, otrzymałem następujące wartości błędów:
# 
# * Średni błąd kwadratowy (RMSE): 66954.24
# * Średni błąd procentowy absolutny (MAPE): 0.291
# 
# Wynika z tego, że dzięki optymalizacji udało nam się zmniejszyć wartość MSE o 1481.44, ale jednocześnie zwiększyła się wartość MAPE o 0.007.
# 
# Ponieważ RandomForestRegressor jest modelem bardziej elastycznym, co wynika z wykresu przedstawiajacego zależność elastyczności modelu od interpretowalności wyników, model lepiej radzi sobie na danych treningowych, jednak ma problemy z generalizacją nowych przypadków. Potwierdzają to małe błedy MAPE oraz MSE uzyskane na zbiorze treningowym:
# 
# * Średni błąd kwadratowy (RMSE): 18649.484
# * Średni błąd procentowy absolutny (MAPE): 0.067
#   
#  oraz duże błędy uzyskane na zbiorze testowym:
# 
# * Średni błąd kwadratowy (RMSE): 79210.63
# * Średni błąd procentowy absolutny (MAPE): 0.346
# 
# Uzyskane błedy świadczą, że regresja liniowa, pomimo wiekszęgo błędu na zbiorze treningowym, lepiej generalizuje nowe przypadki i lepiej sprawdzi się w naszym badaniu. Warto zauważyć, że po zaimplementowaniu GridSearchCV, którego zadaniem jest znalezienie najlepszych parametrów do modelu, błąd MSE zmniejszył się o 1481.44, a błąd MAPE zwiększył się o 0.007. Może to wynikać z faktu, że zbiór testowy minimalnie różni się od zbioru treningowego i przechowuje proporcjonalnie mniej domów o dużej wartości, więc błąd procentowy będzie większy, pomimo zmniejszenia MSE.
# 
# 
# 
# Ostateczne błędy mogą wynikać z różnych czynników, takich jak nieliniowe zależności między zmiennymi, czy też braku uwzględnienia istotnych predyktorów. Dlatego też, dalsza analiza i doskonalenie modelu mogą być konieczne w celu uzyskania bardziej precyzyjnych prognoz cen mieszkań w Kalifornii.
