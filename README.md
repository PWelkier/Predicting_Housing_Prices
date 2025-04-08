# 🏘️ California Housing Price Prediction – Linear Regression Project

## 📌 Cel projektu

Celem projektu była predykcja cen mieszkań w Kalifornii przy użyciu **regresji liniowej**. Analiza została przeprowadzona na podstawie danych zawierających cechy opisujące dystrykty takie jak:

- długość geograficzna,
- szerokość geograficzna,
- średni wiek budynków,
- liczba pokoi,
- liczba sypialni,
- populacja,
- liczba gospodarstw domowych,
- średni dochód,
- odległość od oceanu.

Każda z tych cech wpływa – w mniejszym lub większym stopniu – na wartość nieruchomości w Kalifornii, co potwierdziły przeprowadzone analizy.

---

## 📊 Ocena modelu

Model został oceniony na podstawie:

- **Wartości współczynnika R²**: `0.65` – czyli 65% zmienności cen mieszkań można wyjaśnić na podstawie zmiennych niezależnych.
- **Najbardziej wpływowa zmienna**: `x14` – liczba pokoi przypadających na dom.
- **Parametr p-value**: bardzo małe wartości – co sugeruje istotność zmiennych w modelu.
- **Wartości t-studenta**: od `0.030` (dla liczby sypialni – cecha interakcyjna) do `233.544`.

---

## ⚙️ Optymalizacja modelu

W celu poprawy skuteczności modelu, zastosowano **GridSearchCV**. Wyniki:

### 🔍 Przed optymalizacją:
- RMSE: `68434.686`
- MAPE: `0.284`

### ✅ Po GridSearchCV:
- RMSE: `66954.24`
- MAPE: `0.291`

🧠 **Wniosek**: Choć RMSE spadł o `1481.44`, MAPE nieznacznie wzrosło – może to wynikać z różnic w rozkładzie cen między zbiorem treningowym a testowym.

---

## 🌲 Random Forest Regressor – elastyczność vs. generalizacja

RandomForestRegressor to bardziej elastyczny model, który lepiej dopasowuje dane treningowe, ale może mieć problemy z generalizacją:

### 📉 Błędy na zbiorze treningowym:
- RMSE: `18649.484`
- MAPE: `0.067`

### 📈 Błędy na zbiorze testowym:
- RMSE: `79210.63`
- MAPE: `0.346`

🚀 Wniosek: regresja liniowa, mimo wyższych błędów na danych treningowych, **lepiej generalizuje** i może być trafniejszym wyborem w kontekście tego projektu.

---

## 📝 Podsumowanie

Projekt badawczy skupiał się na wykorzystaniu regresji liniowej do oszacowania cen mieszkań w Kalifornii. Przeprowadzona analiza wykazała, że:
- Ceny nieruchomości są silnie powiązane z liczbą pokoi na dom, średnim dochodem i lokalizacją.
- Regresja liniowa zapewnia dobrą generalizację,
- Optymalizacja modelu przy pomocy GridSearchCV poprawiła RMSE.

Dalsze ulepszanie modelu może obejmować:
- Użycie nieliniowych algorytmów (np. Gradient Boosting),
- Lepszą inżynierię cech,
- Zastosowanie transformacji zmiennych.

---

## 📌 Uwagi

- Zbiór danych użyty w projekcie pochodzi z Kaggle:  
  👉 [House Pricing - Machine Learning](https://www.kaggle.com/code/franvaluch/house-pricing-machine-learning-properati-proyect-1/input)
