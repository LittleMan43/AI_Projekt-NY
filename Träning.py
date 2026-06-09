import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
from Data_processing import load_data

data, X, y = load_data()

# 3. Dela data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #X_train och Y_train innebär den data den får träna med dvs det kan vara 1 rad x och 1 rad y som den ska analysera mönster och använda dem för testet

# 4. Träna modell

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train) #fit matar informationen till modellen och låter den helt enkelt analysera datan av X_train och Y_train för att koppla ihop mönster och gissa utifrån dem
y_predict = model.predict(X_test) #Model.predict antar jag innebär att modellen gissar utifrån X_test då kan man säga att y_predict är = dess svar
cm = confusion_matrix(y_test, y_predict)

joblib.dump(model, "heart_model.pkl") #sparar modelen 