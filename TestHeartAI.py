import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk

# 1. Läs data
data = pd.read_csv("heart.csv")

# 2. X och y
X = data.drop("target", axis=1) #alla rader tills den när target
y = data["target"] #allt om Target leden

# 2.5 Värden som saknar
NoVal = data.isnull().sum() #kollar vilka led som saknar värden
print(NoVal)
dtype = data.dtypes #Kollar vad det är för typ av data i varje led 
print(dtype)
# 3. Dela data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #X_train och Y_train innebär den data den får träna med dvs det kan vara 1 rad x och 1 rad y som den ska analysera mönster och använda dem för testet

# 4. Träna modell
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train) #fit matar informationen till modellen och låter den helt enkelt analysera datan av X_train och Y_train för att koppla ihop mönster och gissa utifrån dem
y_predict = model.predict(X_test) #Model.predict antar jag innebär att modellen gissar utifrån X_test då kan man säga att y_predict är = dess svar 
#Predict använder sig av modellen. Detta innebär enligt mig att utan fit, gissar modellen slumpmässigt

# 5. Testa modell
print("Guessing Accuracy:", accuracy_score(y_test, y_predict)) # accuracy_score startar processen att jämföra y_test vilket = facit och y_predict som är modellens gissning
cm = confusion_matrix(y_test, y_predict) #Confusion matrix jämför modellens gissningar och med hjälp av facit visar exakta fel och rätt som modellen fick. Tillsammans med accuracy score får vi bättre insikt av vad modellen fick rätt och vad den fick fel.
print(cm)
#plt.scatter(X["age"], y) #Scatter gör prickar i en graf beroende på age och target
plt.xlabel("Sjuk eller inte 0 = frisk, 1 = sjuk") #namn på x led
plt.ylabel("Antal personer") #namn på y led
#plt.plot(X,y)
sns.countplot(x=y)
plt.show()

# 6. Objekt med tkinter

class HeartApp:

    def __init__(self, model):

        self.model = model

        self.window = tk.Tk() #skapar fönstret
        self.window.title("Heart Disease AI predicter") #namn på fönstret
        self.window.geometry("600x600") #storlek på fönstret

        self.entries = {} #Dict som ska innehålla alla entries. Entries = varje ruta i applikationen

        for column in X.columns: #Gör en entry och döper den efter varje rad i x leden på heart csv

            label = tk.Label(self.window, text=column)
            label.pack()

            entry = tk.Entry(self.window)
            entry.pack()

            self.entries[column] = entry

        self.result_label = tk.Label(self.window, text="")
        self.result_label.pack()

        button = tk.Button(self.window, text="Predict", command=self.predict) #Skapar knapp, döper den och command = funktionen den upropar
        button.pack()

    def predict(self):

        person = [] #tom lista av person som ska fyllas av användaren

        for column in X.columns: #loop som lägger till värden till person listan

            if column == "oldpeak": #if sats som kollar efter den enda floaten vilket är oldpeak
                value = float(self.entries[column].get())
            else:
                value = int(self.entries[column].get())

            person.append(value)

        person = [person]

        prediction = self.model.predict(person)[0] #modelen gissar på all data som skrevs

        if prediction == 1:
            self.result_label.config(text="Risk för hjärtsjukdom")
        else:
            self.result_label.config(text="Ingen risk för hjärtsjukdom")

    def run(self):
        self.window.mainloop()


# 7. Terminal
terminal = HeartApp(model) #refererar till objektet
terminal.run() #genomför metod "run" i objektet
