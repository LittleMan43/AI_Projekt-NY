import pandas as pd
import tkinter as tk
import joblib
from Data_processing import load_data

data, X, y = load_data()
loadmodel = joblib.load("heart_model.pkl") #laddar in modelen från notebooken

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
terminal = HeartApp(loadmodel) #refererar till objektet
terminal.run() #genomför metod "run" i objektet