import pandas as pd

def load_data(): #funktion som innehåller datan
    data = pd.read_csv("heart.csv") #läser heart.csv
    X = data.drop("target", axis=1) #alla rader tills den när target
    y = data["target"] #allt om Target leden
    return data, X, y #ger oss tillbaka data, X och y för att återanvända