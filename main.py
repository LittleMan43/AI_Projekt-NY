from RealHeartAI import HeartApp #hämtar appen
import joblib

loadmodel = joblib.load("heart_model.pkl")

appen = HeartApp(loadmodel)