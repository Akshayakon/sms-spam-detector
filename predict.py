"""Classify messages with the trained model.  Usage: python predict.py "your message" """
import sys
import joblib
from train_utils import clean  # noqa

model = joblib.load("spam_model.joblib")
messages = sys.argv[1:] or [
    "WINNER!! You have won a £1000 prize. Call 09061701461 now to claim!",
    "Hey, are we still meeting for lunch at 1?",
]
for m in messages:
    p = model.predict_proba([clean(m)])[0][1]
    print(f"{'SPAM' if p > 0.5 else 'HAM ':4} ({p:.1%} spam)  {m}")
