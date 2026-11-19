# SMS Spam Detector

Detects spam SMS messages using **TF-IDF + Multinomial Naive Bayes** (scikit-learn, NLTK).
Dataset: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (5,572 labeled messages).

## Run
```
pip install -r requirements.txt
python train.py                      # trains, tunes, evaluates, saves spam_model.joblib
python predict.py "Free entry! Text WIN to 80086"
```

## How it works
1. **Clean** (NLTK): lowercase, remove punctuation/stopwords, Porter-stem words.
2. **Vectorize**: TF-IDF (unigrams + bigrams).
3. **Classify**: Multinomial Naive Bayes; hyper-parameters tuned with 5-fold grid search.
4. **Evaluate** on a stratified 20% held-out test set.

## Results (test set, 1,115 messages)
| Metric | Value |
|---|---|
| Accuracy | 98.8% |
| Spam precision | 99.3% |
| Spam recall | 92.0% |
| Spam F1 | 95.5% |

Only 1 legitimate message was flagged as spam; 12 of 149 spam messages were missed.

## Example predictions
```
$ python predict.py "Free entry in a weekly competition! Text WIN to 80086 now" "Can you pick up milk on your way home?"
SPAM (99.9% spam)  Free entry in a weekly competition! Text WIN to 80086 now
HAM  (0.2% spam)  Can you pick up milk on your way home?
```
More tested: "URGENT: your account is suspended. Call 08001234567 immediately" -> SPAM (99.6%);
"Meeting moved to 3pm, see you in the conference room" -> HAM (0.5%).

## Files
- `train.py` – training + evaluation
- `train_utils.py` – text cleaning (shared)
- `predict.py` – classify new messages
- `data/SMSSpamCollection` – dataset
