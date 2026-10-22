# Titanic Survival Prediction

Predicting which Titanic passengers survived from five features (sex, passenger class, age, whether they travelled with family, and fare), using the `titanic3` dataset of 1,309 passengers.

![Model comparison](model_comparison.png)

## Results (5-fold cross-validated accuracy)

| Model | Accuracy | Spread across folds |
|---|---|---|
| Gradient boosting | 80.1% | ±1.1 |
| Neural network (Keras, 5-5-1) | 79.5% | ±0.9 |
| Logistic regression | 78.6% | ±1.6 |
| Naive Bayes | 75.8% | ±1.8 |
| Baseline (always predict "died") | 61.8% | — |

Every model beats the no-skill baseline by a wide margin, but the top three are within about 1.5 points of each other, so there isn't a clear winner. Gradient boosting matches the neural network with much less setup.

## What's in the notebook

1. **Explore:** violin and point plots of survival by age, family aboard, class and fare, split by sex (`Titanic Graphs.png`)
2. **Prepare:** encode sex, add a `relatives` flag, and fill missing ages with the median inside the model pipeline (about 20% of passengers have no recorded age; they're kept rather than dropped)
3. **Compare:** score each model on the same five stratified folds so the comparison doesn't depend on one train/test split

## Run it

```bash
pip install -r requirements.txt
jupyter notebook RDR_Titanic.ipynb
```

## Data

`titanic3.csv` (with an Excel copy, `titanic3.xlsx`) from the Vanderbilt Biostatistics Titanic dataset. Missing values are recorded as `?` and converted to `NaN` on load.
