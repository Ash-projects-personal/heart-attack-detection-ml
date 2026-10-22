# How this works, in plain words

## Why 303 rows changes everything

Most of the care in this project is about not fooling myself with
a tiny dataset. A test set of 60 patients means one patient is
worth 1.7 accuracy points. A model can look better than another
by luck. That is why every model also gets cross validation:
the data is split five ways, each fold takes a turn as the test
set, and I report the mean. The ranking by cross validation
(logistic regression first) is different from the ranking on the
one test split (Random Forest first), and the cross validated
ranking is the one I trust more.

## Target leakage, the old problem

The old version's label was a formula of its own input features,
including the engineered ones, so high scores were guaranteed
and meaningless. Here the label comes from the dataset itself:
a clinician recorded disease status, measured independently of
anything my code computes. My engineered features are functions
of the inputs only, never of the label, and they still did not
help. That is a real finding, just a negative one.

## What each model brings

Random Forest: strong on small tabular data, gives feature
importances. Logistic regression: simple, and its cross validated
score shows a plain linear model is hard to beat at this size.
MLP: the most flexible and the most data hungry, which is
exactly wrong for 297 rows, and its score shows it.

## Limits

- One clinic, data from the 1980s, 303 patients. Nothing here
  transfers to a modern hospital population without new data and
  proper validation.
- The binarised target mixes four disease severity levels into
  one yes. A real clinical tool would not do that.
- No calibration analysis. A model used for risk would need its
  probabilities checked, not just its accuracy.
