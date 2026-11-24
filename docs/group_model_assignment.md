# Group model assignment: your project's first model

Do for your team's project what you did in the individual assignment for the same model family, on your own data, as one team: one repo, one set of slides.

## The question

State the business question your project's model answers, in one sentence, from the point of view of the person who will act on it. What decision does the model inform, and what does your product do today without it?

## The model

The family your team chose in your team meeting: **regression**, **decision tree**, or **clustering**. One model, v1: the best you think you can build from the data you have now. Choose the features you think matter and go straight to them.

## Context

State the decision rule your product will apply on top of the model (buy or pass, act or hold, which segment gets what). State the **baseline**: what the product does with no model (the average for regression; the most common class for a decision tree; a single group for clustering).

## The data

Describe the dataset you are fitting on: source, rows, period, one row per what, and a data dictionary with the outcome marked (none for clustering). If the data is synthetic or scraped, say so and say how.

## The task

1. **Split the data.** Randomly set aside 30% as a test set. Fit everything on the remaining 70%. If your data set is under 30 data points, skip the split and use the whole data set to train.
2. **Fit the model.** Fit v1 on the training rows. Read the output and say what it means for the business.
3. **Evaluate fit on the test rows, baseline vs v1.** Regression: adjusted R² from the training fit; R² and typical error in the outcome's units on the test rows. Decision tree: confusion matrix, accuracy, precision, recall on the test rows, plus training vs test accuracy. Clustering: the elbow plot on training, the test elbow recomputed from the trained centers, and confirmation that the elbow is at the same k.
4. **Make the decision.** Write the rule as a function that takes one new input your product will see and returns the decision. Run it on at least one realistic new input and show the result.
5. **Quantify it, baseline vs v1.** Put a value on what the model changes, in the business's units, with the assumptions stated. Where you have no real cost or revenue figures, make reasonable ones up and label them as assumptions.
6. **Present it.** Two slides for the business owner, not for a statistics audience.

The model runs standalone: a script in your repo that loads your data, fits v1, prints the fit evaluation, and returns a decision for a new input. It does not need to be wired into your app yet.

## Deliverables
Note: you may use AI throughout and it is expected that you do so for coding, however, do not use AI to generate your slides. The value of this assignment is interpreting the technical output and synthesizing for business decision-makers. It is better to have succinct black text on white background that you wrote yourself rather than volumes of AI-generated nonsense for your "boss" to have to parse. (Asking AI to write python code to create graphs/charts for your slides is expected, copying-pasting an a graph made by ChatGPT/Claude/Gemini is not.)

One submission per team: **one PDF.**

1. **Title slide:** team name, members, and the link to your team's GitHub repo, with the path to the model script. The script reproduces every number on the slides.
2. **Two slides covering:**
   - Data: source, rows, period, the 70/30 split.
   - Model specification: outcome (if any), features, model family, and the key fit statistic from the training fit.
   - Fit baseline vs v1 comparisons on the test rows, with a chart.
   - Business decision and value quantification:
   -- Headline in one sentence- what the model changes for the business and what it is worth
   -- The decision rule and the result for your new input
   -- Compare value quantification of baseline vs v1

Every team member should be able to explain how every number on the slides was calculated.
