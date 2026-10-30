# Mid Review notes: EconoCausal

Plain-English notes for the Mid Review and viva. Read the code next to each section so you can explain it in your own words.

## The problem in one minute

A normal model predicts who is likely to buy. If we give discounts to likely buyers, we waste money on people who
would have bought anyway. EconoCausal estimates the **Individual Treatment Effect (ITE)**: how much a discount raises
one customer's chance of buying. We then spend a fixed budget on the customers the discount actually moves.

## 1. Confounding and why a simple comparison fails

Marketing historically gave discounts to loyal, high-spending customers. Those customers also buy more even without a
discount. So comparing buyers with and without a discount mixes two things: the effect of the discount and the effect
of who got it.

- Naive difference in purchase rate: **+0.280**
- True effect (known, because the data is simulated): **+0.125**

The naive number is more than double the truth.

## 2. The causal DAG

A DAG (directed acyclic graph) is a diagram of what causes what. Ours says:

- past spend, tenure, visits, price sensitivity and region affect **both** whether a customer got a discount and whether they buy (these are confounders)
- the discount affects purchase (the effect we want)

DoWhy uses the DAG to work out what to adjust for. This is "backdoor adjustment": control for the confounders and the
only remaining path from discount to purchase is the causal one. The DAG is an **assumption** we make, and it says
there are no important confounders we failed to measure.

## 3. Propensity scores and balance

The propensity score is each customer's probability of receiving a discount, estimated from their features with a
logistic regression. Weighting each customer by 1 / (probability of the treatment they actually got) makes treated and
untreated groups look alike. This is inverse propensity weighting (IPW).

We check it worked with the **standardised mean difference (SMD)**: the gap between group averages divided by the
spread. Below 0.1 is considered balanced. Before weighting, past spend had an SMD of 0.75. After weighting it was 0.015.

We clip propensity scores to between 0.01 and 0.99 so no single customer gets a huge weight.

## 4. Double Machine Learning

Steps in plain words:

1. Predict the **outcome** (purchase) from the confounders with one model.
2. Predict the **treatment** (discount) from the same confounders with another model.
3. Subtract the predictions to get **residuals**: what the confounders could not explain.
4. Find how the outcome residual changes with the treatment residual. That relationship is the causal effect.

**Cross-fitting** (we use 5 folds) means no customer is scored by a model that saw them in training. This stops
overfitting from biasing the effect estimate. That is the "double" in Double ML: two helper models, with the ML used
only for the nuisance parts so the effect estimate stays reliable.

## 5. LinearDML vs CausalForestDML

- **LinearDML** assumes the effect changes in a straight line with the features. Simple and stable.
- **CausalForestDML** lets the effect change in any shape, which helps when effects are complex.

On our data they perform about the same (LinearDML slightly better), because the simulated effect is close to linear.
Do not claim the forest is better. Say it should win on more complex data.

## 6. Reading a Qini curve

The x-axis is the share of customers given a discount, best-ranked first. The y-axis is the extra purchases the
discount causes. The grey random line is what you get with no skill. A curve above it means the ranking helps.

Extra purchases from discounting the top 30% (ground truth):

| Method | Extra purchases |
|---|---|
| Linear DML | 96.9 |
| Causal forest | 93.6 |
| Random | 57.0 |
| Predictive "likely buyer" model | 44.0 |

The predictive model is worse than random because customers who are likely to buy are the ones the discount helps
least. Their correlation with the true effect is -0.28.

## 7. Refutation tests

These try to break the estimate (original effect +0.161):

| Test | New effect | Meaning |
|---|---|---|
| Placebo treatment | -0.001 | A fake treatment gives no effect, as it should |
| Random common cause | +0.161 | An irrelevant extra variable changes nothing |
| Data subset (80%) | +0.160 | Estimate is stable across subsets |

Passing these is **evidence, not proof**. They cannot rule out a real confounder we never measured.

## 8. Extra analysis added after Week 2

- **Customer segments:** Persuadables (top 30% effect) had an average true effect of 0.208. Sleeping dogs (lowest 10%) had 0.019. The segments separate customers in the right order.
- **Feature importance:** price sensitivity ranked first (0.558) and past spend second (0.246), matching how the data was simulated.
- **Base learners:** Random Forest, Gradient Boosting and LightGBM all gave similar results. LightGBM had the lowest error (0.0519) and was fastest, but the gaps are within noise on 1,500 customers.
- **Budget optimizer:** with a $5,000 budget, the optimizer reaches about 500 customers and is estimated to win about 81 extra purchases, against about 39 for a blanket $20 rollout. The effect of a $10 discount is assumed to be 70% of a $20 discount. This is an assumption for Week 3 to refine.

## 9. Limitations to admit openly

- The data is simulated. Real campaign data is messier.
- The "observed data" Qini curve is noisy at 1,500 customers. The ground-truth curve is the cleaner proof, and it only exists because we simulated the truth.
- The method assumes no unmeasured confounders.
- The optimizer's discount-size response is an assumption, not something measured.

## 15 likely viva questions

**1. What is the difference between prediction and causation here?**
Prediction asks who will buy. Causation asks who will buy because of the discount. We need the second to avoid wasting money.

**2. What is a confounder? Give an example from your project.**
A variable that affects both treatment and outcome. Past spend: big spenders were more likely to get discounts and more likely to buy anyway.

**3. Why is the naive estimate 0.280 when the truth is 0.125?**
Discounts went to loyal customers who buy more anyway, so the comparison credits the discount with their loyalty.

**4. What is the ITE and how is it different from the ATE?**
The ITE is the effect on one customer. The ATE is the average over all customers. We need the ITE to decide who to target.

**5. What does the DAG do?**
It records our assumptions about what causes what, and DoWhy uses it to decide which variables to adjust for.

**6. What is a propensity score?**
The probability a customer receives the treatment given their features. We use it to weight customers so groups are comparable.

**7. How do you know balancing worked?**
Standardised mean differences fell below 0.1 for every covariate after weighting, for example past spend from 0.75 to 0.015.

**8. Explain Double ML in simple steps.**
Predict outcome and treatment from confounders, take residuals, and relate the two residuals. Cross-fitting keeps the helper models from overfitting.

**9. Why cross-fitting?**
So a customer is never scored by a model that trained on them. Otherwise overfitting leaks into the effect estimate.

**10. Why use a causal forest as well as LinearDML?**
The forest can find effects that change non-linearly. On our nearly linear simulated data they performed similarly.

**11. What is a Qini curve?**
It shows cumulative extra purchases as you target more customers in ranked order. Above the random line means the ranking beats guessing.

**12. Why does the predictive model do worse than random?**
It ranks by purchase chance. Likely buyers would buy anyway, so the discount does little for them.

**13. What do the refutation tests prove?**
That the estimate behaves as a real causal effect should: it vanishes for a fake treatment and is stable under noise. They do not prove there is no hidden confounder.

**14. How does the optimizer decide who gets what?**
It solves a mixed-integer program: each customer gets at most one discount tier, total spend stays within budget, and it maximises estimated extra purchases.

**15. What are the weaknesses of your project?**
Simulated data, an assumption of no hidden confounders, noisy evaluation on observed data at this sample size, and an assumed response for smaller discounts.
