# DSB 2019

## Setup Environment

```
source setup.sh
```

## Notes

- The temporal order must be maintained. When using `groupby`, don't forget to set `sort=False`.
- This is a synchronous rerun code competition and **the private test set has approximately 8MM rows**. You should be mindful of memory in your notebooks to avoid submission errors.
- **In the test set**, we have truncated the history after the start event of a single assessment, **chosen randomly**, for which you must predict the number of attempts.
- Assessment attempts are captured in **event_code 4100 for all assessments except for Bird Measurer, which uses event_code 4110**. If the attempt was correct, it contains "correct":true.
- When the user complete the assessment game, it ends with the event code 2010.

## Assessment distribution

https://www.kaggle.com/c/data-science-bowl-2019/discussion/122767

Train

- Cases where sessions have no previous assessments: ~20%
- Cases where sessions have 1 previous assessment: ~15%
- Cases where sessions have 2 or more previous assessments: ~65%

Test (the private set might have different distribution.)

- Cases where sessions have no previous assessment: ~43%
- Cases where sessions have 1 previous assessment: ~19%
- Cases where sessions have 2 or more previous assessments: ~38%

To address this issue, @fergusoci tried:

- Adjusting training/validation samples to give a greater weighting to this set. (Didn't help, disimproved CV results)
- Building a separate model for this group in particular. (Didn't help)
- Building a binary 0/1 accuracy group model and a) combining the results from this with the overall model. (Didn't help) b) feeding the out of sample probabilities from this as explanatory variables in the main model. (Gave a slight improvement, but nothing ground breaking).
- In post-processing, optimizing regression cutoffs for this group separately. (Didn't help)

## Discussions

https://www.kaggle.com/c/data-science-bowl-2019/discussion/114783#679024

### Truncation

https://www.kaggle.com/c/data-science-bowl-2019/discussion/122149#697661

> For validation only. Truncating for training gave bad results in my case.

### Misses features

https://www.kaggle.com/bhavikapanara/2019-data-science-bowl-some-interesting-features#707199

> What is interesting is that although the features histogram of train vs test looks somehow similar and the features are not highly correlated with other input features, the qwk score on public leaderboard is lower with this features (**went from 0.549 to 0.532**).

## Didn't work

- using 4020 attempts
- augment training data with test
-

## Worked

- Using `title` as `categorical_feature`. (Everyone does this.)
- Truncation (use one assessment for each `installation_id`).
- `percentile_boundaries` (increased the public LB score, but this forces predicted labels to have the same distribution as the train set. The private set might have different label distribution.)

## Techniques

- Data profiler.
- features scripts.
- Be modular.
- doctest for small functions.
- feather.
- Profiler
- Test if testable.

## Why Kaggle doesn't give us the information of submission errors?

> If Kaggle does not reduce number of submissions for failure, we can probe a lot of submissions with 1 bit of information each. For instance, deliberately add a bug to code if the first row in private test is Bird Measurer or not. So, they are smart enough and they have a reason to do so.

## IOError

On Kaggle kernel, The disk utility is limited to 4.9 GB. Saving large amount of data might cause IOError.

## Remove randomness

- Set `seed` and `random_state`.
- Don't use `set`.

## Lessons

- Check every single time (null, zero, constant).
- Test if testable.
