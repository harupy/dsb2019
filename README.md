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

Test

- Cases where sessions have no previous assessment: ~43%
- Cases where sessions have 1 previous assessment: ~19%
- Cases where sessions have 2 or more previous assessments: ~38%

To address this issue, @fergusoci tried:

- Adjusting training/validation samples to give a greater weighting to this set. (Didn't help, disimproved CV results)
- Building a separate model for this group in particular. (Didn't help)
- Building a binary 0/1 accuracy group model and a) combining the results from this with the overall model. (Didn't help) b) feeding the out of sample probabilities from this as explanatory variables in the main model. (Gave a slight improvement, but nothing ground breaking).
- In post-processing, optimizing regression cutoffs for this group separately. (Didn't help)

## Public and Private
