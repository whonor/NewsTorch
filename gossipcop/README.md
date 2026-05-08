# Differences between the cold-start splits and warm-start user splits
([rebuild_gossipcop_splits.py](../scripts/rebuild_gossipcop_splits.py))

• The new split is much easier and much more aligned with training. The high AUC is expected.

Key reasons:

1. The old test split had severe label shift.

Old positive target distribution:

train: 13.4% fake
test:  59.7% fake

New split:

train: 18.28% fake
test:  18.37% fake

So the model is no longer trained mostly on real-positive behavior and tested mostly on fake-positive behavior.

2. No cold test users now.

New split:

test users seen in train: 100%
old test users seen in train: 42.3%

NRMS is a personalization model. When test users are unseen or weakly seen, its user-history encoder has much less useful signal. The new split evaluates mostly warm-user
recommendation.

3. Most test target items are now seen in train.

New split diagnostics:

test row target seen in train: 88.3%
test unique target items seen in train: 85.4%
old unique test target items seen in train: 12.4%

That is a huge change. The model is now often ranking items whose text/topic patterns appeared as positives during training.

4. Same-user preference patterns are much better represented.

For each test row, the same user has on average about:

5.4 train rows with the same fake/real target label

Even though fake/real is not explicitly fed as category, the article text/publisher/topic distribution can correlate with that behavior. This makes the user-history signal much
stronger.

5. It is not exact duplicate leakage, but it is an easier warm-start split.

I checked:

Exact user-target pair seen in train: 0%
Positive target already in row history: 0%
Clicked candidate already positive for same user in train: 0%

So it is not trivially memorizing identical user-item positives. But it is still a warm-user, mostly warm-item evaluation.

Bottom line: the new AUC is high because the split now measures warm-start personalized ranking under matched fake/real distribution, while the old split measured a much harder setting
with distribution shift and cold users/items.

For reporting, describe this clearly:

We rebuilt the split to control target-label distribution and ensure user overlap across train/validation/test. This removes cold-user effects and fake/real target-label shift, so the
resulting metrics represent warm-start recommendation performance rather than cold-start temporal/generalization performance.

I would also report both split types if possible: old split as stress/cold-shift performance, new split as controlled warm-start performance.