## DuoRC

### Evaluation
For evaluation, the answers should be stored in a JSON file with the following format:
```
{
    'id1': 'answer1',
    'id2': 'NA',
    ...
}
```
**Note**: 'NA' is the expected string for `no answer`

Use the provided evaluation script to calculate the exact match and F1 scores
```
python evaluate.py <test-file-path>.json <answer-file-path>.json
```

See https://duorc.github.io/ for more details
