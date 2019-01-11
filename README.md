## DuoRC
DuoRC contains 186,089 unique question-answer pairs created from a collection of 7680 pairs of movie plots where each pair in the collection reflects two versions of the same movie.

### Evaluation
For evaluation, the answers should be stored in a JSON file with the following format:
```
{
    'id1': 'answer1',
    'id2': 'NA',
    ...
}
```
**Note**: `NA` is the expected string for `no answer`

Next, use the provided evaluation script to calculate the exact match and F1 scores:
```
python evaluate.py <test-file-path>.json <answer-file-path>.json
```

See https://duorc.github.io/ for more details
