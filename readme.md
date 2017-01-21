# Notes

## the `ctrl=1` setting is weird. 

There is a dedicated function to generate new weights, and I have never wanted to revert back to init weights. You don't want to evaluate performance using the same set of weights many times, just as you don't want to use the same `prez_order` many times.

Is this a catlearn thing that just doesn't apply to DIVA? If not, i think it's [YAGNI](https://en.wikipedia.org/wiki/You_aren't_gonna_need_it).

Another possibility is that `ctrl = 1` means the user wants a fresh model object. But `st` is passed into `slpDIVA`, so it doesn't make sense to produce a fresh `st` within that function. So again i am lead to YAGNI.

## How does this interface with the catlearn CIRPs and datasets?

I just don't see anything in the code to match it up. Is this needed?

## `colskip` code is not DRY

I see lines like this a lot:

```R
X <- tr[number,(st$colskip + 1):(st$colskip + st$num_feats)]
```

It's not a problem, because this sort of mechanism is pretty far under the hood. But for clarity we could generalize to a function that would access features in `tr`. Something like

```R
tr_features <- function(tr, st, idx) {
    return tr[idx,(st$colskip + 1):(st$colskip + st$num_feats)]
}
```
