catlearn.suppls
===============

A constellation of functions that work with the catlearn package to facilitate the broad modeling goals of the [Learning and Representation in Cognition Laboratory](http://kurtzlab.psychology.binghamton.edu/).

*Package under active development*

The package can be installed using these commands:

``` r
install.packages("devtools")
devtools::install_github("ghonk/catlearn.suppls")
```

`catlearn.suppls` is intended to be used as a supplement to the `catlearn` package. It facilitates modeling with the DIVergent Autoencoder architecture and features a suite of functions under development for active research projects.

Modeling example
----------------

First load the packages and data

``` r
library(catlearn)
```

    ## Warning: package 'catlearn' was built under R version 3.2.5

``` r
library(catlearn.suppls)

# get some sample data
test_inputs <- get_test_inputs('unifr')

# set inputs and class labels
ins  <- test_inputs$ins
labs <- test_inputs$labels

# get number of categories and features
nfeats <- dim(ins)[2]
ncats <- length(unique(labs))
```

Then set the model parameters. The design pattern for `catlearn` is called a *stateful list processor*. This means we provide the model with a list that contains all of the models hyperparamters and it will return to us a similar list that includes the model's learned parameters (weights).

``` r
# construct a state list
st <- list(learning_rate = 0.25, num_feats = nfeats, num_hids = 3, num_cats = ncats,
  beta_val = 5, continuous = FALSE, in_wts = NULL, out_wts = NULL, wts_range = 1,
  wts_center = 0, colskip = 4)
```

We then use the `catlearn.suppls` package to create a training matrix, `tr`.

``` r
# tr_init makes the empty training matrix
tr <- tr_init(nfeats, ncats)

# tr_add fills in the data and procedure (i.e., training, test, model reset)
tr <- tr_add(inputs = ins, tr = tr, labels = labs, blocks = 100, ctrl = 0, 
  shuffle = TRUE, reset = TRUE)
```

Finally, we run the model with our state list `st` and training matrix `tr`.

``` r
diva_model <- slpDIVA(st, tr)
```

To examine performance we can match the output (response probabilities) to the training matrix `tr`.

``` r
# name the output columns
colnames(diva_model$out) <- paste0('o', 1:dim(diva_model$out)[2])
# combine the output with the input
trn_result <- cbind(tr, round(diva_model$out, 4))
```

<!-- And to plot the results, we just need to match the category labels to the response probabiliites. -->
<!-- ```{r} -->
<!-- diva_model$out[1:dim(diva_model$out)[1],apply(trn_result[,c('t1', 't2')], 1, which.max)] -->
<!-- ``` -->
