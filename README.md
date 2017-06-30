catlearn.suppls
===============

A constellation of functions that work with the catlearn package to facilitate the broad modeling goals of the [Learning and Representation in Cognition Laboratory](http://kurtzlab.psychology.binghamton.edu/).

*Package under active development*

The package can be installed using these commands:

``` r
install.packages("devtools")
devtools::install_github("ghonk/catlearn.suppls")
```

`catlearn.suppls` is intended to be used as a supplement to the `catlearn` package. It facilitates modeling with the DIVergent Autoencoder architecture (`slpDIVA()`) and features a suite of functions under development for active research projects.

Package Demonstration
---------------------

First load the packages.

``` r
library(catlearn)
library(catlearn.suppls)
```

Then load some data from 'get\_test\_inputs' and assign variables for the properties of the data. The package includes some classic category structures and we'll use a 4D family resemblance + unidimensional rule category structure for this demo.

``` r
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
  beta_val = 0, continuous = FALSE, in_wts = NULL, out_wts = NULL, wts_range = 1,
  wts_center = 0, colskip = 4)
```

We then use the `catlearn.suppls` package to create a training matrix, `tr`.

``` r
# tr_init makes the empty training matrix
tr <- tr_init(nfeats, ncats)

# tr_add fills in the data and procedure (i.e., training, test, model reset)
tr <- tr_add(inputs = ins, tr = tr, labels = labs, blocks = 12, ctrl = 0, 
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


trn_result <- cbind(tr, round(diva_model$out, 4))
tail(trn_result)
```

    ##       ctrl trial block example f1 f2 f3 f4 t1 t2     o1     o2
    ## [91,]    0    91    12       8  1 -1 -1 -1 -1  1 0.0906 0.9094
    ## [92,]    0    92    12       4 -1  1  1  1  1 -1 0.9021 0.0979
    ## [93,]    0    93    12       1  1  1  1  1  1 -1 0.9313 0.0687
    ## [94,]    0    94    12       3  1 -1  1  1  1 -1 0.9414 0.0586
    ## [95,]    0    95    12       6 -1 -1  1 -1 -1  1 0.0861 0.9139
    ## [96,]    0    96    12       2  1  1 -1  1  1 -1 0.8974 0.1026

`model_accuracy` can be used to extract the response probabilities for the target category (for every training step or averaged across blocks)

``` r
model_accuracy(tr, diva_model$out, blocks = TRUE)
```

    ##  [1] 0.5804156 0.7402500 0.7827221 0.8063014 0.8286746 0.8481832 0.8626191
    ##  [8] 0.8801585 0.8967152 0.9018396 0.9122791 0.9259406