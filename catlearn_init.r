# # # CatLearn Example Script

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
# # # NEED TO IMPLEMENT CONTINUOUS VS BINOMIAL FEATURE PARAMETER
# # # SHOULD RESET GENERATE NEW WEIGHTS? OR GO BACK TO ORIGINAL WEIGHTS?

# # # load model functions
source('utils.r')

# # # load some data, shj type IV
cases <- demo_cats(6)
  inputs <- cases$inputs
  labels <- cases$labels

# # # number of blocks
blocks <- 20

# # # construct state list
st <- generate_state(num_feats = 3, num_cats = 2, colskip = 4)

# # # construct example ctrl variable
ctrl <- rep(0, blocks * dim(inputs)[1])
ctrl[1] <- 1
ctrl <- c(ctrl, rep(2, 8))

# # # construct the training matrix
tr <- generate_tr(ctrl, inputs, labels, blocks, st)

out <- slpDIVA(st, tr)

results <- cbind(tr, pred = apply(out$out, 1, which.max))

results[,'category'] == results[, 'pred']

sum(results[100:dim(tr)[1],'category'] == results[100:dim(tr)[1], 'pred']) / 
length(100:dim(tr)[1])