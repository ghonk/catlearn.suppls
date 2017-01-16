# # # CatLearn Example Script

# # # load model functions
source('utils.r')

# # # load some data, shj type IV
cases <- demo_cats(4)
  inputs <- cases$inputs
  labels <- cases$labels

# # # number of blocks
blocks <- 20

# # # construct state list
st <- generate_state(3, 2, 3)

str(st)

# # # construct example ctrl variable
ctrl <- rep(0, blocks * dim(inputs)[1])
ctrl[1] <- 1
ctrl <- c(ctrl, rep(2, 8))

# # # stopped at building the tr function

tr <- generate_tr(ctrl, inputs, labels, blocks, st)