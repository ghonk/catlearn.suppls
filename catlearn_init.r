#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #

#             dddddddd                                                                    
#             d::::::d  iiii                                          RRRRRRRRRRRRRRRRR   
#             d::::::d i::::i                                         R::::::::::::::::R  
#             d::::::d  iiii                                          R::::::RRRRRR:::::R 
#             d:::::d                                                 RR:::::R     R:::::R
#     ddddddddd:::::d iiiiiiivvvvvvv           vvvvvvvaaaaaaaaaaaaa     R::::R     R:::::R
#   dd::::::::::::::d i:::::i v:::::v         v:::::v a::::::::::::a    R::::R     R:::::R
#  d::::::::::::::::d  i::::i  v:::::v       v:::::v  aaaaaaaaa:::::a   R::::RRRRRR:::::R 
# d:::::::ddddd:::::d  i::::i   v:::::v     v:::::v            a::::a   R:::::::::::::RR  
# d::::::d    d:::::d  i::::i    v:::::v   v:::::v      aaaaaaa:::::a   R::::RRRRRR:::::R 
# d:::::d     d:::::d  i::::i     v:::::v v:::::v     aa::::::::::::a   R::::R     R:::::R
# d:::::d     d:::::d  i::::i      v:::::v:::::v     a::::aaaa::::::a   R::::R     R:::::R
# d::::::ddddd::::::ddi::::::i       v:::::::v      a::::a    a:::::a RR:::::R     R:::::R
#  d:::::::::::::::::di::::::i        v:::::v       a:::::aaaa::::::a R::::::R     R:::::R
#   d:::::::::ddd::::di::::::i         v:::v         a::::::::::aa:::aR::::::R     R:::::R
#    ddddddddd   dddddiiiiiiii          vvv           aaaaaaaaaa  aaaaRRRRRRRR     RRRRRRR

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #

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