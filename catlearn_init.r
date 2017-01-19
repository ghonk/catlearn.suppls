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
# # # SHOULD RESET GENERATE NEW WEIGHTS? OR GO BACK TO ORIGINAL WEIGHTS?
# # # WHAT SHOULD WE PUT IN XTENDO

# # # load model functions
source('utils.r')

# # # load some data: shj types 1-6 or 4-class prob (7)
cases <- demo_cats(2)
  inputs <- cases$inputs
  labels <- cases$labels

# # # number of blocks
blocks <- 20

# # # construct state list
st <- generate_state(input = inputs, categories = labels, colskip = 4, 
  continuous = FALSE, learning_rate = .15)

# # # construct example ctrl variable
ctrl <- rep(0, blocks * dim(inputs)[1])
ctrl[1] <- 1
ctrl <- c(ctrl, rep(2, dim(inputs)[1]))

# # # construct the training matrix
tr <- generate_tr(ctrl, inputs, labels, blocks, st)

# # # result
out <- slpDIVA(st, tr)

# # # combine predictions and input data
results <- cbind(tr, pred = apply(out$out, 1, which.max))

# # # view comparison of predictions to labeled data
# results[,'category'] == results[, 'pred']

# # # view and calculate accuracy
sum(results[(dim(tr)[1] - dim(inputs)[1]):dim(tr)[1],'category'] == 
  results[(dim(tr)[1] - dim(inputs)[1]):dim(tr)[1], 'pred']) / 
	length((dim(tr)[1] - dim(inputs)[1]):dim(tr)[1])

# # # # continuous test // IRIS data
inputs <- as.matrix(iris[,1:4])
labels <- as.numeric(iris$Species)

# # # number of blocks
blocks <- 200

# # # construct state list
st <- generate_state(input = inputs, categories = labels, colskip = 4, 
	continuous = TRUE, learning_rate = .15)

# # # construct example ctrl variable
ctrl <- rep(0, blocks * dim(inputs)[1])
ctrl[1] <- 1
ctrl <- c(ctrl, rep(2, dim(inputs)[1]))

# # # construct the training matrix
tr <- generate_tr(ctrl, inputs, labels, blocks, st)

# # # run the model and get results
out <- slpDIVA(st, tr)

# # # combine predictions and input data
results <- cbind(tr, pred = apply(out$out, 1, which.max))

# # # view comparison of predictions to labeled data
# results[,'category'] == results[, 'pred']

# # # view and calculate accuracy
sum(results[(dim(tr)[1] - dim(inputs)[1]):dim(tr)[1],'category'] == 
  results[(dim(tr)[1] - dim(inputs)[1]):dim(tr)[1], 'pred']) / 
	length((dim(tr)[1] - dim(inputs)[1]):dim(tr)[1])