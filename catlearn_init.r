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

# # # load model functions
source('utils.r')

# # # load some data: shj types 1-6 or 4-class prob (7)
cases <- demo_cats(4)
  inputs <- cases$inputs
  labels <- cases$labels

# # # construct state list
st <- generate_state(input = inputs, categories = labels, colskip = 4, 
  continuous = FALSE, make_wts = TRUE, learning_rate = .15, beta_val = 5)

# # # construct tr object
# 20 training blocks, then a test pass for every example.
tr <- tr_init( dim(inputs)[2] )
tr <- tr_add(inputs, tr, labels = labels, blocks = 20, 
  ctrl = 0, shuffle = TRUE, reset = TRUE)
tr <- tr_add(inputs, tr, labels = labels, ctrl = 2)

# # # result
out <- slpDIVA(st, tr, xtdo = TRUE)

# # # view comparison of predictions to labeled data
get_p_correct <- function(outs, tr){
  test_index <- tr[,1] == 2
  test_ps <- outs[test_index,]
  assignment <- tr[test_index,ncol(tr)]
  p_correct <- test_ps[cbind(seq_along(assignment),assignment)]
  return(p_correct)
}

ps <- get_p_correct(out$out, tr)
print(mean(ps))


# ------------------------------------------------------------ #
# # # # continuous test // iris data

inputs <- as.matrix(iris[,1:4])
labels <- as.numeric(iris$Species)

# # # construct state list
st <- generate_state(input = inputs, categories = labels, colskip = 4, 
    continuous = TRUE, make_wts = TRUE, learning_rate = .15)

# # # construct tr object
# 200 training blocks, then a test pass for every example.
tr <- tr_init( dim(inputs)[2] )
tr <- tr_add(inputs, tr, labels = labels, blocks = 200, 
  ctrl = 0, shuffle = TRUE, reset = TRUE)
tr <- tr_add(inputs, tr, labels = labels, ctrl = 2)

# # # run the model and get results
out <- slpDIVA(st, tr, xtdo = TRUE)

# print accuracy
ps <- get_p_correct(out$out, tr)
print(mean(ps))