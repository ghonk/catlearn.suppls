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

# # # load utilities script
#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
source('utils.r')

# # # Initialize model parameters
#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
model <- list(num_blocks    = 20,
			  num_inits     = 5,
			  wts_range     = 1,
			  num_hids      = 3,
			  learning_rate = 0.15,
			  beta_val      = 5,
			  out_rule      = 'sigmoid') # anything else runs linear

# # # The demo below trains the DIVA model on the Shepard, Hovland, 
# # # and Jenkins' elemental types (1961) plus one 4-class problem.
# # # This demo can be used as a template for your own problem.
#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #

# # # create training results 
training = matrix(rep(0, model$num_blocks * 7), ncol = 7)

# # # initialize model and run it on each SHJ category structure
for (category_type in 1:7) { 

  # # # get shj stimuli
  cases <- demo_cats(category_type)
  model$inputs <- cases$inputs
  model$labels <- cases$labels

  # # # train model
  result <- run_diva(model)

# # # add result to training matrix
training[,category_type] <- result$training

}

# # # display results
print(training)
train_plot(training)
save.image('diva_run.rdata')

# warnings()


