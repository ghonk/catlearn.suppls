#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #

#                          tttt            iiii  lllllll                  
#                       ttt:::t           i::::i l:::::l                  
#                       t:::::t            iiii  l:::::l                  
#                       t:::::t                  l:::::l                  
# uuuuuu    uuuuuuttttttt:::::ttttttt    iiiiiii  l::::l     ssssssssss   
# u::::u    u::::ut:::::::::::::::::t    i:::::i  l::::l   ss::::::::::s  
# u::::u    u::::ut:::::::::::::::::t     i::::i  l::::l ss:::::::::::::s 
# u::::u    u::::utttttt:::::::tttttt     i::::i  l::::l s::::::ssss:::::s
# u::::u    u::::u      t:::::t           i::::i  l::::l  s:::::s  ssssss 
# u::::u    u::::u      t:::::t           i::::i  l::::l    s::::::s      
# u::::u    u::::u      t:::::t           i::::i  l::::l       s::::::s   
# u:::::uuuu:::::u      t:::::t    tttttt i::::i  l::::l ssssss   s:::::s 
# u:::::::::::::::uu    t::::::tttt:::::ti::::::il::::::ls:::::ssss::::::s
#  u:::::::::::::::u    tt::::::::::::::ti::::::il::::::ls::::::::::::::s 
#   uu::::::::uu:::u      tt:::::::::::tti::::::il::::::l s:::::::::::ss  
#     uuuuuuuu  uuuu        ttttttttttt  iiiiiiiillllllll  sssssssssss    

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #

# # # backprop
# backpropagate error and update weights
#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
backprop <- function(out_wts, in_wts, out_activation, current_target, 
                     hid_activation, hid_activation_raw, ins_w_bias, learning_rate){

  # # # calc error on output units
  out_delta <- 2 * (out_activation - current_target)
  
  # # # calc error on hidden units
  hid_delta <- out_delta %*% t(out_wts)
  hid_delta <- hid_delta[,2:ncol(hid_delta)] * sigmoid_grad(hid_activation_raw)
  
  # # # calc weight changes
  out_delta <- learning_rate * (t(hid_activation) %*% out_delta)
  hid_delta <- learning_rate * (t(ins_w_bias) %*% hid_delta)

  # # # adjust wts
  out_wts <- out_wts - out_delta
  in_wts <- in_wts - hid_delta

  return(list(out_wts = out_wts, 
              in_wts  = in_wts))

}

# # # forward_pass
# conduct forward pass
#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
forward_pass <- function(in_wts, out_wts, inputs, out_rule) {
  # # # init needed vars
  num_feats <- ncol(out_wts)
  num_cats  <- dim(out_wts)[3]
  num_stims <- nrow(inputs)
  if (is.null(num_stims)) {num_stims <- 1}

  
  # # # add bias to ins
  bias_units <- matrix(rep(1, num_stims), ncol = 1, nrow = num_stims)
  ins_w_bias <- cbind(bias_units,
    matrix(inputs, nrow = num_stims, ncol = num_feats, byrow = TRUE))

  # # # ins to hids propagation
  hid_activation_raw <- ins_w_bias %*% in_wts
  hid_activation <- sigmoid(hid_activation_raw)

  # # # add bias unit to hid activation
  hid_activation <- cbind(bias_units, hid_activation)  

  # # # hids to outs propagation
  out_activation <- array(rep(0, (num_stims * num_feats * num_cats)), 
    dim = c(num_stims, num_feats, num_cats))
  
  # # NEED VECTORIZED HERE?
  # # # get output activation
  for (category in 1:num_cats) {
  	out_activation[,,category] <- hid_activation %*% out_wts[,,category]
  }
  
  # # # apply output activatio rule
  if(out_rule == 'sigmoid') {
  	out_activation <- sigmoid(out_activation)
  }

  return(list(out_activation     = out_activation, 
              hid_activation     = hid_activation,
              hid_activation_raw = hid_activation_raw, 
              ins_w_bias         = ins_w_bias))

}

# # # get_wts
# generate net weights
#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
get_wts <- function(num_feats, num_hids, num_cats, wts_range, wts_center) {
  # # # set bias
  bias <- 1
  
  # # # generate wts between ins and hids
  in_wts <- 
    (matrix(runif((num_feats + bias) * num_hids), ncol = num_hids) - 0.5) * 2 
  in_wts <- wts_center + (wts_range * in_wts)

  # # # generate wts between hids and outs
  out_wts <- 
    (array(runif((num_hids + bias) * num_feats * num_cats), 
      dim = c((num_hids + bias), num_feats, num_cats)) - 0.5) * 2
  out_wts <- wts_center + (wts_range * out_wts)   
  
  return(list(in_wts  = in_wts, 
              out_wts = out_wts))

}

# # # global_scale
# scale inputs to 0/1
#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
global_scale <- function(x) { x / 2 + 0.5 }

# # # train_plot
# function to produce line plot of training
#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
train_plot <- function(training) {
  # # # get dimensions
  n_cats <- dim(training)[2]
  xrange <- c(1, dim(training)[1])
  yrange <- range(training)

  # # # open plot device
  pdf('training_plot.pdf')

  # # # create frame
  plot(xrange, c(.01, yrange[2]), xlab = 'Block Number', ylab = 'Accuracy')

  # # # aesthetics 
  colors <- rainbow(n_cats)
  line_type <- c(1:n_cats)
  plot_char <- seq(18, 18 + n_cats, 1)

  # # # plot lines
  for (i in 1:n_cats) {
    target_cat <- training[,i]
    lines(seq(1, xrange[2], 1), target_cat, type = 'b', lwd = 1.5, 
      lty = line_type[i], col = colors[i],  pch = plot_char[i])
  }

  # # # title and legend
  title('DIVA Training Accuracy across Blocks')
  legend('bottomright', y = NULL, 1:n_cats, cex = 0.8, col = colors, 
    pch = plot_char, lty = line_type, title = 'SHJ Categories')

  # # # produce plot
  dev.off()

}

# response_rule
# convert output activations to classification
#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
response_rule <- function(out_activation, target_activation, beta_val){
  num_feats <- ncol(out_activation)
  num_cats  <- dim(out_activation)[3]
  num_stims <- nrow(target_activation)
  if (is.null(num_stims)) {num_stims <- 1}

  # # # calc error  
  ssqerror <- array(as.vector(
    apply(out_activation, 3, function(x) {x - target_activation})),
      c(num_stims, num_feats, num_cats))
  ssqerror <- ssqerror ^ 2
  ssqerror[ssqerror < 1e-7] <- 1e-7

  # # # get list of channel comparisons
  pairwise_comps <- combn(1:num_cats, 2)
  
  # # # get differences between each feature
  dist_mat <- as.matrix(dist(out_activation, upper = TRUE))

  # # # get key differences between each feature for all channels
  pairwise_diffs <- t(apply(pairwise_comps, 2, function(x) {
    diag(dist_mat[((x[1] * num_feats) - (num_feats - 1)):(x[1] * num_feats),
      ((x[2] * num_feats) - (num_feats - 1)):(x[2] * num_feats)])
  }))

  # # # calculate diversities
  diversities <- exp(beta_val * colMeans(pairwise_diffs))
  diversities[diversities > 1e+7] <- 1e+7

  # divide diversities by sum of diversities
  fweights = diversities / sum(diversities)

  # # # apply focus weights; then get sum for each category
  ssqerror <- t(apply(ssqerror, 3, function(x) sum(x * fweights))) 
  ssqerror <- 1 / ssqerror


return(list(ps       = (ssqerror / sum(ssqerror)), 
            fweights = fweights, 
            ssqerror = ssqerror))

}

# run_diva
# trains vanilla diva
#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
run_diva <- function(model) {
  
  # # # get new seed
  model.seed <- runif(1) * 100000 * runif(1)
  set.seed(model.seed)
  
  # # # set mean value of weights
  model$wts_center <- 0 
  # # # convert targets to 0/1
  model$targets <- global_scale(model$inputs) 
  # # # init size parameter variables
  model$num_feats   <- ncol(model$inputs)
  model$num_stims   <- nrow(model$inputs)
  model$num_cats    <- length(unique(model$labels))
  model$num_updates <- model$num_blocks * model$num_stims
  # # # init training accuracy matrix
  training <- 
    matrix(rep(NA, model$num_updates * model$num_inits), 
      nrow = model$num_updates, ncol = model$num_inits)
  
  # # # initialize and run DIVA models
  for (model_num in 1:model$num_inits) {
    
    # # # generate weights
    wts <- get_wts(model$num_feats, model$num_hids, model$num_cats, model$wts_range, model$wts_center)
    
    # # # generate random presentation order
    prez_order <- as.vector(apply(replicate(model$num_blocks, 
      seq(1, model$num_stims)), 2, sample, model$num_stims))

    # # # iterate over each trial in the presentation order 
    for (trial_num in 1:model$num_updates) {
      current_input  <- model$inputs[prez_order[[trial_num]], ]
      current_target <- model$targets[prez_order[[trial_num]], ]
      current_class  <- model$labels[prez_order[[trial_num]]] 

      # # # complete forward pass
      fp <- forward_pass(wts$in_wts, wts$out_wts, current_input, model$out_rule)

      # # # calculate classification probability
      response <- response_rule(fp$out_activation, current_target, model$beta_val)

      # # # store classification accuracy
      training[trial_num, model_num] = response$ps[current_class]

      # # # back propagate error to adjust weights
      class_wts <- wts$out_wts[,,current_class]
      class_activation <- fp$out_activation[,,current_class]

      adjusted_wts <- backprop(class_wts, wts$in_wts, class_activation, current_target,  
               fp$hid_activation, fp$hid_activation_raw, fp$ins_w_bias, model$learning_rate)

      # # # set new weights
      wts$out_wts[,,current_class] <- adjusted_wts$out_wts
      wts$in_wts <- adjusted_wts$in_wts
  
    }

  }

training_means <- 
  rowMeans(matrix(rowMeans(training), nrow = model$num_blocks, ncol = model$num_stims, byrow = TRUE))

return(list(training = training_means,
            model    = model))

}

# shj_cats
# loads shj category structures
#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
demo_cats <- function(type){
  
  in_pattern <- 
    matrix(c(-1, -1, -1,
	           -1, -1,  1,
	           -1,  1, -1,
  	         -1,  1,  1,
	            1, -1, -1,
	            1, -1,  1,
	            1,  1, -1,
	            1,  1,  1), 
      nrow = 8, ncol = 3, byrow = TRUE)		

  cat_assignment <- 
    matrix(c(1, 1, 1, 1, 2, 2, 2, 2,  # type I
             1, 1, 2, 2, 2, 2, 1, 1,  # type II
             1, 1, 2, 1, 1, 2, 2, 2,  # type III
             1, 1, 1, 2, 1, 2, 2, 2,  # type IV
             2, 1, 1, 1, 1, 2, 2, 2,  # type V
             1, 2, 2, 1, 2, 1, 1, 2,  # type VI
             1, 1, 2, 2, 3, 3, 4, 4), # type II multiclass  
      ncol = 8, byrow = TRUE)

return(list(inputs = in_pattern, 
			      labels = cat_assignment[type,]))

}

# sigmoid
# returns sigmoid evaluated elementwize in X
#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
sigmoid <- function(x) {
  g = 1 / (1 + exp(-x))

return(g)

}

# sigmoid gradient
# returns the gradient of the sigmoid function evaluated at x
#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
sigmoid_grad <- function(x) {
  
return(g = ((sigmoid(x)) * (1 - sigmoid(x))))

}