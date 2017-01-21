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


# demo_cats
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

# # # forward_pass
# conduct forward pass
#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
forward_pass <- function(in_wts, out_wts, inputs, continuous) {
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
  
  # # # get output activation
  for (category in 1:num_cats) {
  	out_activation[,,category] <- hid_activation %*% out_wts[,,category]
  }
  
  # # # apply output activatio rule
  if(continuous == FALSE) out_activation <- sigmoid(out_activation)

  return(list(out_activation     = out_activation, 
              hid_activation     = hid_activation,
              hid_activation_raw = hid_activation_raw, 
              ins_w_bias         = ins_w_bias))

}

# # # generate_state
# function to construct the state list
generate_state <- function(input, categories, colskip, continuous, wts_range = NULL, 
  wts_center = NULL, num_hids = NULL, learning_rate = NULL, beta_val = NULL,
  model_seed = NULL) {

  # # # input variables
  num_cats  <- length(unique(categories))
  num_feats <- dim(inputs)[2]

  # # # assign default values if needed
  if (is.null(wts_range))      wts_range     <- 1
  if (is.null(wts_center))     wts_center    <- 0 
  if (is.null(num_hids))       num_hids      <- max(num_feats-1, 2)
  if (is.null(learning_rate)) {
    if (continuous) {          learning_rate <- 0.15
    } else {                   learning_rate <- 0.25
    }
  } 
  if (is.null(beta_val))       beta_val      <- 0
  if (is.null(model_seed))     model_seed    <- runif(1) * 100000 * runif(1)

  # # # initialize weight matrices
  wts <- get_wts(num_feats, num_hids, num_cats, wts_range, wts_center)  
  
  # model state is stored as list object
  st <- list(
      beta_val = beta_val, 
      colskip = colskip,
      continuous = continuous, 
      in_wts = wts$in_wts, 
      learning_rate = learning_rate, 
      model_seed = model_seed, 
      num_cats = num_cats, 
      num_feats = num_feats, 
      num_hids = num_hids, 
      out_wts = wts$out_wts,
      wts_center = wts_center, 
      wts_range = wts_range
    )

  return(st = st)
}

# # # generate_tr
# function to construct a sample training matrix
generate_tr <- function(ctrl, inputs, cat_assignment, blocks, st) {
  
  # # # number of examples in input matrix
  eg_num <- dim(inputs)[1]

  # # # generate random presentation order
  prez_order <- as.vector(apply(replicate(blocks, 
    seq(1, eg_num)), 2, sample, eg_num))

  # # # create input matrix. 
  #  First num_feat columns are features. 
  #  Final column is class assignment
  input_mat <- cbind(inputs[prez_order,], cat_assignment[prez_order])

  # # # add block / trial variables to input matrix
  input_mat <- cbind(sort(rep(1:blocks, eg_num)), prez_order, input_mat)

  # # # add test phase
  test_mat <- cbind(0, 1:eg_num, inputs, cat_assignment)

  # # # complete tr matrix
  train_test_mat <- cbind(ctrl, 1:length(ctrl), rbind(input_mat, test_mat))
  
  # # # name the cols in the input matrix
  input_col_names <- c(c('ctrl', 'trial', 'block', 'example'), paste0('f', 1:dim(inputs)[2]), 'category')
  dimnames(train_test_mat) <- list(c(),  input_col_names)
  print(train_test_mat)
  lkdfs
  return(train_test_mat)

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
  
  # # # get differences for each feature between channels
  diff_matrix <- 
    abs(apply(pairwise_comps, 2, function(x) {
      out_activation[,,x[1]] - out_activation[,,x[2]]}))

  # # # reconstruct activation array and get feature diversity means
  diff_array <- array(diff_matrix, dim = c(num_stims, num_feats, num_cats))
  feature_diffs <- apply(diff_array, 2, mean)

  # # # calculate diversities
  diversities <- exp(beta_val * feature_diffs)
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

# slpDIVA
# trains stateful list processor DIVA
#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
slpDIVA <- function(st, tr, xtdo = NULL) {

  # # # set extended output to false if not specified
  if (is.null(xtdo)) {xtdo <- FALSE}
  
  # # # convert targets to 0/1 for binomial input data ONLY
  targets <- tr[,(st$colskip + 1):(st$colskip + st$num_feats)]
  if (st$continuous == FALSE) targets <- global_scale(targets)

  # # # init size parameter variables
  num_stims   <- nrow(tr[tr[,'ctrl'] < 2,]) / max(tr[,'block'])
  out <- matrix(rep(NA, st$num_cats * dim(tr)[1]), 
    ncol = st$num_cats, nrow = dim(tr)[1])
  
  # # # save initial weights
  initial_in_wts  <- st$in_wts
  initial_out_wts <- st$out_wts

  # # # iterate over each trial in the tr matrix 
  for (trial_num in 1:dim(tr)[1]) {
    current_input  <- tr[trial_num, (st$colskip + 1):(st$colskip + st$num_feats)]
    current_target <- targets[trial_num,]
    current_class  <- tr[trial_num, (st$colskip + st$num_feats + 1)]

    # # # if ctrl is set to 1, reset weights to initial values //******* or generate new?
    if (tr[trial_num, 'ctrl'] == 1) {
      st$in_wts  <- initial_in_wts
      st$out_wts <- initial_out_wts 
    }

    # # # complete forward pass
    fp <- forward_pass(st$in_wts, st$out_wts, current_input, st$continuous)

    # # # calculate classification probability
    response <- response_rule(fp$out_activation, current_target, st$beta_val)

    # # # store classification accuracy
    out[trial_num,] = response$ps

    # # # adjust weights based on ctrl setting
    if (tr[trial_num, 'ctrl'] < 2) {
      # # # back propagate error to adjust weights
      class_wts        <- st$out_wts[,,current_class]
      class_activation <- fp$out_activation[,,current_class]

      adjusted_wts <- 
        backprop(class_wts, st$in_wts, class_activation, current_target,  
          fp$hid_activation, fp$hid_activation_raw, fp$ins_w_bias, st$learning_rate)

      # # # set new weights
      st$out_wts[,,current_class] <- adjusted_wts$out_wts
      st$in_wts                   <- adjusted_wts$in_wts
    }
  }

  # # # save extended output
  if (xtdo == TRUE) {
    xtd_output             <- list()
    xtd_output$final_st    <- st
    xtd_output$initial_wts <- 
      list(init_ins = initial_in_wts, init_outs = initial_out_wts)  
    return(list(out = out, xtd_output = xtd_output))
  }  

  return(list(out = out))

}
