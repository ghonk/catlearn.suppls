# # # generate_state
#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
#' 
#' Construct the state list
#' 
#' @param input Complete matrix of inputs for training
#' @param categories Vector of category assignment values
#' @param colskip Scalar for number of columns to skip in the tr matrix
#' @param continuous Boolean value indicating if inputs are continuous
#' @param make_wts Boolean value indicating if initial weights should be generated
#' @param wts_range Scalar value for the range of the generated weights
#' @param wts_center Scalar value for the center of the weights
#' @param num_hids Scalar value for the number of hidden units in the model architecture
#' @param learning_rate Learning rate for weight updates through backpropagation
#' @param beta_val Scalar value for the beta parameter
#' @param model_seed Scalar value to set the random seed
#' @return List of the model hyperparameters and weights (by request) 
generate_state <- function(input, categories, colskip, continuous, make_wts,
  wts_range = NULL,  wts_center    = NULL, 
  num_hids  = NULL,  learning_rate = NULL, 
  beta_val  = NULL,  model_seed    = NULL) {

  # # # input variables
  num_cats  <- length(unique(categories))
  num_feats <- dim(input)[2]

  # # # assign default values if needed
  if (is.null(wts_range))      wts_range     <- 1
  if (is.null(wts_center))     wts_center    <- 0 
  if (is.null(num_hids))       num_hids      <- 3
  if (is.null(learning_rate))  learning_rate <- 0.15
  if (is.null(beta_val))       beta_val      <- 0
  if (is.null(model_seed))     model_seed    <- runif(1) * 100000 * runif(1)

  # # # initialize weight matrices
  if (make_wts == TRUE) {
    wts <- get_wts(num_feats, num_hids, num_cats, wts_range, wts_center)  
  } else {
    wts <- list(in_wts = NULL, out_wts = NULL)
  }

  return(st = list(num_feats = num_feats, num_cats = num_cats, colskip = colskip,
    continuous = continuous, wts_range = wts_range, wts_center = wts_center, 
    num_hids = num_hids, learning_rate = learning_rate, beta_val = beta_val, 
    model_seed = model_seed, in_wts = wts$in_wts, out_wts = wts$out_wts))

}

# tr_init
#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
#'
#' Initialize a tr object
#' 
#' @param n_feats Number of features (integer, > 0)
#' @param feature_type String type: numeric (default), logical, etc
#' @return An initialized dataframe with the appropriate columns
tr_init <- function(n_feats, n_cats, feature_type = 'numeric') {

  feature_cols <- list()
  for(f in 1:n_feats) {
    feature_cols[[paste0('f', f)]] = feature_type
  }

  target_cols <- list()
  for(c in 1:n_cats) {
   target_cols[[paste0('t', c)]] = 'integer' 
  }

  other_cols <- list(
    ctrl = 'integer', 
    trial = 'integer',
    block = 'integer',
    example = 'integer'
  )

  all_cols <- append(other_cols, c(feature_cols, target_cols))

  # create empty df with column types specified by all_cols
  empty_tr <- data.frame()
  for (col in names(all_cols)) {
      empty_tr[[col]] <- vector(mode = all_cols[[col]], length = 0)
  }

  empty_tr <- as.matrix(empty_tr)

  return(empty_tr)
}

# tr_add
#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
#'
#' Add trials to an initialized tr object
#' 
#' @param inputs Matrix of feature values for each item
#' @param tr Initialized trial object
#' @param labels Integer class labels for each input. Default NULL
#' @param blocks Integer number of repetitions. Default 1
#' @param shuffle Boolean, shuffle each block. Default FALSE
#' @param ctrl Integer control parameter, applying to all inputs. Default 2
#' @param reset Boolean, reset state on first trial (ctrl=1). Default FALSE
#' @return An updated dataframe
tr_add <- function(inputs, tr,
  labels = NULL, 
  blocks = 1, 
  ctrl = NULL, 
  shuffle = FALSE,
  reset = FALSE) {

  # some constants
  numinputs <- dim(inputs)[1]
  numfeatures <- dim(inputs)[2]
  numtrials <- numinputs * blocks

  # obtain labels vector if needed
  if (is.null(labels)) labels <- rep(NA, numinputs)
  
  # generate order of trials
  if (shuffle) {
    examples <- as.vector(apply(replicate(blocks,seq(1, numinputs)), 2, 
      sample, numinputs))
  } else{
    examples <- as.vector(replicate(blocks, seq(1, numinputs)))
  }
  
  # create rows for tr
  rows <- list(
    ctrl = rep(ctrl, numtrials),
    trial = 1:numtrials,
    block = sort(rep(1:blocks, numinputs)),
    example = examples
  )

  # add features to rows list
  for(f in 1:numfeatures){
    rows[[paste0('f', f)]] = inputs[examples, f]
  }

  # add category labels
  num_cats <- max(labels)

  label_mat <- matrix(-1, ncol = num_cats, nrow = numinputs)

  for (i in 1:numinputs) {
    label_mat[i, labels[i]] <- 1
  }

  rows <- data.frame(rows)
  rows <- cbind(rows, label_mat)

  # reset on first trial if needed
  if (reset) {rows$ctrl[1] <- 1}

  rows <- as.matrix(data.frame(rows))
  return(rbind(tr, rows))
}