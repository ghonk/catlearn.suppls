#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
#' @title Average Response Probabilities
#'
#' @description
#' Function that takes a list of model response probabilities and averages.
#'
#' @param resp_probs_list List of response probability vectors
#' @return Vector of average response probabilities
avg_resp_probs <- function(resp_probs_list) {

  # # # reshape resp prob list
  resp_probs_mat <- matrix(unlist(resp_probs_list),
    ncol = length(resp_probs_list),
    nrow = length(resp_probs_list[[1]]))

  # # # average across blocks (rows)
  return(rowMeans(resp_probs_mat))
}

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
#' @title Examine model
#'
#' @description
#' Function to examine the parameters, state and classification probabilites
#' of a model across a training or test matrix.
#'
#' @param st List of initial state of the model
#' @param tr Matrix of training or test examples
#' @param model String indicating which model is to be examined
#' \itemize{
#'      \item \code{slpDIVA} DIVA model
#'      \item \code{slpALCOVE} ALCOVE model
#'      \item \code{slpDIVAdev} Developmental DIVA (only tested option)
#'      }
#' @return \code{out} List of lists for each trial containing trial-by-trial
#' model information including:
#'  \itemize{
#'      \item \code{init_wts} List of weights
#'          \itemize{
#'              \item \code{in_wts} Matrix of input to hidden weight (including bias)
#'              \item \code{out_wts} Array of hidden to output weights (including bias)
#'          }
#'      \item \code{inputs} Matrix of complete input information for trial
#'      \item \code{hidden_act} Matrix of hidden unit activation for trial
#'      \item \code{result} List that contains the model's post-trial state
#'           \itemize{
#'               \item \code{st} List of the model's end-trial state (see \code{?slpDIVA})
#'               \item \code{out} Vector of respond probabilities
#'           }
#'      }
#' @export
examine_model <- function(st, tr, model) {

  # # # assign model type to general call
  exam_model <- get(model)
  # # # get dims
  train_dims <- dim(tr)
  # # # initialize list
  out <- list()

  # # # run model for each training item
  for (i in 1:train_dims[1]) {
    # # # initial weights
    initial_wts <- list(in_wts = st$in_wts, out_wts = st$out_wts)
    # # # trial inputs
    inputs <- tr[i, , drop = FALSE]
    # # # trial result
    trial_result <- exam_model(st, tr[i, , drop = FALSE])

    # # # save unit activation
    if (is.null(st$in_wts) == FALSE) {
      hidden_act <-
        st$in_wts * c(1, inputs[ , (st$colskip + 1):(st$colskip + st$num_feats)])
    } else {
      hidden_act <-
        trial_result$st$in_wts *
          c(1, inputs[ , (st$colskip + 1):(st$colskip + st$num_feats)])
    }

    # # # update weights if training is on
    st$in_wts  <- trial_result$st$in_wts
    st$out_wts <- trial_result$st$out_wts

    # # # save information to out list
    out[[paste0('Trial_', i)]] <-
      list(initial_wts = initial_wts,
                inputs = inputs,
            hidden_act = hidden_act,
                result = trial_result)

  }

return(out)

}

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
#' @title Construct DIVA state list
#'
#' @description
#' Construct the state list. The primary use of this function is to construct a
#' state list with the default DIVA parameters or generate data-appropriate weights
#' (particularly when the random seed needs to be set).
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
#' @param phi Scalar value for response mapping (Default = 1, meaning no effect)
#' @param model_seed Scalar value to set the random seed
#' @return List of the model hyperparameters and weights
#' @export

generate_state <- function(input, categories, colskip, continuous, make_wts,
  wts_range  = NULL,  wts_center    = NULL,
  num_hids   = NULL,  learning_rate = NULL,
  beta_val   = NULL,  phi           = NULL,
  model_seed = NULL) {

  # # # input variables
  num_cats  <- length(unique(categories))
  num_feats <- dim(input)[2]

  # # # assign default values if needed
  if (is.null(wts_range))      wts_range     <- 1
  if (is.null(wts_center))     wts_center    <- 0
  if (is.null(num_hids))       num_hids      <- 3
  if (is.null(learning_rate))  learning_rate <- 0.15
  if (is.null(beta_val))       beta_val      <- 0
  if (is.null(phi))            phi           <- 1
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
    phi = phi, model_seed = model_seed, in_wts = wts$in_wts,
    out_wts = wts$out_wts))

}

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
#' @title Produce model inputs from a set of classic category structures
#'
#' @description
#' Function to grab inputs that might be useful for model testing
#'
#' @param target_cats String corresponding to category structure label
#' \itemize{
#'     \item \code{unifr} Unidimensional rule plus family resemblance structure
#'     \item \code{unfr1}, \code{unfr2}
#'     \item \code{type1}, \code{type2}, \code{typeN}... Shepard, Hovland and Jenkin's elemental types
#'     \item \code{multiclass} 4 class problem built from SHJ Type II
#'     }
#' @return A list of inputs and labels
#' \itemize{
#'     \item \code{ins} Inputs for selected category structure
#'     \item \code{labels} Labels for selected category structure
#'     }
#' @export

get_test_inputs <- function(target_cats){

  test_inputs <-
    list(
      unifr = list(ins = matrix(c(
      1,  1,  1,  1,
      1,  1, -1,  1,
      1, -1,  1,  1,
     -1,  1,  1,  1,
     -1, -1, -1, -1,
     -1, -1,  1, -1,
     -1,  1, -1, -1,
      1, -1, -1, -1), ncol = 4, byrow = TRUE),
      labels = c(1, 1, 1, 1, 2, 2, 2, 2)),

      unifr1 = list(ins = matrix(c(
      1,  1,  1,  1,
      1,  1, -1,  1,
      1, -1,  1,  1,
     -1,  1,  1,  1), ncol = 4, byrow = TRUE),
      labels = c(1, 1, 1, 1)),

      unifr2 = list(ins = matrix(c(
     -1, -1, -1, -1,
     -1, -1,  1, -1,
     -1,  1, -1, -1,
      1, -1, -1, -1), ncol = 4, byrow = TRUE),
      labels = c(1, 1, 1, 1)),

      type1 = list(ins = matrix(c(
     -1, -1, -1,
     -1, -1,  1,
     -1,  1, -1,
     -1,  1,  1,
      1, -1, -1,
      1, -1,  1,
      1,  1, -1,
      1,  1,  1),  ncol = 3, byrow = TRUE),
      labels = c(1, 1, 1, 1, 2, 2, 2, 2)),

      type2 = list(ins = matrix(c(
     -1, -1, -1,
     -1, -1,  1,
     -1,  1, -1,
     -1,  1,  1,
      1, -1, -1,
      1, -1,  1,
      1,  1, -1,
      1,  1,  1),  ncol = 3, byrow = TRUE),
      labels = c(1, 1, 2, 2, 2, 2, 1, 1)),

      type3 = list(ins = matrix(c(
     -1, -1, -1,
     -1, -1,  1,
     -1,  1, -1,
     -1,  1,  1,
      1, -1, -1,
      1, -1,  1,
      1,  1, -1,
      1,  1,  1),  ncol = 3, byrow = TRUE),
      labels = c(1, 1, 2, 1, 1, 2, 2, 2)),

      type4 = list(ins = matrix(c(
     -1, -1, -1,
     -1, -1,  1,
     -1,  1, -1,
     -1,  1,  1,
      1, -1, -1,
      1, -1,  1,
      1,  1, -1,
      1,  1,  1),  ncol = 3, byrow = TRUE),
      labels = c(1, 1, 1, 2, 1, 2, 2, 2)),

      type5 = list(ins = matrix(c(
     -1, -1, -1,
     -1, -1,  1,
     -1,  1, -1,
     -1,  1,  1,
      1, -1, -1,
      1, -1,  1,
      1,  1, -1,
      1,  1,  1),  ncol = 3, byrow = TRUE),
      labels = c(2, 1, 1, 1, 1, 2, 2, 2)),

      type6 = list(ins = matrix(c(
     -1, -1, -1,
     -1, -1,  1,
     -1,  1, -1,
     -1,  1,  1,
      1, -1, -1,
      1, -1,  1,
      1,  1, -1,
      1,  1,  1),  ncol = 3, byrow = TRUE),
      labels = c(1, 2, 2, 1, 2, 1, 1, 2)),

      multiclass = list(ins = matrix(c(
     -1, -1, -1,
     -1, -1,  1,
     -1,  1, -1,
     -1,  1,  1,
      1, -1, -1,
      1, -1,  1,
      1,  1, -1,
      1,  1,  1),  ncol = 3, byrow = TRUE),
      labels = c(1, 1, 2, 2, 3, 3, 4, 4)))

  target_cat <- test_inputs[[target_cats]]

  return(target_cat)
}

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
#' @title DIVA Grid Search
#'
#' @description
#' Runs a grid search over a set of provided parameters and produces averaged
#' response probabilities
#'
#' @param param_list List of named parameters to be combined and evaluated for
#'     the DIVA model
#' @param num_inits Scalar for number of random initializations to be averaged
#'     across for the response probability calculation of each parameter combination.
#' @param input_list List of inputs and labels for the grid search
#'     \itemize{
#'         \item \code{ins} Matrix of inputs for selected category structure,
#'             R (stimuli) x C (features)
#'         \item \code{labels} Vector of labels for selected category structure
#'             indexed to the input matrx
#'         }
#' @param fit_type Character specifying the type of fit that is desired.
#'     \itemize{
#'         \item \code{'best'} for the most accurate or best fit
#'         \item \code{'crit'} for the closest match to a provided vector of
#'         response probabilities
#'         }
#' @param crit_fit_vector Vector of response probabilities for the
#'     \code{'crit'} procedure.
#'
#' @return List consisting of models, response probabilities and best model result.
#' @export

diva_grid_search <- function(param_list, num_inits, input_list, fit_type,
  crit_fit_vector = NULL) {

  # # # initialize grid search model list
  model_list <- list()

  # # # continuous? (HACK for now)
  if (length(unique(as.vector(input_list$ins))) <= 3) {cont_data <- FALSE}

  # # # basic state frame
  st <- generate_state(input = input_list$ins, categories = input_list$labels,
    colskip = 4, continuous = cont_data, make_wts = FALSE)

  # # # initialize training dataframe
  init_training_mat <- tr_init(n_feats = st$num_feats, n_cats = st$num_cats,
    feature_type = 'numeric')

  # # # make all combinations of the parameter sets into DF and get dims
  param_df      <- expand.grid(param_list)
  param_df_dims <- dim(param_df)
  param_names   <- colnames(param_df)

  # # # run models
  for (i in 1:param_df_dims[1]) {

    param_set_list <- list()

    # # # assign parameters
    for (j in 1:param_df_dims[2]) {
      st[param_names[j]] <- param_df[i, j]
    }

    # # # create list for model inits
    k_list <- list()

    # # # generate training sets and run
    for (k in 1:num_inits) {
      # # # construct training matrix
      tr <- tr_add(input_list$ins, init_training_mat,
        labels = input_list$labels, blocks = 12, ctrl = 0,
        shuffle = TRUE, reset = TRUE)

      # # # run model
      out <- slpDIVA(st, tr)

      # # # add result to list
      k_list[[k]] <- list()
      k_list[[k]]$resp_probs <- response_probs(tr, out$out, blocks = TRUE)
      k_list[[k]]$st         <- out$st
    }

    # # # assign outcome to list
    param_set_list$resp_probs <-
      avg_resp_probs(lapply(k_list, function(x) x$resp_probs))
    param_set_list$params    <- param_df[i, ]
    param_set_list$st        <- lapply(k_list, function(x) x$st)

    # # # assign everything to model list
    model_list[[i]] <- param_set_list

  }

  return(model_list)

}

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
#' @title Plot Training
#'
#' @description
#' Plots the training curve for N models.
#'
#' @param model_list List of model response probabilities across training blocks
#' @param model_names Optional vector of character values corresponding to model names
#' @return Training line plot
#' @export

plot_training <- function(model_list, model_names = NULL) {
  blocks <- length(model_list[[1]])
  n_models <- length(model_list)
  if (is.null(model_names)) {model_names <- 1:length(model_list)}

  line_cols <- rainbow(n_models)
  line_typs <- (0:(n_models - 1) %% 6) + 1

  # create blank plot
  plot.new()

  # plot first model
  plot(model_list[[1]], type = 'b', lty = line_typs[1], col = line_cols[1],
    xlim = c(1, blocks), ylim = c(0, 1), ylab = 'Accuracy',
    xlab = 'Block', xaxt = 'n', yaxt = 'n')

  # plot remaining models
  if (n_models > 1) {
    for (i in 2:length(model_list)) {
      lines(x = 1:blocks, y = model_list[[i]], lty = line_typs[i],
        col = line_cols[i], type = 'b')
    }
  }

  # fix axis
  axis(1, at = seq(1, blocks, 1), labels = TRUE)
  axis(2, at = seq(0.0, 1, .1), labels = TRUE)

  # make legend
  legend(x = 'bottomright', legend = model_names, lty = line_typs,
    col = line_cols, cex = 0.75)
}

#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
#' @title Initialize training matrix
#'
#' @description
#' Initialize a tr object
#'
#' @param n_feats Number of features (integer, > 0)
#' @param feature_type String type: numeric (default), logical, etc
#' @return An initialized dataframe with the appropriate columns
#' @export

tr_init <- function(n_feats, n_cats, feature_type = 'numeric') {

  feature_cols <- list()
  for(f in 1:n_feats) {
    feature_cols[[paste0('x', f)]] = feature_type
  }

  target_cols <- list()
  for(c in 1:n_cats) {
   target_cols[[paste0('t', c)]] = 'integer'
  }

  other_cols <- list(
    ctrl    = 'integer',
    trial   = 'integer',
    blk     = 'integer',
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



#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
#' @title Response Probabilities
#'
#' @description
#' Produces classification probability for the target class, by item or by block.
#'
#' @param tr Matrix used to train the model.
#' @param out_probs Matrix of output probabilities produced by the model.
#' @param blocks Boolean to toggle block averaged classification probabilities, default is TRUE
#' @return Vector of classification probabilities for the target class
#' @export

response_probs <- function(tr, out_probs, blocks = TRUE) {
  n_trials <- dim(tr)[1]
  all_cols <- colnames(tr)

  # find the target columns and correct class
  targets <-
    substr(all_cols, 1, 1) == 't' &
      is.finite(
        suppressWarnings(
          as.numeric(substr(all_cols, 2, 2))))
  target_cols <- apply(tr[,targets], 1, which.max)

  # get probability of correct class
  class_prob <- rep(NA, n_trials)
  for (i in 1:n_trials) {
    class_prob[i] <- out_probs[i, target_cols[i]]
  }

  # get probability averaged for each block
  if (blocks == TRUE) {
    tr_comp  <- cbind(tr, class_prob)
    n_blocks <- max(tr_comp[,'blk'])
    blk_avg <- rep(NA, n_blocks)

    # average for each block
    for (i in 1:n_blocks) {
      blk_avg[i] <-
        mean(tr_comp[tr_comp[,'blk'] == i,'class_prob'])
    }
    return(blk_avg)
  }
  return(class_prob)
}


#  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #  #
#' @title Training Matrix filler
#'
#' @description
#' Add trials to an initialized tr object.
#'
#' @param inputs Matrix of feature values for each item
#' @param tr Initialized trial object
#' @param labels Integer class labels for each input. Default NULL
#' @param blocks Integer number of repetitions. Default 1
#' @param shuffle Boolean, shuffle each block. Default FALSE
#' @param ctrl Integer control parameter, applying to all inputs. Default 2
#' \itemize{
#'    \item \code{0} Run model and train
#'    \item \code{1} Re-initialize model
#'    \item \code{2} Test model (no training)
#'    }
#' @param reset Boolean, reset state on first trial (\code{ctrl = 1}). Default FALSE
#' @return An updated dataframe
#' @export

tr_add <- function(inputs, tr,
  labels = NULL,
  blocks = 1,
  ctrl = 2,
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
    ctrl    = rep(ctrl, numtrials),
    trial   = 1:numtrials,
    blk     = sort(rep(1:blocks, numinputs)),
    example = examples
  )
#
  # add features to rows list
  for(f in 1:numfeatures){
    rows[[paste0('x', f)]] <- inputs[examples, f]
  }

  # add category labels
  num_cats <- max(labels)
  label_mat <- matrix(-1, ncol = num_cats, nrow = numtrials)

  for (i in 1:numtrials) {
    label_mat[i, labels[examples[i]]] <- 1
  }

  rows <- data.frame(rows)
  rows <- cbind(rows, label_mat)

  # reset on first trial if needed
  if (reset) {rows$ctrl[1] <- 1}

  rows <- as.matrix(rows)
  return(rbind(tr, rows))
}
