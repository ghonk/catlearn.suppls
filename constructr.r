#' Initialize a tr object.
#' 
#' @param nf Number of features (integer, > 0).
#' @param feature_type String type: numeric (default), logical, etc.
#' @return An initialized df with the appropriate columns.
tr_init <- function(nf, feature_type = 'numeric') {

  feature_cols <- list()
  for(f in 1:nf){
    feature_cols[[ paste0('f', f)]] = feature_type
  }

  other_cols <- list(
    ctrl = 'integer', 
    trial = 'integer',
    block = 'integer',
    example = 'integer',
    category= 'integer'
  )

  all_cols <- append(other_cols, feature_cols) 

  # create empty df with column types specified by all_cols
  empty_tr <- data.frame()
  for (col in names(all_cols)) {
      empty_tr[[col]] <- vector(mode = all_cols[[col]], length=0)
  }

  return(empty_tr)
}

#' Add trials to an initialized tr object
#' 
#' @param inputs Matrix of feature values for each item.
#' @param tr Initialized trial object.
#' @param labels Integer class labels for each input. Default NULL.
#' @param blocks Integer number of repetitions. Default 1. 
#' @param shuffle Boolean, shuffle each block. Default FALSE.
#' @param ctrl Integer control parameter, applying to all inputs. Default 2.
#' @param reset Boolean, reset state on first trial (ctrl=1). Default FALSE.
#' @return An updated df.
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
    ctrl = rep(ctrl, numtrials),
    trial = 1:numtrials,
    block = sort(rep(1:blocks, numinputs)),
    example = examples,
    category = labels[examples]
  )

  # add features to rows list
  for(f in 1:numfeatures){
    rows[[paste0('f', f)]] = inputs[examples, f]
  }
  
  # reset on first trial if needed
  if (reset) {rows$ctrl[1] = 1}
  
  return(rbind(tr, rows))
}
