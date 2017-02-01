\name{slpDIVA}
\alias{slpDIVA}
\title{DIVA category learning model}
\description{
  DIVergent Autoencoder (Kurtz, 2007; 2015) artificial neural network category learning model.
}
\usage{
slpDIVA(st, tr, xtdo = FALSE)
}
\arguments{
  \item{st}{List of model parameters.}
  \item{tr}{R-by-C matrix of training items.}
  \item{xtdo}{Boolean specifying whether to write extended information to the console}
}
\details{
  This documentation provides a bare-bones guide to using \code{slpDIVA} to model category learning in the context of the \code{catlearn} package. This function follows the design pattern outlined in Wills et al. (2016).

  Adapted from the \code{slpALCOVE} documentation: This function works as a stateful list processor. Specifically, it takes a matrix as an argument, where each row is one trial for the network, and the columns specify the input representation, teaching signals, and other control signals. It returns a matrix where each row is a trial, and the columns are the response probabilities at the output units. It also returns the final state of the network (attention and connection weights), hence its description as a 'stateful' list processor.

  Argument \code{st} must be a list containing the following items:

  \code{num_feats} - Number of features for the problem.

  \code{num_cats} - Number of categories for the problem.

  \code{colskip} - Skip the first N columns of the tr array, where N = colskip. colskip should be set to the number of optional columns you have added to matrix tr, PLUS ONE. So, if you have added no optional columns, \code{colskip = 1}. This is because the first (non-optional) column contains the control values, below.

  \code{in_wts} - A matrix of weights of dimensions \code{num_feats + 1} x \code{num_hids}. Can be set to \code{NULL} when the first line of the \code{tr} matrix includes control code 1, \code{ctrl = 1}.  

  \code{out_wts} - A matrix of weights of dimensions \code{num_hids + 1} x \code{num_cats}. Can be set to \code{NULL} when the first line of the \code{tr} matrix includes control code 1, \code{ctrl = 1}.   

  \code{continuous} - A boolean value to indicate if the inputs are continuous or dichotomous. Set \code{Continuous = TRUE} when the inputs are continuous. 

  \code{wts_range} - A scalar value for the range of the generated weights.

  \code{wts_center} - A scalar value for the center of the weights.
    
  \code{num_hids} - A scalar value for the number of hidden units. A rough rule of thumb for this hyperparameter is \code{num_feats - 1}. 

  \code{learning_rate} - Learning rate for weight updates through backpropagation

  \code{beta_val} - Scalar value for the Beta parameter. \code{beta_val} controls the degree of feature focusing (not unlike attention) that the model uses to make classification decisions (Conaway & Kurtz, 2014) 
    
  \code{model_seed} - 






}
\value{
  Returns a list containing two components: (1) matrix of response probabilities for each category on each trial, (2) an \code{st} list object that contains the model's final state. A weight initialization history is also available when the extended output parameter is set \code{xtdo = TRUE} in the \code{slpDIVA} call. 

}

\authors{
  Garrett Honke
  Nolan B. Conaway
}