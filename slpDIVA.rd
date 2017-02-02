\name{slpDIVA}
\alias{slpDIVA}
\title{DIVA category learning model}
\description{
  DIVergent Autoencoder (Kurtz, 2007; 2015) artificial neural network category learning model
}
\usage{
slpDIVA(st, tr, xtdo = FALSE)
}
\arguments{
  \item{st}{List of model parameters}
  \item{tr}{R-by-C matrix of training items}
  \item{xtdo}{Boolean specifying whether to write extended information to the console}
}
\details{
  This documentation provides a bare-bones guide to using \code{slpDIVA} to model category learning in the context of the \code{catlearn} package. This function follows the design pattern outlined in Wills et al. (2016), thus, more comprehensive information can be found there. 

  Adapted from the \code{slpALCOVE} documentation: This function works as a stateful list processor. Specifically, it takes a matrix as an argument, where each row is one trial for the network, and the columns specify the input representation, teaching signals, and other control signals. It returns a matrix where each row is a trial, and the columns are the response probabilities for each category. It also returns the final state of the network (connection weights and other parameters), hence its description as a 'stateful' list processor.

  Argument \code{st} must be a list containing the following items:

  \code{num_feats} - Number of features for the problem.

  \code{num_cats} - Number of categories for the problem.

  \code{colskip} - Skip the first N columns of the tr array, where \code{N = colskip}. \code{colskip} should be set to the number of optional columns you have added to matrix \code{tr}, PLUS ONE. So, if you have added no optional columns, \code{colskip = 1}. This is because the first (non-optional) column contains the control values, below.

  \code{in_wts} - A matrix of weights of dimensions \code{num_feats + 1} x \code{num_hids}. Can be set to \code{NULL} when the first line of the \code{tr} matrix includes control code 1, \code{ctrl = 1}.  

  \code{out_wts} - A matrix of weights of dimensions \code{num_hids + 1} x \code{num_cats}. Can be set to \code{NULL} when the first line of the \code{tr} matrix includes control code 1, \code{ctrl = 1}.   

  \code{continuous} - A boolean value to indicate if the inputs are continuous or binary. Set \code{Continuous = TRUE} when the inputs are continuous. 

  \code{wts_range} - A scalar value for the range of the generated weights.

  \code{wts_center} - A scalar value for the center of the weights. This is commonly fixed at 0.
    
  \code{num_hids} - A scalar value for the number of hidden units. A rough rule of thumb for this hyperparameter is \code{num_feats - 1}. 

  \code{learning_rate} - Learning rate for weight updates through backpropagation

  \code{beta_val} - Scalar value for the Beta parameter. \code{beta_val} controls the degree of feature focusing (not unlike attention) that the model uses to make classification decisions (see: Conaway & Kurtz, 2014; Kurtz, 2015) 
    
  \code{model_seed} - Scalar value used to set the random seed for weight generation. 

  Argument \code{tr} must be a matrix, where each row is one trial presented to the network. Trials are always presented in the order specified. The columns must be as described below, in the order described below:

  \code{ctrl} - column of control codes. Available codes are: 0 = normal learning trial, 1 = reset network (i.e. initialize a new set of weights following the \code{st} parameters), 2 = Freeze learning. Control codes are actioned before the trial is processed.

  \code{opt1, opt2, ...} - optional columns, which may have any names you wish, and you may have as many as you like, but they must be placed after the \code{ctrl} column, and before the remaining columns (see below). These optional columns are ignored by this function, but you may wish to use them for readability. For example, you might include columns for block number, trial number, and stimulus ID number. The argument \code{colskip} (see above) must be set to the number of optional columns plus 1.

  \code{x1, x2, ...} - input to the model, there must be one column for each input unit. Each row is one trial. Dichotomous inputs should be in the format \code{-1, 1}. Continuous inputs should be scaled to the range of \code{-1, 1}. As the model's learning objective is to accurately reconstruct the inputs, the input to the model is also the teaching signal. For testing under conditions of missing information, input features can be set to 0 to negate the contribution of the feature(s) for the classification decision of that trial. 

}
\value{
  Returns a list containing two components: (1) matrix of response probabilities for each category on each trial, (2) an \code{st} list object that contains the model's final state. A weight initialization history is also available when the extended output parameter is set \code{xtdo = TRUE} in the \code{slpDIVA} call. 

}
\author{
  Garrett Honke

  Nolan B. Conaway
}
\references{
  Conaway, N. B., & Kurtz, K. J. (2014). Now you know it, now you don't: Asking the right question about category knowledge. In P. Bello, M. Guarini, M. McShane, & B. Scassellati (Eds.), \emph{Proceedings of the Thirty-Sixth Annual Conference of the Cognitive Science Society} (pp. 2062-2067). Austin, TX: Cognitive Science Society.

  Kurtz, K.J. (2007). The divergent autoencoder (DIVA) model of category learning. \emph{Psychonomic Bulletin & Review}.

  Kurtz, K. J. (2015). Human Category Learning: Toward a Broader Explanatory Account. \emph{Psychology of Learning and Motivation}.

  Wills, A.J., O'Connell, G., Edmunds, C.E.R., & Inkster, A.B.(2016). Progress in modeling through distributed collaboration: Concepts, tools, and category-learning examples. \emph{The Psychology of Learning and Motivation}.
}
