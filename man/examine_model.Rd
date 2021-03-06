% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/catlearn_suppls.r
\name{examine_model}
\alias{examine_model}
\title{Examine model}
\usage{
examine_model(st, tr, model)
}
\arguments{
\item{st}{List of initial state of the model}

\item{tr}{Matrix of training or test examples}

\item{model}{String indicating which model is to be examined
\itemize{
     \item \code{slpDIVA} DIVA model
     \item \code{slpALCOVE} ALCOVE model
     \item \code{slpDIVAdev} Developmental DIVA (only tested option)
     }}
}
\value{
\code{out} List of lists for each trial containing trial-by-trial
model information including:
 \itemize{
     \item \code{init_wts} List of weights
         \itemize{
             \item \code{in_wts} Matrix of input to hidden weight (including bias)
             \item \code{out_wts} Array of hidden to output weights (including bias)
         }
     \item \code{inputs} Matrix of complete input information for trial
     \item \code{hidden_act} Matrix of hidden unit activation for trial
     \item \code{result} List that contains the model's post-trial state
          \itemize{
              \item \code{st} List of the model's end-trial state (see \code{?slpDIVA})
              \item \code{out} Vector of respond probabilities
          }
     }
}
\description{
Function to examine the parameters, state and classification probabilites
of a model across a training or test matrix.
}
