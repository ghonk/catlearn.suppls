
source('catlearn_suppls.r')
source('slpAE.r')

source('C:/Users/garre/Dropbox/aa projects/catlearn_top/diva-branch/R/slpDIVA.R')


test_inputs <- 
  list(
    unifr = matrix(c(  
    1,  1,  1,  1,
    1,  1, -1,  1,
    1, -1,  1,  1,
   -1,  1,  1,  1,
   -1, -1, -1, -1,
   -1, -1,  1, -1,
   -1,  1, -1, -1,
    1, -1, -1, -1), ncol = 4, byrow = TRUE),
    unifr1 = matrix(c(  
    1,  1,  1,  1,
    1,  1, -1,  1,
    1, -1,  1,  1,
   -1,  1,  1,  1), ncol = 4, byrow = TRUE),
   
    unifr2 = matrix(c(  
   -1, -1, -1, -1,
   -1, -1,  1, -1,
   -1,  1, -1, -1,
    1, -1, -1, -1), ncol = 4, byrow = TRUE),
    type2 = matrix(c(
    1,  1, -1, 
    1,  1,  1, 
   -1, -1, -1,
   -1, -1,  1,
    1, -1,  1,
    1, -1, -1,
   -1,  1,  1,
   -1,  1, -1), ncol = 3, byrow = TRUE))

ins <- test_inputs[[2]]
nfeats <- dim(ins)[2]
    
cla_st <- list(learning_rate = 0.15, num_feats = nfeats, num_hids = 3, num_cats = 1, 
  beta_val = 0, continuous = FALSE, in_wts = NULL, out_wts = NULL, wts_range = 1,
  wts_center = 0, colskip = 4)

# # # construct classify training matrix
cla_tr <- tr_init(nfeats, 1)
cla_tr <- tr_add(inputs = ins, tr = cla_tr, 
  labels = c(1,1,1,1), blocks = 100, ctrl = 0, shuffle = TRUE, 
  reset = TRUE)

(cla_model <- slpAE(cla_st, cla_tr))

cbind(cla_tr, cla_model$out)

plot(cla_tr[,'trial'], cla_model$out, type = "n", 
  main = 'Reconstruction Error', ylab = 'Error', xlab = 'Trial') 
lines(cla_tr[,'trial'], cla_model$out, type = 'l')
  