
# # # load model functions
source('utils.r')
source('constructr.r')

# # # load some data: shj types 1-6 or 4-class prob (7)
cases <- demo_cats(2)
  inputs <- cases$inputs
  labels <- cases$labels

# make a single presentation order with a test set after training
tr <- tr_init( dim(inputs)[1] )
tr <- tr_add(inputs, tr, labels = labels, blocks = 2, 
  ctrl = 0, shuffle = TRUE, reset = TRUE)
tr <- tr_add(inputs, tr, ctrl = 2)
print(tr)

# And you can just add another model's presentation order on there!
tr <- tr_add(inputs, tr, labels = labels, blocks = 2, 
  ctrl = 0, shuffle = TRUE, reset = TRUE)
tr <- tr_add(inputs, tr, ctrl = 2)
print(tr)


