# Reference: Fitting a Stan Model with ADVI in R using CmdStanR
# Direct implementation of: https://gmoncrieff.github.io/posts/advi-vs-mcmc/

library(cmdstanr)

# Compile Stan Model
model <- cmdstan_model('ring_visibility_model.stan', compile = TRUE)

# Prepare visibility data
uvtable <- read.table('uvtable.txt', header = FALSE)
data_list <- list(
  N = nrow(uvtable),
  u = uvtable[, 1],
  v = uvtable[, 2],
  vis_re = uvtable[, 3],
  vis_im = uvtable[, 4],
  weight = uvtable[, 5]
)

# Run ADVI Variational Inference
fit_vb <- model$variational(
  data = data_list,
  adapt_engaged = FALSE,
  eta = 0.1,
  tol_rel_obj = 0.001,
  seed = 123
)

# Check total duration
print(fit_vb$time()$total)
