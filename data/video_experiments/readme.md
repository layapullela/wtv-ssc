For WTV-SSC, the lambda_2_row_ref is related to lambda_2 as follows: 
lambda_2 = f * lambda_2_row_ref, f = ||D_raw||_2 / ||D_raw's row||

The reason for this formulation is because when we normalize D_raw by spectral norm, this actually rescales lambda_2. To keep the grid search consistent along each axis, we search for lambda_2_row_ref, and then rescale lambda_2 accordingly.