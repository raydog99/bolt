open Torch

let sample_u_x n =
  Tensor.rand [n]

let sample_q1 n theta =
  let mean = Tensor.repeat theta [n] in
  let std = Tensor.ones [n] in
  Tensor.normal ~mean ~std

let sample_q2 n theta =
  let mean = Tensor.repeat theta [n] in
  let std = Tensor.full [n] 0.1 in
  Tensor.normal ~mean ~std

let estimate_l_theta x x_prime theta =
  let log_q1 = Tensor.sum (Tensor.log_normal x ~mean:theta ~std:(Tensor.ones_like theta)) in
  let log_q2 = Tensor.sum (Tensor.log_normal x_prime ~mean:theta ~std:(Tensor.full_like theta 0.1)) in
  Tensor.sub log_q1 log_q2

let update_eta eta x =
  let alpha = 0.9 in (* Decay factor *)
  Tensor.(add (mul eta (scalar alpha)) (mul x (scalar (1. -. alpha))))

let accept_reject_samples x eta =
  let u = Tensor.rand (Tensor.shape x) in
  Tensor.lt u (Tensor.sigmoid (Tensor.sub x eta))

let lsb learning_rate pi theta_0 k n =
  let rec lsb_loop k theta x_hat eta =
    if k = 0 then theta
    else
      let x = sample_q1 n theta in
      let x_prime = sample_q2 n theta in
      
      let theta = Tensor.set_requires_grad theta true in
      
      let l_theta_hat = estimate_l_theta x x_prime theta in
      
      Tensor.backward l_theta_hat;
      let grad_l_theta = Option.get (Tensor.grad theta) in
      let theta = Tensor.(sub theta (mul (scalar (learning_rate /. float_of_int n)) grad_l_theta)) in
      
      let eta = update_eta eta x in
      
      let accepted = accept_reject_samples x eta in
      
      let x_hat = Tensor.where accepted x x_hat in
      
      let theta = Tensor.detach theta in
      
      lsb_loop (k - 1) theta x_hat eta
  in
  
  let initial_x_hat = sample_u_x n in
  let initial_eta = Tensor.zeros [n] in
  lsb_loop k theta_0 initial_x_hat initial_eta