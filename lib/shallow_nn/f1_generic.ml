open Torch

module EBM_Trainer = struct
  let sample_from_mu0 m d =
    Tensor.randn [m; d + 2]

  let compute_gradient_estimates g samples =
    let w, theta = Tensor.split samples ~split_size:1 ~dim:1 in
    let g_output = g samples in
    let grad_w, grad_theta = Tensor.grad g_output [w; theta] in
    grad_w, grad_theta

  let update_samples samples grad_w grad_theta stepsize =
    let w, theta = Tensor.split samples ~split_size:1 ~dim:1 in
    let w_new = Tensor.(w - (grad_w * f stepsize)) in
    let theta_new = Tensor.(theta - (grad_theta * f stepsize)) in
    Tensor.cat [w_new; theta_new] ~dim:1

  let compute_energy phi samples =
    let m = Tensor.size samples ~dim:0 in
    Tensor.(mean (phi samples) / f (float_of_int m))

  let train m stepsize t_max d g phi =
    let samples = sample_from_mu0 m (d + 2) in
    
    let rec train_loop t samples =
      if t = t_max then samples
      else
        let grad_w, grad_theta = compute_gradient_estimates g samples in
        let updated_samples = update_samples samples grad_w grad_theta stepsize in
        train_loop (t + 1) updated_samples
    in
    
    let final_samples = train_loop 0 samples in
    compute_energy phi final_samples
end