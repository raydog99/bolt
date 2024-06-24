open Torch

module PseudoSphericalCD = struct
  type t = {
    energy_function: Tensor.t -> Tensor.t;
    gamma: float;
    learning_rate: float;
  }

  let create energy_function gamma learning_rate =
    { energy_function; gamma; learning_rate }

  let sample_langevin model x num_steps step_size =
    let rec loop x step =
      if step = 0 then x
      else
        let grad = Tensor.grad (model.energy_function x) [x] in
        let noise = Tensor.randn_like x in
        let x' = Tensor.(x - (grad * scalar step_size) + (noise * scalar (sqrt (2. *. step_size)))) in
        loop x' (step - 1)
    in
    loop x num_langevin_steps

  let update model x_pos x_neg =
    let n = Tensor.shape x_pos |> List.hd in
    
    let exp_neg_gamma_e_pos = Tensor.(exp (neg (scalar model.gamma * model.energy_function x_pos))) in
    let sum_exp_neg_gamma_e_pos = Tensor.sum exp_neg_gamma_e_pos in
    let log_mean_exp_neg_gamma_e_pos = Tensor.((scalar (1. /. model.gamma)) * log (sum_exp_neg_gamma_e_pos / scalar (float_of_int n))) in
    
    let exp_neg_gamma_e_neg = Tensor.(exp (neg (scalar model.gamma * model.energy_function x_neg))) in
    let sum_exp_neg_gamma_e_neg = Tensor.sum exp_neg_gamma_e_neg in
    
    let grad_e_neg = Tensor.grad (model.energy_function x_neg) [x_neg] in
    let weighted_grad_e_neg = Tensor.(exp_neg_gamma_e_neg * grad_e_neg) in
    let sum_weighted_grad_e_neg = Tensor.sum weighted_grad_e_neg in
    
    let grad_log_mean_exp_neg_gamma_e_pos = Tensor.grad log_mean_exp_neg_gamma_e_pos [x_pos] in
    
    let gradient = Tensor.(neg grad_log_mean_exp_neg_gamma_e_pos - (sum_weighted_grad_e_neg / sum_exp_neg_gamma_e_neg)) in
    
    { model with energy_function = 
        (fun x -> Tensor.(model.energy_function x - (gradient * scalar model.learning_rate))) }

  let train model data_sampler num_iterations batch_size =
    let rec loop model iter =
      if iter = 0 then model
      else
        let x_pos = data_sampler batch_size in
        let x_neg = sample_langevin model (Tensor.randn_like x_pos) num_langevin_steps langevin_step_size in
        let model' = update model x_pos x_neg in
        loop model' (iter - 1)
    in
    loop model num_iterations
end