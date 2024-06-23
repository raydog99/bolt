open Torch

let energy_function theta x =
  Tensor.(sum (pow x (scalar 2.)) |> mul theta)

let gradient_energy theta x =
  let x = Tensor.set_requires_grad x true in
  let energy = energy_function theta x in
  Tensor.backward energy;
  Tensor.grad x

let compute_proposal_distribution grad_energy =
  let exponent = Tensor.(mul (scalar 2.) (sub grad_energy (scalar 1.))) in
  let numerator = Tensor.exp exponent in
  let denominator = Tensor.sum numerator ~dim:[0] in
  Tensor.div numerator denominator

let sample_from_proposal n_tilde s =
  Tensor.multinomial n_tilde ~num_samples:s ~replacement:true

let compute_j_rm theta x x_samples n_tilde =
  let d = Tensor.size x 0 |> float_of_int in
  let s = Tensor.size x_samples 0 |> float_of_int in
  let energies = Tensor.stack (List.map (energy_function theta) (Tensor.to_list x_samples ~dim:0)) in
  let m_x = Tensor.(exp (neg energies)) in
  let numerator = Tensor.(pow (sub (energy_function theta x) energies) (scalar 2.)) in
  let denominator = n_tilde in
  Tensor.(mean (div (mul m_x numerator) denominator) |> mul (scalar (d /. s)))

let rmwggis dataset theta num_samples num_iterations learning_rate =
  let optimizer = Optimizer.adam [theta] ~lr:learning_rate in
  
  for _ = 1 to num_iterations do
    dataset |> Tensor.to_list ~dim:0 |> List.iter (fun x ->
      let grad_energy = gradient_energy theta x in
      let n_tilde = compute_proposal_distribution grad_energy in
      let x_samples = sample_from_proposal n_tilde num_samples in
      let j_rm = compute_j_rm theta x x_samples n_tilde in
      
      Optimizer.zero_grad optimizer;
      Tensor.backward j_rm;
      Optimizer.step optimizer
    )
  done;
  theta