open Torch

let top_k_joint_sampling p_lm n k energy_function =
  let samples = Tensor.empty [n] in
  for i = 0 to n - 1 do
    let sample = p_lm.sample ~k () in
    Tensor.set samples [|i|] sample
  done;

  let energies = Tensor.map energy_function samples in

  let neg_energies = Tensor.neg energies in
  let probs = Tensor.softmax neg_energies ~dim:0 in

  let indices = Tensor.multinomial probs ~num_samples:1 ~replacement:true in
  Tensor.index_select samples ~dim:0 ~index:indices