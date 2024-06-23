open Torch

let construct_q_i theta i alpha =
  let mean = theta |> Tensor.get ~idx:[i] in
  let std = Tensor.scalar (sqrt alpha) in
  Normal.Normal (mean, std)

let sample_from_q_i q_i =
  Normal.sample q_i ~shape:[1]

(* Compute q(θ'|θ) and q(θ|θ') *)
let compute_q theta theta_prime d construct_q alpha =
  let compute_prod old_theta new_theta =
    List.init d (fun i ->
      let q_i = construct_q old_theta i alpha in
      let prob = Normal.log_prob q_i (Tensor.get new_theta ~idx:[i])
      in Tensor.exp prob
    )
    |> List.fold_left Tensor.mul (Tensor.ones [])
  in
  let q_forward = compute_prod theta theta_prime in
  let q_backward = compute_prod theta_prime theta in
  (q_forward, q_backward)

let compute_acceptance_prob q_forward q_backward theta theta_prime =
  let ratio = Tensor.(div q_backward q_forward) in
  Tensor.min ratio (Tensor.ones_like ratio)

let sample_dula_dmala alpha d num_samples =
  let rec sampling_loop theta samples count =
    if count >= num_samples then
      List.rev samples
    else
      let theta_prime = Tensor.zeros_like theta in
      for i = 0 to d - 1 do
        let q_i = construct_q_i theta i alpha in
        let theta_i_prime = sample_from_q_i q_i in
        Tensor.set_ theta_prime ~idx:[i] ~src:theta_i_prime;
      done;
      
      let q_forward, q_backward = compute_q theta theta_prime d (construct_q_i) alpha in
      let acceptance_prob = compute_acceptance_prob q_forward q_backward theta theta_prime in
      
      let new_theta = 
        if Tensor.(to_float0_exn (rand [1])) < Tensor.to_float0_exn acceptance_prob then
          theta_prime
        else
          theta
      in
      
      sampling_loop new_theta (new_theta :: samples) (count + 1)
  in
  
  let initial_theta = Tensor.randn [d] in
  sampling_loop initial_theta [] 0