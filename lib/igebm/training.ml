open Torch

let training p_d step_size k_steps =
  let replay_buffer = ref [] in
  let theta = Tensor.randn [1; 100] in
  
  let sample_p_d () = 
    Tensor.randn [1; 784]
  in
  
  let sample_uniform () =
    Tensor.uniform ~from:(-1.) ~to_:1. [1; 784]
  in
  
  let energy_function x =
    Tensor.sum (Tensor.mul x x)
  in
  
  let langevin_dynamics x_init =
    let rec loop k x =
      if k >= k_steps then x
      else
        let grad = Tensor.grad_of_fn1 energy_function x in
        let noise = Tensor.randn_like x in
        let x_next = Tensor.(x - (grad * step_size) + (noise * Float.(sqrt (2. * step_size)))) in
        loop (k + 1) x_next
    in
    loop 0 x_init
  in
  
  let optimize_objective x_pos x_neg =
    let n = Float.of_int (List.length !replay_buffer) in
    let l2_term = Tensor.(
      (energy_function x_pos + energy_function x_neg) * Float.(of_float 0.5 / n)
    ) in
    let ml_term = Tensor.(energy_function x_pos - energy_function x_neg) in
    Tensor.(l2_term + ml_term)
  in
  
  let rec training_loop () =
    let x_pos = sample_p_d () in
    let x_init = 
      if Random.float 1.0 < 0.95 && not (List.is_empty !replay_buffer) then
        List.hd !replay_buffer
      else
        sample_uniform ()
    in
    
    let x_k = langevin_dynamics x_init in
    let x_neg = Tensor.detach x_k in
    
    let loss = optimize_objective x_pos x_neg in
    
    Tensor.backward loss;
    Tensor.Adam.step theta;
    
    replay_buffer := x_neg :: !replay_buffer;
    
    if (* convergence check *) false
    then ()
    else training_loop ()
  in
  
  training_loop ()