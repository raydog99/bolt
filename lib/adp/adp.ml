open Torch

module Adp = struct
  let adversarial_purification x s_theta g_phi sigma S T lambda delta tau =
    let x_T = Tensor.zeros_like x in
    let x_s = Array.make S (Tensor.zeros_like x) in
    
    for s = 0 to S - 1 do
      let epsilon = Tensor.(randn_like x * f sigma) in
      let x_0 = Tensor.(x + epsilon) in
      
      let rec purification_run t x_t =
        if t = T then x_t
        else
          let x_prime = Tensor.(x_t + (f delta * s_theta x_t)) in
          let s_x_t = s_theta x_t in
          let s_x_prime = s_theta x_prime in
          let alpha = Tensor.(
            (f lambda * f delta) * 
            (f 1. - ((dot s_x_t s_x_prime) / (dot s_x_t s_x_t)))
          ) in
          let x_t_next = Tensor.(x_t + (alpha * s_theta x_t)) in
          
          if Tensor.norm x_t_next < tau then
            x_t_next
          else
            purification_run (t + 1) x_t_next
      in
      
      x_s.(s) <- purification_run 0 x_0
    done;
    
    let x_T_sum = Array.fold_left Tensor.add (Tensor.zeros_like x) x_s in
    Tensor.(x_T_sum / f (float_of_int S))
  
  let argmax_k tensor =
    let _, indices = Tensor.max tensor ~dim:(-1) ~keepdim:false in
    Tensor.squeeze indices ~dim:[-1]

  let run x s_theta g_phi sigma S T lambda delta tau =
    let x_T = adversarial_purification x s_theta g_phi sigma S T lambda delta tau in
    argmax_k (g_phi x_T)
end