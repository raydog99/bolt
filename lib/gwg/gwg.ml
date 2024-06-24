open Torch

module Gibbs_with_Gradients = struct
  let categorical_softmax d =
    let softmax = Tensor.softmax d ~dim:(-1) ~dtype:(T Float) in
    Tensor.multinomial softmax ~num_samples:1 ~replacement:true

  let compute_q x d =
    let d_div_2 = Tensor.(d / f 2.) in
    categorical_softmax d_div_2

  let flipdim x i =
    let dims = Tensor.shape x in
    let flipped = Tensor.flip x ~dims:[i] in
    Tensor.set x ~index:[i] (Tensor.get flipped ~index:[i]);
    x

  let accept_probability f x x' q_x q_x' =
    let exp_term = Tensor.(exp (f x' - f x)) in
    let ratio = Tensor.(q_x' / q_x) in
    Tensor.(min (exp_term * ratio) (ones []))

  let gibbs_step f x d =
    let q_x = compute_q x d in
    let i = Tensor.item q_x |> Int.of_float in
    let x' = flipdim x i in
    let q_x' = compute_q x' d in
    let accept_prob = accept_probability f x x' q_x q_x' in
    if Tensor.item accept_prob > Random.float 1.0 then x' else x

  let sample f d num_iterations =
    let x = Tensor.rand [Tensor.shape d |> List.hd] in
    let rec loop x n =
      if n = 0 then x
      else loop (gibbs_step f x d) (n - 1)
    in
    loop x num_iterations
end