open Torch

module PCD = struct
  type t = {
    mutable theta: Tensor.t;
    alpha: float;
    t: int;
    buffer_size: int;
    mutable buffer: Tensor.t array;
    e_theta: Tensor.t -> Tensor.t;
    p_data: unit -> Tensor.t;
  }

  let create theta alpha t buffer_size e_theta p_data =
    let buffer = Array.init buffer_size (fun _ -> Tensor.uniform ~low:0. ~high:1. []) in
    { theta; alpha; t; buffer_size; buffer; e_theta; p_data }

  let sample_buffer t =
    if Random.float 1.0 < 0.95 then
      t.buffer.(Random.int t.buffer_size)
    else
      Tensor.uniform ~low:0. ~high:1. []

  let langevin_dynamics t x' =
    let rec loop x' i =
      if i = t.t then x'
      else
        let epsilon = Tensor.randn_like x' ~std:t.alpha in
        let grad = Tensor.grad t.e_theta x' in
        let x_next = Tensor.(x' - (grad * f (t.alpha /. 2.)) + epsilon) in
        loop x_next (i + 1)
    in
    loop x' 0

  let update t =
    let x = t.p_data () in
    let x' = sample_buffer t in
    let x'_updated = langevin_dynamics t x' in
    let grad_x = Tensor.grad t.e_theta x in
    let grad_x' = Tensor.grad t.e_theta x'_updated in
    let delta_theta = Tensor.(grad_x - grad_x') in
    t.theta <- Tensor.adam t.theta ~grad:delta_theta ~alpha:0.001 ~beta1:0.9 ~beta2:0.999;
    t.buffer.(Random.int t.buffer_size) <- x'_updated

  let train t max_iterations =
    let rec loop i =
      if i = max_iterations then t.theta
      else begin
        update t;
        loop (i + 1)
      end
    in
    loop 0
end