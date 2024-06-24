open Torch

module LSDModelComparison = struct
  type model = Tensor.t -> Tensor.t
  type critic = Tensor.t -> Tensor.t

  let split_data x =
    let n = Tensor.shape x |> List.hd in
    let train_size = n * 6 / 10 in
    let val_size = n * 2 / 10 in
    let x_train = Tensor.narrow x ~dim:0 ~start:0 ~length:train_size in
    let x_val = Tensor.narrow x ~dim:0 ~start:train_size ~length:val_size in
    let x_test = Tensor.narrow x ~dim:0 ~start:(train_size + val_size) ~length:(n - train_size - val_size) in
    (x_train, x_val, x_test)

  let lsde f_phi x q =
    let q_x = q x in
    let f_phi_x = f_phi x in
    Tensor.mse_loss q_x f_phi_x

  let regularization lambda f_phi =
    let params = Tensor.grad_params f_phi in
    Tensor.mul_scalar (Tensor.norm params) lambda

  let optimize_critic f_phi x_train q x_val lambda max_iterations lr =
    let optimizer = Optimizer.adam (Tensor.grad_params f_phi) ~lr in
    let rec loop i best_phi best_val_loss =
      if i >= max_iterations then best_phi
      else
        let loss = Tensor.(sub (lsde f_phi x_train q) (regularization lambda f_phi)) in
        Optimizer.zero_grad optimizer;
        Tensor.backward loss;
        Optimizer.step optimizer;
        let val_loss = lsde f_phi x_val q in
        if Tensor.to_float0_exn val_loss < Tensor.to_float0_exn best_val_loss then
          loop (i + 1) f_phi val_loss
        else
          loop (i + 1) best_phi best_val_loss
    in
    loop 0 f_phi (Tensor.of_float Float.infinity)

  let lsd f_phi x q =
    Tensor.mse_loss (q x) (f_phi x)

  let compare_models f_phi models x lambda =
    let x_train, x_val, x_test = split_data x in
    let results = Array.map (fun q ->
      let optimized_f_phi = optimize_critic f_phi x_train q x_val lambda 1000 0.001 in
      let s_i = lsd optimized_f_phi x_test q in
      (q, Tensor.to_float0_exn s_i)
    ) models in
    Array.sort (fun (_, s1) (_, s2) -> compare s1 s2) results;
    Array.map fst results
end