open Torch

module SpectralNorm = struct
  let apply w ~iteration =
    let w_shape = Tensor.shape w in
    let w = Tensor.view w ~size:[-1; List.hd (List.rev w_shape)] in
    let u = Var_store.new_var (Var_store.create ~name:"u" ()) ~shape:[1; List.hd (List.rev w_shape)] ~init:Var_store.Init.normal in
    let rec power_iteration u_hat iteration =
      if iteration = 0 then
        u_hat
      else
        let v_ = Tensor.matmul u_hat (Tensor.transpose w) in
        let v_hat = Tensor.(l2_normalize v_) in
        let u_ = Tensor.matmul v_hat w in
        let u_hat = Tensor.(l2_normalize u_) in
        power_iteration u_hat (iteration - 1)
    in
    let u_hat = power_iteration u (iteration - 1) in
    let v_hat = Tensor.(l2_normalize (Tensor.matmul u_hat (Tensor.transpose w))) in
    let sigma = Tensor.matmul (Tensor.matmul v_hat w) (Tensor.transpose u_hat) in
    let w_norm = Tensor.(w / sigma) in
    Tensor.view w_norm ~size:w_shape

  let snconv2d ~x ~channels ?(kernel=4) ?(stride=2) ?(use_bias=true) ?(kernel_initializer=Var_store.Init.normal ~mean:0.0 ~std:0.02) ?scope () =
    let var_store = Var_store.create ?name:scope () in
    let w = Var_store.new_var var_store ~shape:[kernel; kernel; Tensor.shape x |> List.hd |> List.hd; channels] ~init:kernel_initializer in
    let bias = Var_store.new_var var_store ~shape:[channels] ~init:Var_store.Init.constant in
    let x = Tensor.conv2d x (apply w ~iteration:1) ~stride:(stride, stride) ~padding:Valid in
    let x = if use_bias then Tensor.add x (Tensor.view bias ~size:[1; channels; 1; 1]) else x in
    x
end