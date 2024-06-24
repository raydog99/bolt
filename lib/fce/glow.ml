open Torch

module Actnorm = struct
  type t = {
    scale : Tensor.t ref;
    bias : Tensor.t ref;
    mutable initialized : Tensor.t;
  }

  let create ?(param_dim=(1, 2)) () =
    let scale = Tensor.ones param_dim |> Tensor.to_var in
    let bias = Tensor.zeros param_dim |> Tensor.to_var in
    let initialized = Tensor.zeros [1] |> Tensor.to_device Device.CPU in
    { scale = ref scale; bias = ref bias; initialized }

  let forward t x =
    let open Tensor in
    if Scalar.to_int (to_scalar t.initialized) = 0 then (
      let bias_val = x |> transpose ~dim0:0 ~dim1:1 |> flatten 1 |> mean1 ~dim:[1] ~keepdim:false in
      let scale_val = x |> transpose ~dim0:0 ~dim1:1 |> flatten 1 |> std ~dim:[1] ~keepdim:false false + f 1e-6 in
      t.bias := Tensor.view_as (Tensor.copy_ (Tensor.squeeze !t.bias) bias_val) !t.scale;
      t.scale := Tensor.view_as (Tensor.copy_ (Tensor.squeeze !t.scale) scale_val) !t.bias;
      t.initialized <- Tensor.add_scalar_ t.initialized 1.;
    );
    let z = div (sub x !t.bias) !t.scale in
    let logdet = neg (Tensor.sum (abs !t.scale |> log)) in
    z, logdet

  let inverse t z =
    let open Tensor in
    let x = add (mul z !t.scale) !t.bias in
    let logdet = sum (abs !t.scale |> log) in
    x, logdet
end

module Invertible1x1Conv = struct
  type t = {
    w : Tensor.t ref;
  }

  let create ?(dim=2) () =
    let w = Tensor.randn [dim; dim] |> Tensor.linalg_qr in
    let w = Tensor.select w 0 in
    { w = ref (Tensor.to_var w) }

  let forward t x =
    let open Tensor in
    let logdet = select (slogdet !t.w) (-1) in
    let y = matmul x (transpose !t.w) in
    y, logdet

  let inverse t z =
    let open Tensor in
    let w_inv = inverse (transpose !t.w) in
    let logdet = neg (select (slogdet !t.w) (-1)) in
    let x = matmul z w_inv in
    x, logdet
end