open Torch

module Kernel = struct
  type t = {
    mutable w: Tensor.t;
    mutable lambda: Tensor.t * Tensor.t;
    mutable z: Tensor.t;
    mutable q0: Tensor.t;
  }

  let create w lambda z q0 =
    { w; lambda; z; q0 }

  let split_dataset dataset =
    let size = Tensor.shape dataset |> List.hd in
    let split_point = size / 2 in
    Tensor.split ~split_size:[split_point; size - split_point] ~dim:0 dataset

  let sample_subsets dataset size =
    let total_size = Tensor.shape dataset |> List.hd in
    let indices = Tensor.randperm total_size ~dtype:(T Int64) in
    let subset_indices = Tensor.narrow indices ~dim:0 ~start:0 ~length:size in
    Tensor.index_select dataset ~dim:0 ~index:subset_indices

  let compute_f alpha lambda kw z data =
    let m = Tensor.shape alpha |> List.hd in
    List.init m (fun i ->
      let alpha_i = Tensor.select alpha ~dim:0 ~index:i in
      let z_i = Tensor.select z ~dim:0 ~index:i in
      Tensor.mul alpha_i (Tensor.dot kw z_i)
    ) |> Tensor.stack ~dim:0 |> Tensor.sum ~dim:0

  let compute_j f data =
    let n = Tensor.shape data |> List.hd in
    let d = Tensor.shape data |> List.tl |> List.hd in
    let grad_f = Tensor.grad f in
    let hessian_f = Tensor.hessian f in
    Tensor.add
      (Tensor.sum (Tensor.pow hessian_f 2) ~dim:[0; 1])
      (Tensor.mul (Tensor.pow grad_f 2) (Tensor.of_float 0.5))
    |> Tensor.mean

  let optimize_params t dataset1 dataset2 =
    let rec loop () =
      let dt, dv = sample_subsets dataset1 (Tensor.shape dataset1 |> List.hd |> ( / ) 10) in
      let f = compute_f t.w (fst t.lambda) t.z dt in
      let j = compute_j f dv in
      let grads = Tensor.grad j [t.w; fst t.lambda; snd t.lambda; t.z; t.q0] in
      List.iter2 (fun param grad ->
        Tensor.sub_ param (Tensor.mul grad (Tensor.of_float 0.01))
      ) [t.w; fst t.lambda; snd t.lambda; t.z; t.q0] grads;
      if Tensor.item j > 1e-6 then loop () else ()
    in
    loop ()

  let optimize_lambda t dataset2 =
    let rec loop () =
      let dv = sample_subsets dataset2 (Tensor.shape dataset2 |> List.hd |> ( / ) 10) in
      let f = compute_f t.w (fst t.lambda) t.z dv in
      let j = compute_j f dv in
      let grads = Tensor.grad j [fst t.lambda; snd t.lambda] in
      List.iter2 (fun param grad ->
        Tensor.sub_ param (Tensor.mul grad (Tensor.of_float 0.01))
      ) [fst t.lambda; snd t.lambda] grads;
      if Tensor.item j > 1e-6 then loop () else ()
    in
    loop ()

  let finalize_alpha t dataset1 =
    let alpha = Tensor.ones [Tensor.shape t.z |> List.hd] in
    compute_f alpha (fst t.lambda) t.w dataset1

  let train dataset =
    let d1, d2 = split_dataset dataset in
    let t = create
      (Tensor.randn [10; 10])
      (Tensor.randn [1], Tensor.randn [1])
      (Tensor.randn [10; 10])
      (Tensor.randn [1])
    in
    optimize_params t d1 d2;
    optimize_lambda t d2;
    let alpha = finalize_alpha t d1 in
    let m = Tensor.shape alpha |> List.hd in
    List.init m (fun i ->
      let alpha_i = Tensor.select alpha ~dim:0 ~index:i in
      let z_i = Tensor.select t.z ~dim:0 ~index:i in
      Tensor.mul alpha_i (Tensor.dot t.w z_i)
    ) |> Tensor.stack ~dim:0 |> Tensor.sum ~dim:0 |> Tensor.log |> Tensor.add (Tensor.log t.q0)
end