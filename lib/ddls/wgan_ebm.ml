open Torch

module WGAN_EBM = struct
  type t = {
    mutable d: Layer.t;  (* Discriminator D_φ *)
    mutable g: Layer.t;  (* Generator G_θ *)
    n: int;
    epsilon: float;
    delta: float;
    m: int;  (* Batch size *)
  }

  let create d g n epsilon delta m =
    { d; g; n; epsilon; delta; m }

  let sample_noise shape =
    Tensor.randn shape

  let sample_real_data () =
    Tensor.randn [1]

  let energy z =
    let p0_z = Tensor.normal_like z ~mean:0. ~std:1. in
    let g_z = Layer.forward t.g z in
    Tensor.neg (Tensor.log p0_z) - Layer.forward t.d g_z

  let train t =
    while not (is_converged t) do
      let z0 = List.init t.m (fun _ -> sample_noise [1]) in
      let x_real = List.init t.m (fun _ -> sample_real_data ()) in
      
      let z = ref z0 in
      for i = 0 to t.n - 1 do
        let noise = List.map (fun _ -> Tensor.randn [1]) !z in
        let grad_e = List.map (fun zi -> Tensor.grad (energy zi)) !z in
        z := List.map3 (fun zi grad n ->
          zi - (t.epsilon /. 2.) * grad + Tensor.sqrt (Tensor.full_like zi t.epsilon) * n
        ) !z grad_e noise
      done;

      let grad_d_real = List.map (fun x -> Tensor.grad (Layer.forward t.d x)) x_real in
      let grad_d_fake = List.map (fun z -> Tensor.grad (Layer.forward t.d (Layer.forward t.g z))) !z in
      let grad_d = List.map2 Tensor.sub grad_d_fake grad_d_real |> List.fold_left Tensor.add (Tensor.zeros [1]) in
      
      let grad_g = List.map (fun z -> Tensor.grad (Layer.forward t.d (Layer.forward t.g z))) z0 
                   |> List.fold_left Tensor.add (Tensor.zeros [1]) in

      t.d <- Layer.update t.d ~learning_rate:(Tensor.f t.delta) grad_d;
      t.g <- Layer.update t.g ~learning_rate:(Tensor.f t.delta) grad_g
    done;
    (t.d, t.g)

  let is_converged t =
    false
end