open Torch

module ESMq_sigma = struct
  let forward psi q_sigma x theta =
    let log_q_sigma = q_sigma x in
    let grad_log_q_sigma = Tensor.grad log_q_sigma ~inputs:[x] |> List.hd in
    let psi_x = psi x theta in
    let diff = Tensor.(psi_x - grad_log_q_sigma) in
    Tensor.((pow diff (Scalar 2.)) / (Scalar 2.))

  let objective psi q_sigma x_samples theta =
    let batch_size = Tensor.size x_samples 0 in
    let losses = Tensor.map (fun x -> forward psi q_sigma x theta) x_samples in
    Tensor.(mean losses)
end

module Parzen_window = struct
  let estimate x_samples sigma =
    let kernel x = 
      Tensor.(exp (neg (pow x (Scalar 2.)) / (Scalar (2. *. sigma *. sigma))))
    fun x ->
      let diffs = Tensor.(x - x_samples) in
      let kernels = Tensor.map kernel diffs in
      Tensor.(mean kernels / (Scalar sigma))

  let log_estimate x_samples sigma =
    let q = estimate x_samples sigma in
    fun x -> Tensor.log (q x)
end