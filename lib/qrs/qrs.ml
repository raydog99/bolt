open Torch

let sample_from_q () =
  Tensor.randn [1]

let calculate_p x =
  Tensor.exp (Tensor.neg (Tensor.pow x (Tensor.of_float 2.)))

let calculate_q x =
  Tensor.exp (Tensor.neg (Tensor.pow x (Tensor.of_float 2.)))

let qrs p q beta =
  let rec sample_loop () =
    let x = sample_from_q () in
    let px = calculate_p x in
    let qx = calculate_q x in
    let rx = Tensor.min (Tensor.ones_like px) (Tensor.div px (Tensor.mul (Tensor.of_float beta) qx)) in
    let u = Tensor.rand [1] in
    if Tensor.(u <= rx) then
      x
    else
      sample_loop ()
  in
  sample_loop

let generate_samples p q beta num_samples =
  let sampler = qrs p q beta in
  List.init num_samples (fun _ -> sampler ())