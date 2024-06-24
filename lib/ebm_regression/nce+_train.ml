open Torch
open Printf

let model_id = "1-nce+"

let num_epochs = 75
let batch_size = 32
let learning_rate = 0.001

let num_samples = 1024
let () = printf "%d\n" num_samples

let beta = 0.025
let () = printf "beta: %g\n" beta

let gauss_density_centered x std =
  Tensor.(exp (-0.5 * (x / std) ** f 2.) / (f (sqrt (2. *. Float.pi)) * std))

let gmm_density_centered x std =
  let x = if Tensor.dim x = Tensor.dim std - 1 then Tensor.unsqueeze x ~dim:~(-1) else x in
  if not (Tensor.dim x = Tensor.dim std && Tensor.shape x |> List.rev |> List.hd = Some 1)
  then failwith "Last dimension must be the gmm stds.";
  Tensor.(gauss_density_centered x std |> prod ~dim:(~(-2)) |> mean ~dim:(~(-1)))

let sample_gmm_centered std ~num_samples =
  let num_components = Tensor.shape std |> List.rev |> List.hd |> Option.get in
  let num_dims = Tensor.numel std / num_components in
  let std = Tensor.view std ~size:[1; num_dims; num_components] in
  let k = Tensor.randint ~high:num_components ~size:[num_samples] ~options:(T.int64, Cpu) in
  let std_samp = Tensor.gather std ~dim:2 ~index:k |> Tensor.transpose ~dim0:0 ~dim1:1 in
  let x_centered = Tensor.(std_samp * randn [num_samples; num_dims]) in
  let prob_dens = gmm_density_centered x_centered std in
  let prob_dens_zero = gmm_density_centered (Tensor.zeros_like x_centered) std in
  (x_centered, prob_dens, prob_dens_zero)

let sample_gmm_centered2 beta std ~num_samples =
  let num_components = Tensor.shape std |> List.rev |> List.hd |> Option.get in
  let num_dims = Tensor.numel std / num_components in
  let std = Tensor.view std ~size:[1; num_dims; num_components] in
  let k = Tensor.randint ~high:num_components ~size:[num_samples] ~options:(T.int64, Cpu) in
  let std_samp = Tensor.gather std ~dim:2 ~index:k |> Tensor.transpose ~dim0:0 ~dim1:1 in
  let x_centered = Tensor.(f beta * std_samp * randn [num_samples; num_dims]) in
  let prob_dens = gmm_density_centered x_centered std in
  let prob_dens_zero = gmm_density_centered (Tensor.zeros_like x_centered) std in
  (x_centered, prob_dens, prob_dens_zero)