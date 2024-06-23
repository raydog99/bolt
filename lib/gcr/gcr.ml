open Torch

module GibbsLangevinSampler = struct
  type buffer = (Tensor.t * Tensor.t) list
  
  let sample_from_buffer buffer =
    let idx = Random.int (List.length buffer) in
    List.nth buffer idx |> fst

  let sample_y_given_x p_y_given_x x =
    Tensor.randn_like x

  let gradient f x y =
    let x' = Tensor.requires_grad x in
    let y' = Tensor.requires_grad y in
    let loss = f x' y' in
    Tensor.backward loss;
    Tensor.grad x'

  let sample buffer s f p_y_given_x =
    let x0 = sample_from_buffer buffer in
    
    let rec loop s x_prev buffer =
      if s = 0 then (x_prev, List.hd buffer |> snd, buffer)
      else
        let eta_s = 1. /. float_of_int s in
        let y_s = sample_y_given_x p_y_given_x x_prev in
        let grad = gradient f x_prev y_s in
        let epsilon = Tensor.(randn_like x_prev * scalar (sqrt eta_s)) in
        let x_s = Tensor.(x_prev + (grad * scalar (0.5 *. eta_s)) + epsilon) in
        let buffer' = (x_s, y_s) :: buffer in
        loop (s - 1) x_s buffer'
    in
    
    loop s x0 buffer

  let run buffer s f p_y_given_x =
    let x_s, y_s, buffer' = sample buffer s f p_y_given_x in
    (x_s, y_s, buffer')
end