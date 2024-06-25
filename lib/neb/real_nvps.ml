open Torch

module RealNVP = struct
  type t = {
    d: int;
    alpha: float;
    s: Layer.t;
    t: Layer.t;
  }

  let clamp alpha s =
    let two_alpha_over_pi = 2.0 *. alpha /. Float.pi in
    Tensor.(two_alpha_over_pi * atan (s / (scalar alpha)))

  let create ~data_dim ~context_dim ~hidden_layer_dim ~alpha =
    let d = data_dim / 2 in
    let output_dim = data_dim - d in
    let s = Layer.sequential_ [
      Layer.linear ~input_dim:(d + context_dim) hidden_layer_dim;
      Layer.relu;
      Layer.linear ~input_dim:hidden_layer_dim hidden_layer_dim;
      Layer.relu;
      Layer.linear ~input_dim:hidden_layer_dim output_dim
    ] in
    let t = Layer.sequential_ [
      Layer.linear ~input_dim:(d + context_dim) hidden_layer_dim;
      Layer.relu;
      Layer.linear ~input_dim:hidden_layer_dim hidden_layer_dim;
      Layer.relu;
      Layer.linear ~input_dim:hidden_layer_dim output_dim
    ] in
    { d; alpha; s; t }

  let forward t ~y ?context () =
    let z, _ = forward_and_compute_log_jacobian t ~y ?context () in
    z

  let forward_and_compute_log_jacobian t ~y ?context () =
    let z1 = Tensor.narrow y ~dim:1 ~start:0 ~length:t.d in
    let s_input = match context with
      | Some ctx -> Tensor.cat [z1; ctx] ~dim:1
      | None -> z1
    in
    let s = Layer.forward t.s s_input |> clamp t.alpha in
    let t_input = match context with
      | Some ctx -> Tensor.cat [z1; ctx] ~dim:1
      | None -> z1
    in
    let t_val = Layer.forward t.t t_input in
    let z2 = Tensor.((narrow y ~dim:1 ~start:t.d ~length:(Tensor.shape y).(1) - t.d) * exp s + t_val) in
    Tensor.(cat [z1; z2] ~dim:1), Tensor.sum s ~dim:1

  let invert t ~z ?context () =
    let y1 = Tensor.narrow z ~dim:1 ~start:0 ~length:t.d in
    let s_input = match context with
      | Some ctx -> Tensor.cat [y1; ctx] ~dim:1
      | None -> y1
    in
    let s = Layer.forward t.s s_input |> clamp t.alpha in
    let t_input = match context with
      | Some ctx -> Tensor.cat [y1; ctx] ~dim:1
      | None -> y1
    in
    let t_val = Layer.forward t.t t_input in
    let y2 = Tensor.((narrow z ~dim:1 ~start:t.d ~length:(Tensor.shape z).(1) - t.d) - t_val * exp (-s)) in
    Tensor.cat [y1; y2] ~dim:1
end