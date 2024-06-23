open Torch

module Point_JEPA_Masking = struct
  type t = {
    num_tokens: int;
    target_ratio: float * float;
    context_ratio: float * float;
  }

  let create_masks t sequenced_tokens =
    let target_start, target_end = t.target_ratio in
    let context_start, context_end = t.context_ratio in
    
    let target_start_idx = int_of_float (float t.num_tokens *. target_start) in
    let target_end_idx = int_of_float (float t.num_tokens *. target_end) in
    let context_start_idx = int_of_float (float t.num_tokens *. context_start) in
    let context_end_idx = int_of_float (float t.num_tokens *. context_end) in
    
    let target_mask = Tensor.zeros [t.num_tokens] in
    let context_mask = Tensor.ones [t.num_tokens] in
    
    (* Create target mask *)
    Tensor.narrow target_mask ~dim:0 ~start:target_start_idx ~length:(target_end_idx - target_start_idx)
    |> Tensor.fill_ 1.0;
    
    (* Create context mask by removing target region *)
    Tensor.narrow context_mask ~dim:0 ~start:target_start_idx ~length:(target_end_idx - target_start_idx)
    |> Tensor.fill_ 0.0;
    
    (* Apply context ratio *)
    let context_length = context_end_idx - context_start_idx in
    let context = Tensor.narrow sequenced_tokens ~dim:0 ~start:context_start_idx ~length:context_length in
    
    (target_mask, context_mask, context)

  let apply_masks sequenced_tokens target_mask context_mask =
    let target = Tensor.mul sequenced_tokens target_mask in
    let context = Tensor.mul sequenced_tokens context_mask in
    (target, context)
end