open Torch

module TS_JEPA_Masking = struct
  type t = {
    sequence_length: int;
    feature_dim: int;
    target_scale: float;
    target_aspect_ratio: float;
    context_scale: float;
    context_aspect_ratio: float;
    target_mask_number: int;
    training_step: int;
    curriculum: float -> float;
  }

  let sample_block_size (seq_len, feat_dim) scale aspect_ratio =
    let size = min seq_len feat_dim |> float_of_int |> ( *. ) scale |> int_of_float in
    let aspect = max 0.5 (min 2.0 aspect_ratio) in
    if aspect > 1.0 then
      (size, int_of_float (float_of_int size /. aspect))
    else
      (int_of_float (float_of_int size *. aspect), size)

  let sample_time_frequency_mask size =
    let (t, f) = size in
    let mask = Tensor.ones [t; f] in
    let t_start = Random.int (t - 1) in
    let f_start = Random.int (f - 1) in
    let t_length = 1 + Random.int (t - t_start) in
    let f_length = 1 + Random.int (f - f_start) in
    let mask_slice = Tensor.zeros [t_length; f_length] in
    Tensor.narrow mask ~dim:0 ~start:t_start ~length:t_length
    |> Tensor.narrow ~dim:1 ~start:f_start ~length:f_length
    |> Tensor.copy_ mask_slice;
    (mask, Tensor.( - ) (Tensor.ones [t; f]) mask)

  let sample_block_mask size =
    let (t, f) = size in
    let mask = Tensor.ones [t; f] in
    let start_t = Random.int (t - 1) in
    let start_f = Random.int (f - 1) in
    let length_t = 1 + Random.int (t - start_t) in
    let length_f = 1 + Random.int (f - start_f) in
    let mask_slice = Tensor.zeros [length_t; length_f] in
    Tensor.narrow mask ~dim:0 ~start:start_t ~length:length_t
    |> Tensor.narrow ~dim:1 ~start:start_f ~length:length_f
    |> Tensor.copy_ mask_slice;
    (mask, Tensor.( - ) (Tensor.ones [t; f]) mask)

  let sample_context_mask context_size acceptable_region =
    let (t, f) = context_size in
    let mask = Tensor.ones [t; f] in
    let acceptable_indices = Tensor.nonzero acceptable_region in
    let num_acceptable = Tensor.size acceptable_indices ~dim:0 in
    if num_acceptable > 0 then
      let random_index = Random.int num_acceptable in
      let chosen_index = Tensor.select acceptable_indices ~dim:0 ~index:random_index in
      let t_start = Tensor.get chosen_index ~dim:0 ~index:0 |> Tensor.int_value in
      let f_start = Tensor.get chosen_index ~dim:0 ~index:1 |> Tensor.int_value in
      Tensor.narrow mask ~dim:0 ~start:t_start ~length:1
      |> Tensor.narrow ~dim:1 ~start:f_start ~length:1
      |> Tensor.fill_ 0.0;
    mask

  let create_masks t =
    let context_size = sample_block_size (t.sequence_length, t.feature_dim) t.context_scale t.context_aspect_ratio in
    let target_size = sample_block_size (t.sequence_length, t.feature_dim) t.target_scale t.target_aspect_ratio in
    let p = Random.float 1.0 < t.curriculum (float_of_int t.training_step) in
    let target_mask_list = ref [] in
    let compliment_target_mask_list = ref [] in
    
    for _ = 1 to t.target_mask_number do
      let m, c =
        if p then sample_time_frequency_mask target_size
        else sample_block_mask target_size
      in
      target_mask_list := m :: !target_mask_list;
      compliment_target_mask_list := c :: !compliment_target_mask_list
    done;

    let acceptable_region = List.hd !compliment_target_mask_list in
    let context_mask = sample_context_mask context_size acceptable_region in
    
    (!target_mask_list, context_mask)
end

let () =
  let ts_jepa_masking = TS_JEPA_Masking.{
    sequence_length = 100;
    feature_dim = 128;
    target_scale = 0.5;
    target_aspect_ratio = 1.0;
    context_scale = 0.8;
    context_aspect_ratio = 1.0;
    target_mask_number = 4;
    training_step = 1000;
    curriculum = (fun s -> 0.5 *. (1.0 +. cos (Float.pi *. s /. 10000.0)));
  } in
  let target_masks, context_mask = TS_JEPA_Masking.create_masks ts_jepa_masking in
  Printf.printf "Generated %d target masks and 1 context mask\n" (List.length target_masks)