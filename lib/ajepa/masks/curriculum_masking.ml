open Torch

module CurriculumMasking = struct
  type t = {
    image_size: int * int;
    target_scale: float;
    target_aspect_ratio: float;
    context_scale: float;
    context_aspect_ratio: float;
    target_mask_number: int;
    training_step: int;
    curriculum: float -> float;
  }

  let sample_block_size (w, h) scale aspect_ratio =
    let size = min w h |> float_of_int |> ( *. ) scale |> int_of_float in
    let aspect = max 0.5 (min 2.0 aspect_ratio) in
    if aspect > 1.0 then
      (size, int_of_float (float_of_int size /. aspect))
    else
      (int_of_float (float_of_int size *. aspect), size)

  let sample_time_frequency_mask size =
    let mask = Tensor.zeros size in
    (mask, mask)

  let sample_block_mask size =
    let mask = Tensor.zeros size in
    (mask, mask)

  let sample_context_mask context_size acceptable_region =
    let mask = Tensor.zeros context_size in
    mask

  let create_masks t =
    let context_size = sample_block_size t.image_size t.context_scale t.context_aspect_ratio in
    let target_size = sample_block_size t.image_size t.target_scale t.target_aspect_ratio in
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