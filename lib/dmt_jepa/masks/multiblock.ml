open Base
open Torch

let global_seed = 0
let logger = ref ""

type mask_collator = {
  patch_size : int;
  height : int;
  width : int;
  enc_mask_scale : float * float;
  pred_mask_scale : float * float;
  aspect_ratio : float * float;
  nenc : int;
  npred : int;
  min_keep : int;
  allow_overlap : bool;
  mutable itr_counter : int ref;
}

let create_mask_collator
    ~(input_size : int * int)
    ~(patch_size : int)
    ~(enc_mask_scale : float * float)
    ~(pred_mask_scale : float * float)
    ~(aspect_ratio : float * float)
    ~(nenc : int)
    ~(npred : int)
    ~(min_keep : int)
    ~(allow_overlap : bool) () =

  let height = fst input_size / patch_size in
  let width = snd input_size / patch_size in
  {
    patch_size;
    height;
    width;
    enc_mask_scale;
    pred_mask_scale;
    aspect_ratio;
    nenc;
    npred;
    min_keep;
    allow_overlap;
    itr_counter = ref (-1);
  }

let step mask_collator =
  mask_collator.itr_counter := !(mask_collator.itr_counter) + 1;
  !(mask_collator.itr_counter)

let sample_block_size mask_collator generator scale aspect_ratio_scale =
  let rand = Tensor.rand [1] ~generator |> Tensor.item |> Float.to_float in
  let min_s, max_s = scale in
  let mask_scale = min_s +. rand *. (max_s -. min_s) in
  let max_keep = Float.to_int (Float.of_int mask_collator.height *. Float.of_int mask_collator.width *. mask_scale) in
  let min_ar, max_ar = aspect_ratio_scale in
  let aspect_ratio = min_ar +. rand *. (max_ar -. min_ar) in
  let sqrt_max_keep_aspect_ratio = Float.sqrt (Float.of_int max_keep *. aspect_ratio) in
  let h = Int.of_float (Float.round sqrt_max_keep_aspect_ratio) in
  let w = Int.of_float (Float.round (Float.of_int max_keep /. aspect_ratio)) in
  let rec adjust_size size limit =
    if size >= limit then adjust_size (size - 1) limit else size
  in
  let h = adjust_size h mask_collator.height in
  let w = adjust_size w mask_collator.width in
  (h, w)

let sample_block_mask mask_collator block_size acceptable_regions =
  let h, w = block_size in

  let rec constrain_mask mask tries =
    let n = max (List.length acceptable_regions - tries) 0 in
    for k = 0 to n - 1 do
      Tensor.mul_ mask acceptable_regions.(k)
    done
  in

  let rec sample_mask tries =
    let top = Tensor.randint1 (Tensor.zeros [1]) (mask_collator.height - h) |> Tensor.to_int_exn in
    let left = Tensor.randint1 (Tensor.zeros [1]) (mask_collator.width - w) |> Tensor.to_int_exn in
    let mask = Tensor.zeros [mask_collator.height; mask_collator.width] Int32 in
    Tensor.narrow_copy_ mask ~dim:0 ~start:top ~length:h;
    Tensor.narrow_copy_ mask ~dim:1 ~start:left ~length:w;
    if Option.is_some acceptable_regions then constrain_mask mask tries;
    let mask = Tensor.nonzero mask |> Tensor.squeeze in
    let valid_mask = Tensor.length mask > mask_collator.min_keep in
    if valid_mask then mask else
      let timeout = 20 in
      if tries >= timeout then
        let tries = tries + 1 in
        logger := "Mask generator says: \"Valid mask not found, decreasing acceptable-regions [" ^ Int.to_string tries ^ "]\"";
        sample_mask tries
      else mask
  in
  let mask = sample_mask 0 in
  let mask_complement = Tensor.ones [mask_collator.height; mask_collator.width] Int32 in
  Tensor.narrow_copy_ mask_complement ~dim:0 ~start:top ~length:h;
  Tensor.narrow_copy_ mask_complement ~dim:1 ~start:left ~length:w;
  (mask, mask_complement)