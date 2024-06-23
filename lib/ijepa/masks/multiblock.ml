open Torch

let global_seed = 0
let logger = ref ""

let step_counter = ref (-1)

let step () =
  step_counter := !step_counter + 1;
  !step_counter

let sample_block_size generator scale aspect_ratio_scale =
  let rand = Tensor.rand [1] ~generator in
  let min_s, max_s = scale in
  let mask_scale = min_s +. (Tensor.float_value rand) *. (max_s -. min_s) in
  let max_keep = int_of_float (1920.0 *. mask_scale) in
  let min_ar, max_ar = aspect_ratio_scale in
  let aspect_ratio = min_ar +. (Tensor.float_value rand) *. (max_ar -. min_ar) in
  let h = int_of_float (sqrt ((float_of_int max_keep) *. aspect_ratio)) in
  let w = int_of_float (sqrt ((float_of_int max_keep) /. aspect_ratio)) in
  let h = min h 1920 in
  let w = min w 1920 in
  h, w

let constrain_mask mask acceptable_regions tries =
  let n = max (List.length acceptable_regions - tries) 0 in
  for k = 0 to n - 1 do
    Tensor.mul_ mask acceptable_regions.(k)
  done

let rec sample_block_mask b_size acceptable_regions =
  let h, w = b_size in
  let tries = ref 0 in
  let timeout = ref 20 in
  let valid_mask = ref false in
  let mask = ref Tensor.(zeros [1920; 1920] ~dtype:Int32) in
  while not !valid_mask do
    let top = Tensor.randint 0 [1] (1920 - h) in
    let left = Tensor.randint 0 [1] (1920 - w) in
    Tensor.zero_ !mask;
    Tensor.scatter_ !mask 0 top left Tensor.(ones [h; w] ~dtype:Int32);
    if acceptable_regions <> None then constrain_mask !mask (Option.get acceptable_regions) !tries;
    let mask_indices = Tensor.nonzero (Tensor.flatten !mask) in
    valid_mask := Tensor.size mask_indices.(0) > 4;
    if not !valid_mask then begin
      decr timeout;
      if !timeout = 0 then begin
        incr tries;
        timeout := 20;
        logger := "Mask generator says: \"Valid mask not found, decreasing acceptable-regions [" ^ string_of_int !tries ^ "]\"";
      end
    end
  done;
  let mask = Tensor.squeeze mask in
  let mask_complement = Tensor.(ones [1920; 1920] ~dtype:Int32) in
  Tensor.scatter_ mask_complement 0 top left Tensor.(zeros [h; w] ~dtype:Int32);
  mask, mask_complement

let mask_collator
    ?(input_size=(224, 224))
    ?(patch_size=16)
    ?(enc_mask_scale=(0.2, 0.8))
    ?(pred_mask_scale=(0.2, 0.8))
    ?(aspect_ratio=(0.3, 3.0))
    ?(nenc=1)
    ?(npred=2)
    ?(min_keep=4)
    ?(allow_overlap=false) () =
  let input_size =
    match input_size with
    | h, w -> (h / patch_size, w / patch_size)
  in
  let height, width = fst input_size, snd input_size in
  let _itr_counter = ref (-1) in

  let step () =
    let i = _itr_counter in
    i := !i + 1;
    !i
  in

  let sample_block_size generator scale aspect_ratio_scale =
    let rand = Tensor.rand [1] ~generator in
    let min_s, max_s = scale in
    let mask_scale = min_s +. (Tensor.float_value rand) *. (max_s -. min_s) in
    let max_keep = int_of_float (float_of_int (height * width) *. mask_scale) in
    let min_ar, max_ar = aspect_ratio_scale in
    let aspect_ratio = min_ar +. (Tensor.float_value rand) *. (max_ar -. min_ar) in
    let h = int_of_float (sqrt ((float_of_int max_keep) *. aspect_ratio)) in
    let w = int_of_float (sqrt ((float_of_int max_keep) /. aspect_ratio)) in
    let h = min h height in
    let w = min w width in
    h, w
  in

  let sample_block_mask b_size acceptable_regions =
    let h, w = b_size in
    let tries = ref 0 in
    let timeout = ref 20 in
    let valid_mask = ref false in
    let mask = ref Tensor.(zeros [height; width] ~dtype:Int32) in
    while not !valid_mask do
      let top = Tensor.randint 0 [1] (height - h) in
      let left = Tensor.randint 0 [1] (width - w) in
      Tensor.zero_ !mask;
      Tensor.scatter_ !mask 0 top left Tensor.(ones [h; w] ~dtype:Int32);
      if acceptable_regions <> None then constrain_mask !mask (Option.get acceptable_regions) !tries;
      let mask_indices = Tensor.nonzero (Tensor.flatten !mask) in
      valid_mask := Tensor.size mask_indices.(0) > min_keep;
      if not !valid_mask then begin
        decr timeout;
        if !timeout = 0 then begin
          incr tries;
          timeout := 20;
          logger := "Mask generator says: \"Valid mask not found, decreasing acceptable-regions [" ^ string_of_int !tries ^ "]\"";
        end
      end
    done;
    let mask = Tensor.squeeze mask in
    let mask_complement = Tensor.(ones [height; width] ~dtype:Int32) in
    Tensor.scatter_ mask_complement 0 top left Tensor.(zeros [h; w] ~dtype:Int32);
    mask, mask_complement
  in

  let collate_batch batch =
    let seed = step () in
    let g = Tensor.default_generator () in
    Tensor.manual_seed g seed;
    let p_size = sample_block_size g pred_mask_scale aspect_ratio in
    let e_size = sample_block_size g enc_mask_scale (1.0, 1.0) in
    let collated_masks_pred, collated_masks_enc = ref [], ref [] in
    let min_keep_pred = ref (height * width) in
    let min_keep_enc = ref (height * width) in
    List.iter (fun _ ->
      let masks_p, masks_C = ref [], ref [] in
      for _ = 1 to npred do
        let mask, mask_C = sample_block_mask p_size None in
        masks_p := mask :: !masks_p;
        masks_C := mask_C :: !masks_C;
        min_keep_pred := min !min_keep_pred (Tensor.size mask);
      done;
      collated_masks_pred := !masks_p :: !collated_masks_pred;
      let acceptable_regions =
        if allow_overlap then None
        else Some !masks_C
      in
      let masks_e = ref [] in
      for _ = 1 to nenc do
        let mask, _ = sample_block_mask e_size acceptable_regions in
        masks_e := mask :: !masks_e;
        min_keep_enc := min !min_keep_enc (Tensor.size mask);
      done;
      collated_masks_enc := !masks_e :: !collated_masks_enc;
    ) batch;
    let collated_masks_pred =
      List.map (fun cm_list ->
        List.map (fun cm -> Tensor.slice cm 0 !min_keep_pred) cm_list
      ) !collated_masks_pred
    in
    let collated_masks_enc =
      List.map (fun cm_list ->
        List.map (fun cm -> Tensor.slice cm 0 !min_keep_enc) cm_list
      ) !collated_masks_enc
    in
    (batch, collated_masks_enc, collated_masks_pred)
  in

  collate_batch