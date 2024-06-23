open Torch

let global_seed = 0
let logger = ref ""

type mask_generator_config = {
  spatial_scale : float;
  temporal_scale : float;
  aspect_ratio : float;
  num_blocks : int;
  max_temporal_keep : float option;
  max_keep : int option;
}

type mask_generator = {
  crop_size : int * int;
  num_frames : int;
  spatial_patch_size : int * int;
  temporal_patch_size : int;
  spatial_pred_mask_scale : float option;
  temporal_pred_mask_scale : float option;
  aspect_ratio : float option;
  npred : int option;
  max_context_frames_ratio : float option;
  max_keep : int option;
}

let mask_collator
    ~(cfgs_mask : mask_generator_config list)
    ~(crop_size : int * int)
    ~(num_frames : int)
    ~(patch_size : int * int)
    ~(tubelet_size : int) () =

  let mask_generators =
    List.map (fun m ->
      let spatial_scale = Option.value m.spatial_scale ~default:1.0 in
      let temporal_scale = Option.value m.temporal_scale ~default:1.0 in
      let aspect_ratio = Option.value m.aspect_ratio ~default:1.0 in
      let num_blocks = Option.value m.num_blocks ~default:0 in
      let max_temporal_keep = Option.value m.max_temporal_keep ~default:1.0 in
      let max_keep = Option.value m.max_keep ~default:None in
      let mask_generator =
        {
          crop_size;
          num_frames;
          spatial_patch_size = patch_size;
          temporal_patch_size = tubelet_size;
          spatial_pred_mask_scale = Some spatial_scale;
          temporal_pred_mask_scale = Some temporal_scale;
          aspect_ratio = Some aspect_ratio;
          npred = Some num_blocks;
          max_context_frames_ratio = Some max_temporal_keep;
          max_keep;
        }
      in
      mask_generator
    ) cfgs_mask
  in

  let step () =
    List.iter (fun mask_gen -> ()) mask_generators
  in

  let collate_batch batch =
    let batch_size = List.length batch in
    let collated_batch = Tensor_utils.collate batch in
    let collated_masks_pred = ref [] in
    let collated_masks_enc = ref [] in
    List.iteri (fun i mask_gen ->
      let masks_enc, masks_pred = mask_gen batch_size in
      collated_masks_enc := masks_enc :: !collated_masks_enc;
      collated_masks_pred := masks_pred :: !collated_masks_pred;
    ) mask_generators;
    (collated_batch, List.rev !collated_masks_enc, List.rev !collated_masks_pred)
  in

  step, collate_batch