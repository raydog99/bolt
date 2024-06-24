open Base
open Torch

let rand_active tgt_unlabeled_ds tgt_selected_ds active_ratio totality =
  let length = List.length tgt_unlabeled_ds.samples in
  let num_samples_to_select = Float.to_int (Float.round_up (totality *. active_ratio)) in
  let indices = List.init length Fun.id |> List.permute |> List.take num_samples_to_select in
  let active_samples = List.take_indices tgt_unlabeled_ds.samples indices in
  
  tgt_selected_ds#add_item active_samples;
  tgt_unlabeled_ds#remove_item indices;

  active_samples

let eada_active tgt_unlabeled_loader_full tgt_unlabeled_ds tgt_selected_ds active_ratio totality model cfg =
  model#eval;

  let first_stat = ref [] in
  Torch.no_grad (fun () ->
    tgt_unlabeled_loader_full |> Torch.iter (fun data ->
      let tgt_img = data#"img" |> Torch.to_cuda |> Torch.tensor in
      let tgt_lbl = data#"label" |> Torch.to_cuda |> Torch.tensor in
      let tgt_path = data#"path" |> Torch.tensor in
      let tgt_index = data#"index" |> Torch.tensor in

      let tgt_out = model#forward tgt_img ~return_feat:false in

      let min2 = tgt_out |> Torch.topk 2 ~dim:1 ~largest:false ~sorted:false |> Torch.values in
      let mvsm_uncertainty = min2#[:, 0] - min2#[:, 1] in

      let output_div_t = tgt_out * -1.0 / cfg#trainer#energy_beta in
      let output_logsumexp = output_div_t |> Torch.logsumexp ~dim:1 ~keepdim:false in
      let free_energy = output_logsumexp * -1.0 * cfg#trainer#energy_beta in

      for i = 0 to Torch.size free_energy#dim 0 - 1 do
        first_stat := (tgt_path#[(i)]) :: (tgt_lbl#[(i)] |> Torch.item) :: (tgt_index#[(i)] |> Torch.item) :: (mvsm_uncertainty#[(i)] |> Torch.item) :: (free_energy#[(i)] |> Torch.item) :: !first_stat
      done
    )
  );

  let first_sample_ratio = cfg#trainer#first_sample_ratio in
  let first_sample_num = Float.round_up (totality *. first_sample_ratio) |> Float.to_int in
  let second_sample_ratio = active_ratio /. cfg#trainer#first_sample_ratio in
  let second_sample_num = Float.round_up (Float.of_int first_sample_num *. second_sample_ratio) |> Float.to_int in

  let first_stat = List.sort (fun a b -> compare b#.(4) a#.(4)) !first_stat in
  let second_stat = List.take first_stat first_sample_num in

  let second_stat = List.sort (fun a b -> compare b#.(3) a#.(3)) second_stat in
  let active_samples = List.take second_stat second_sample_num |> Array.of_list |> Array.map (fun x -> [| x#.(0); x#.(1) |]) in
  let candidate_ds_index = List.take second_stat second_sample_num |> Array.of_list |> Array.map (fun x -> x#.(2) |> Float.to_int) in

  tgt_selected_ds#add_item active_samples;
  tgt_unlabeled_ds#remove_item candidate_ds_index;

  active_samples