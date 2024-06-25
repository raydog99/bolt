open Torch

module Multigrid = struct
  type t = {
    sample_dir : string;
    learn_rate : float;
    wt_decay : float;
    beta1 : float;
    load_models : bool;
    model_dir : string;
    delta : float;
    l : int;
    scale_list : int list;
    epoch_file : string;
    epochs : int;
    batch_size : int;
    mutable models : (string, Layer.t) Hashtbl.t;
    mutable optimizers : (string, Optimizer.t) Hashtbl.t;
  }

  let net_wrapper im_size =
    match im_size with
    | 4 -> net1 ()
    | 16 -> net2 ()
    | 64 -> net3 ()
    | _ -> failwith (Printf.sprintf "Unsupported model version %d" im_size)

  let weights_init m =
    let classname = Layer.name m in
    if String.contains classname "Conv" then
      Layer.set_weights m (Tensor.normal ~mean:0.0 ~std:0.02 (Layer.weights_shape m))
    else if String.contains classname "BatchNorm" then
      Layer.set_weights m (Tensor.normal ~mean:1.0 ~std:0.02 (Layer.weights_shape m))

  let langevin_dynamics t samples size =
    let noise = Tensor.zeros [1; Tensor.shape samples |> List.nth 1; size; size] in
    Layer.train (Hashtbl.find t.models (string_of_int size));
    let samples = scale_up samples size in
    let samples = Tensor.requires_grad samples in
    for _ = 1 to t.l do
      let noise_term = Tensor.(sqrt (Scalar.float t.delta) * noise.normal_ ()) in
      let e = Layer.forward (Hashtbl.find t.models (string_of_int size)) samples in
      Tensor.backward e;
      samples := Tensor.clamp (Tensor.(!samples + 0.5 * Scalar.float t.delta * t.delta * Tensor.grad samples) + noise_term) ~min:0. ~max:255.;
      Tensor.zero_grad samples;
    done;
    samples

  let hundred_images t data =
    let originals = Tensor.zeros [100; 3; 1; 1] in
    let scale_list = t.scale_list in
    for i = 0 to 99 do
      let rand = Random.int 1000 in
      Tensor.copy_ (Tensor.narrow originals 0 i 1) data.(rand).(1);
    done;
    let samples = Hashtbl.create (List.length scale_list) in
    Hashtbl.add samples "1" originals;
    for i = 1 to List.length scale_list - 1 do
      let prev_scale = string_of_int (List.nth scale_list (i - 1)) in
      let curr_scale = string_of_int (List.nth scale_list i) in
      let sample = langevin_dynamics t (Hashtbl.find samples prev_scale) (List.nth scale_list i) in
      Hashtbl.add samples curr_scale sample;
    done;
    samples

  let train t =
    let data = process_data_parallel t.sample_dir in
    let e = int_of_string (Stdlib.input_line (Stdlib.open_in t.epoch_file)) in
    let epochs = t.epochs - e in
    Printf.printf "Will train for %d epochs\n" epochs;
    let batch_size = t.batch_size in
    let scale_list = t.scale_list in
    List.iter (fun (_, model) -> Layer.train model) (Hashtbl.to_seq t.models);
    for epoch = 1 to epochs do
      let start_time = Unix.gettimeofday () in
      for batch = 0 to Array.length data / batch_size - 1 do
        let batch_ori = Hashtbl.create (List.length scale_list) in
        let samples = Hashtbl.create (List.length scale_list) in
        for image_no = batch * batch_size to min ((batch + 1) * batch_size - 1) (Array.length data - 1) do
          List.iter (fun scale -> 
            let scale_str = string_of_int scale in
            if not (Hashtbl.mem batch_ori scale_str) then Hashtbl.add batch_ori scale_str [];
            Hashtbl.replace batch_ori scale_str (data.(image_no).(scale) :: Hashtbl.find batch_ori scale_str)) scale_list;
        done;
        Hashtbl.iter (fun scale imgs -> 
          Hashtbl.replace batch_ori scale (Tensor.cat (Array.of_list imgs) ~dim:0)) batch_ori;
        Hashtbl.replace samples "1" (Hashtbl.find batch_ori "1");
        for i = 1 to List.length scale_list - 1 do
          let prev_scale = string_of_int (List.nth scale_list (i - 1)) in
          let curr_scale = string_of_int (List.nth scale_list i) in
          let sample = langevin_dynamics t (Hashtbl.find samples prev_scale) (List.nth scale_list i) in
          Hashtbl.add samples curr_scale sample;
        done;
        let losses = Hashtbl.create (List.length scale_list) in
        List.iter (fun scale ->
          if scale <> 1 then (
            let scale_str = string_of_int scale in
            let e_samples = Tensor.mean (Layer.forward (Hashtbl.find t.models scale_str) (Hashtbl.find samples scale_str)) in
            let e_ori = Tensor.mean (Layer.forward (Hashtbl.find t.models scale_str) (Hashtbl.find batch_ori scale_str)) in
            let loss = Tensor.(e_samples - e_ori) in
            Hashtbl.add losses scale_str loss;
            let reconstruction_loss = Tensor.mean Tensor.(loss * loss) in
            Optimizer.zero_grad (Hashtbl.find t.optimizers scale_str);
            Tensor.backward loss;
            Optimizer.step (Hashtbl.find t.optimizers scale_str);
          )) scale_list;
      done;
      let load_dir = t.model_dir in
      Hashtbl.iter (fun size model -> 
        Torch.save (Layer.state_dict model) (Filename.concat load_dir (Printf.sprintf "model_%s.pt" size))) t.models;
      Hashtbl.iter (fun size optimizer -> 
        Torch.save (Optimizer.state_dict optimizer) (Filename.concat load_dir (Printf.sprintf "optimizer_%s.pt" size))) t.optimizers;
      let epo = int_of_string (Stdlib.input_line (Stdlib.open_in t.epoch_file)) in
      let oc = Stdlib.open_out t.epoch_file in
      Printf.fprintf oc "%d" (epo + 1);
      Stdlib.close_out oc;
      let himages = Tensor.(permute (hundred_images t data).(64) [0; 2; 3; 1]) |> Tensor.to_type UInt8 in
      let canvas = merge_images himages in
      Skimage_io.imsave (Filename.concat t.sample_dir (Printf.sprintf "epoch%d.jpg" (epo + 1))) canvas;
      Printf.printf "Epoch %d completed; Time taken: %.2f seconds; Reconstruction Loss: %.4f\n" (epo + 1) (Unix.gettimeofday () -. start_time) reconstruction_loss;
    done;
end