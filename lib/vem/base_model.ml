open Torch

module BaseModel = struct
  type t = {
    opt : Options.t;
    gpu_ids : int list;
    is_train : bool;
    device : Device.t;
    save_dir : string;
    model_names : string list;
    visual_names : string list;
    image_paths : string list;
    mutable metric : float;
    mutable schedulers : Optimizer.t list;
    mutable optimizers : Optimizer.t list;
  }

  let create opt =
    let gpu_ids = opt.gpu_ids in
    let device = 
      if List.length gpu_ids > 0 then Device.Cuda (List.hd gpu_ids)
      else Device.Cpu
    in
    let save_dir = 
      if opt.is_train then 
        let dir = Filename.concat opt.log_dir "checkpoints" in
        Unix.mkdir dir 0o755;
        dir
      else ""
    in
    { opt; gpu_ids; is_train = opt.is_train; device; save_dir;
      model_names = []; visual_names = []; image_paths = [];
      metric = 0.; schedulers = []; optimizers = [] }

  let modify_commandline_options parser is_train =
    parser

  let set_input t input =
    ()

  let forward t =
    ()

  let optimize_parameters t steps =
    ()

  let setup t opt verbose =
    if t.is_train then
      t.schedulers <- List.map (fun optimizer -> Networks.get_scheduler optimizer opt) t.optimizers;
    load_networks t verbose;
    if verbose then print_networks t

  let print_networks t =
    Printf.printf "---------- Networks initialized -------------\n";
    List.iter (fun name ->
      let net = Obj.magic (Obj.field (Obj.repr t) (Obj.field_pos ("net" ^ name))) in
      let num_params = 
        Tensor.sum (Tensor.of_float1 (List.map (fun p -> float (Tensor.numel p)) (Module.parameters net)))
      in
      Printf.printf "[Network %s] Total number of parameters : %.3f M\n" name (Tensor.to_float0_exn num_params /. 1e6);
      if Sys.file_exists t.opt.log_dir then
        let oc = open_out (Filename.concat t.opt.log_dir ("net" ^ name ^ ".txt")) in
        Printf.fprintf oc "%s\n" (Module.to_string net);
        Printf.fprintf oc "[Network %s] Total number of parameters : %.3f M\n" name (Tensor.to_float0_exn num_params /. 1e6);
        close_out oc
    ) t.model_names;
    Printf.printf "-----------------------------------------------\n"

  let eval t =
    List.iter (fun name ->
      let net = Obj.magic (Obj.field (Obj.repr t) (Obj.field_pos ("net" ^ name))) in
      Module.eval net
    ) t.model_names

  let train t =
    List.iter (fun name ->
      let net = Obj.magic (Obj.field (Obj.repr t) (Obj.field_pos ("net" ^ name))) in
      Module.train net
    ) t.model_names

  let test t =
    No_grad.run (fun () -> forward t)

  let get_image_paths t = t.image_paths

  let update_learning_rate t ?logger () =
    List.iter (fun scheduler ->
      if t.opt.lr_policy = "plateau" then
        Optimizer.Step.step scheduler ~metrics:t.opt.metric
      else
        Optimizer.Step.step scheduler
    ) t.schedulers;
    let lr = Optimizer.Param_group.lr (List.hd (Optimizer.param_groups (List.hd t.optimizers))) in
    match logger with
    | Some l -> Logger.print_info l (Printf.sprintf "learning rate = %.7f\n" lr)
    | None -> Printf.printf "learning rate = %.7f\n" lr

  let get_current_visuals t =
    List.fold_left (fun acc name ->
      if Obj.magic (Obj.field (Obj.repr t) (Obj.field_pos name)) <> Obj.magic None then
        (name, Obj.magic (Obj.field (Obj.repr t) (Obj.field_pos name))) :: acc
      else acc
    ) [] t.visual_names

  let get_current_losses t =
    List.fold_left (fun acc name ->
      if Obj.magic (Obj.field (Obj.repr t) (Obj.field_pos ("loss_" ^ name))) <> Obj.magic None then
        let key = 
          if String.contains name '0' then "Specific_loss/" ^ name
          else if String.starts_with ~prefix:"D_" name then "D_loss/" ^ name
          else if String.starts_with ~prefix:"G_" name then "G_loss/" ^ name
          else if String.starts_with ~prefix:"S_" name then "S_loss/" ^ name
          else failwith "Invalid loss name"
        in
        (key, Tensor.to_float0_exn (Obj.magic (Obj.field (Obj.repr t) (Obj.field_pos ("loss_" ^ name))))) :: acc
      else acc
    ) [] t.loss_names

  let load_networks t verbose =
    List.iter (fun name ->
      let net = Obj.magic (Obj.field (Obj.repr t) (Obj.field_pos ("net" ^ name))) in
      let path = Obj.magic (Obj.field (Obj.repr t.opt) (Obj.field_pos ("restore_" ^ name ^ "_path"))) in
      if path <> Obj.magic None then
        Util.load_network net path verbose
    ) t.model_names

  let save_networks t epoch =
    List.iter (fun name ->
      let save_filename = Printf.sprintf "%s_net_%s.pth" epoch name in
      let save_path = Filename.concat t.save_dir save_filename in
      let net = Obj.magic (Obj.field (Obj.repr t) (Obj.field_pos ("net" ^ name))) in
      if List.length t.gpu_ids > 0 && Cuda.is_available () then begin
        if Obj.tag (Obj.repr net) = Obj.tag (Obj.repr (module Nn.DataParallel : Nn.S)) then
          Serialize.save (Module.state_dict (Module.to_device net ~device:Device.Cpu)) ~filename:save_path
        else
          Serialize.save (Module.state_dict (Module.to_device net ~device:Device.Cpu)) ~filename:save_path;
        ignore (Module.to_device net ~device:(Device.Cuda (List.hd t.gpu_ids)))
      end else
        Serialize.save (Module.state_dict (Module.to_device net ~device:Device.Cpu)) ~filename:save_path
    ) t.model_names

  let set_requires_grad t nets requires_grad =
    let nets = if not (List.is_empty nets) then nets else [nets] in
    List.iter (fun net ->
      if net <> Obj.magic None then
        List.iter (fun p -> Tensor.set_requires_grad p requires_grad) (Module.parameters net)
    ) nets
end