open Torch

module EMAHelper = struct
  type t = {
    mu : float;
    mutable shadow : (string, Tensor.t) Hashtbl.t;
  }

  let create ?(mu=0.999) () =
    { mu; shadow = Hashtbl.create 16 }

  let register t module_ =
    let module_ = 
      if Tensor.is_module_type module_ "DataParallel" then 
        Tensor.get_module_field module_ "module" 
      else 
        module_ 
    in
    let named_parameters = Tensor.named_parameters module_ in
    List.iter (fun (name, param) ->
      if Tensor.requires_grad param then
        Hashtbl.add t.shadow name (Tensor.clone (Tensor.data param))
    ) named_parameters

  let update t module_ =
    let module_ = 
      if Tensor.is_module_type module_ "DataParallel" then 
        Tensor.get_module_field module_ "module" 
      else 
        module_ 
    in
    let named_parameters = Tensor.named_parameters module_ in
    List.iter (fun (name, param) ->
      if Tensor.requires_grad param then
        let shadow_param = Hashtbl.find t.shadow name in
        let new_shadow_param = Tensor.(add (mul_scalar shadow_param t.mu) (mul_scalar (data param) (1. -. t.mu))) in
        Hashtbl.replace t.shadow name new_shadow_param
    ) named_parameters

  let ema t module_ =
    let module_ = 
      if Tensor.is_module_type module_ "DataParallel" then 
        Tensor.get_module_field module_ "module" 
      else 
        module_ 
    in
    let named_parameters = Tensor.named_parameters module_ in
    List.iter (fun (name, param) ->
      if Tensor.requires_grad param then
        let shadow_param = Hashtbl.find t.shadow name in
        Tensor.copy_ param shadow_param
    ) named_parameters

  let ema_copy t module_ =
    let module_copy = 
      if Tensor.is_module_type module_ "DataParallel" then 
        let inner_module = Tensor.get_module_field module_ "module" in
        let module_copy = Tensor.create_module (Tensor.get_module_type inner_module) in
        Tensor.load_state_dict module_copy (Tensor.state_dict inner_module);
        Tensor.wrap_module "DataParallel" module_copy
      else 
        let module_copy = Tensor.create_module (Tensor.get_module_type module_) in
        Tensor.load_state_dict module_copy (Tensor.state_dict module_);
        module_copy
    in
    ema t module_copy;
    module_copy

  let state_dict t =
    t.shadow

  let load_state_dict t state_dict =
    t.shadow <- state_dict
end