open Torch

module type GNNModule = sig
  val get_out_dim : unit -> int
  val forward : Tensor.t -> Tensor.t -> Tensor.t -> Tensor.t
end

module type MLPModule = sig
  val create : num_layers:int -> input_dim:int -> hidden_dim:int -> output_dim:int -> activate_func:(Tensor.t -> Tensor.t) -> use_bn:bool -> num_classes:int -> Tensor.t
  val forward : Tensor.t -> Tensor.t -> Tensor.t
end

module EdgeDensePredictionGNNLayer = struct
  type t =
    { multi_channel_gnn_module : (module GNNModule)
    ; translate_mlp : (module MLPModule) }

  let create gnn_module c_in c_out num_classes =
    let gnn_mod = (module struct
      include (val gnn_module : GNNModule)
    end : GNNModule) in
    let mlp = MLP.create ~num_layers:3 ~input_dim:(c_in + 2 * GNNModule.get_out_dim ()) ~hidden_dim:(max c_in c_out * 2) ~output_dim:c_out ~activate_func:Tensor.elu ~use_bn:true ~num_classes in
    { multi_channel_gnn_module = gnn_mod; translate_mlp = mlp }

  let node_feature_to_matrix x =
    let b, n, f = Tensor.shape3_exn x in
    let x_i = Tensor.unsqueeze x ~dim:2 in
    let x_j = Tensor.unsqueeze x ~dim:1 in
    Tensor.cat [x_i; x_j] ~dim:(-1)

  let mask_adjs adjs node_flags =
    let mask = Tensor.(unsqueeze (unsqueeze node_flags ~dim:1) ~dim:1) in
    Tensor.(adjs * mask * Tensor.transpose mask ~dim0:2 ~dim1:3)

  let forward t x adjs node_flags =
    let gnn_mod = (val t.multi_channel_gnn_module : GNNModule) in
    let mlp_mod = (val t.translate_mlp : MLPModule) in
    let x_o = GNNModule.forward gnn_mod x adjs node_flags in
    let x_o_pair = node_feature_to_matrix x_o in
    let last_c_adjs = Tensor.permute adjs ~dims:[0; 2; 3; 1] in
    let mlp_in = Tensor.cat [last_c_adjs; x_o_pair] ~dim:(-1) in
    let mlp_in_shape = Tensor.shape mlp_in in
    let mlp_out = MLPModule.forward mlp_mod (Tensor.view mlp_in ~size:[-1; List.nth mlp_in_shape ~-1]) in
    let new_adjs = Tensor.view mlp_out ~size:[List.nth mlp_in_shape 0; List.nth mlp_in_shape 1; List.nth mlp_in_shape 2; -1] |> Tensor.permute ~dims:[0; 3; 1; 2] in
    let new_adjs = Tensor.(new_adjs + Tensor.transpose new_adjs ~dim0:(-1) ~dim1:(-2)) in
    let new_adjs = mask_adjs new_adjs node_flags in
    x_o, new_adjs
end
