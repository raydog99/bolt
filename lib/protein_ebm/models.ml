open Torch

module AbstractRotomerModel : sig
  type t

  val create : int -> t
  val embed_atoms : t -> Tensor.t -> Tensor.t
end = struct
  type t = {
    element_embed : Tensor.t;
    amino_embed : Tensor.t;
    position_embed : Tensor.t;
    xyz_map : Tensor.t;
  }

  let create scale_factor =
    let element_embed = nn_embedding 5 (28 * scale_factor) in
    let amino_embed = nn_embedding 20 (28 * scale_factor) in
    let position_embed = nn_embedding 21 (28 * scale_factor) in
    let xyz_map = nn_linear 3 (172 * scale_factor) in
    { element_embed; amino_embed; position_embed; xyz_map }

  let embed_atoms model x =
    let res_idx = Tensor.long x.(.., 0) in
    let atom_idx = Tensor.long x.(.., 1) in
    let count_idx = Tensor.long x.(.., 2) in

    let res_encode = Tensor.embedding_lookup model.amino_embed res_idx in
    let atom_encode = Tensor.embedding_lookup model.element_embed atom_idx in
    let amino_ordinal_encode = Tensor.embedding_lookup model.position_embed count_idx in

    let xyz_encode = Tensor.relu @@ Tensor.linear model.xyz_map (Tensor.narrow ~dim:2 x 3 3) in

    Tensor.cat ~dim:2 [ res_encode; atom_encode; amino_ordinal_encode; xyz_encode ]
end

module RotomerFCModel : sig
  type t

  val create : Args.t -> t
  val forward : t -> Tensor.t -> Tensor.t
end = struct
  type t = {
    base_model : AbstractRotomerModel.t;
    embed : Tensor.t;
    layers : Tensor.t ModuleList.t;
    energy : Tensor.t;
  }

  let create args =
    let base_model = AbstractRotomerModel.create 1 in
    let embed_size = args.max_size * 256 in
    let embed = nn_linear embed_size 1024 in
    let layers = ModuleList.init 3 ~f:(fun _ -> nn_linear 1024 1024) in
    let energy = nn_linear 1024 1 in
    { base_model; embed; layers; energy }

  let forward model x =
    let embedded_x = AbstractRotomerModel.embed_atoms model.base_model x in
    let flattened_x = Tensor.view ~size:[|-1; 1024|] embedded_x in
    let x = Tensor.relu @@ Tensor.linear model.embed flattened_x in
    let x = ModuleList.fold_left (fun acc layer -> Tensor.relu @@ Tensor.linear layer acc) x model.layers in
    Tensor.linear model.energy x
end