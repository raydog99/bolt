open Base
open Torch

module type ACNNPredictorSig = sig
  type t

  val create :
    in_size:int -> hidden_sizes:int list -> weight_init_stddevs:float list
    -> dropouts:float list -> features_to_use:int list option -> num_tasks:int -> t

  val forward :
    t -> batch_size:int -> frag1_node_indices_in_complex:int list
    -> frag2_node_indices_in_complex:int list
    -> ligand_conv_out:float array array -> protein_conv_out:float array array
    -> complex_conv_out:float array array -> float array array
end

module ACNNPredictor : ACNNPredictorSig = struct
  type t = {
    project : Torch.Tensor.t array;
    fea_layer : Torch.Tensor.t array;
  }

  let create ~in_size ~hidden_sizes ~weight_init_stddevs ~dropouts ~features_to_use ~num_tasks =
    let in_size =
      match features_to_use with
      | None -> in_size
      | Some features -> in_size * List.length features
    in
    let modules =
      List.fold_left
        (fun modules h ->
          let linear_layer = Torch.nn.linear in_size h in
          Torch.nn.init.truncated_normal_ linear_layer.weight ~stddev:(List.hd_exn weight_init_stddevs);
          modules @ [ linear_layer; Torch.nn.ReLU (); Torch.nn.Dropout (List.hd_exn dropouts) ];
          in_size <- h;
          modules)
        []
        hidden_sizes
    in
    let linear_layer = Torch.nn.linear in_size num_tasks in
    Torch.nn.init.truncated_normal_ linear_layer.weight ~stddev:(List.hd_exn weight_init_stddevs);
    let project = Torch.nn.Sequential modules in
    let fea_layer = Torch.nn.MLP ~num_layers:1 ~input_dim:1922 ~hidden_dim:2048 ~output_dim:256 in
    { project; fea_layer }

  let forward t ~batch_size ~frag1_node_indices_in_complex ~frag2_node_indices_in_complex
      ~ligand_conv_out ~protein_conv_out ~complex_conv_out =
    let ligand_feats = Torch.Tensor.project ligand_conv_out t.project in
    let protein_feats = Torch.Tensor.project protein_conv_out t.project in
    let complex_feats = Torch.Tensor.project complex_conv_out t.project in

    let ligand_energy = Torch.Tensor.reshape ligand_feats [ batch_size; -1 ] in
    let protein_energy = Torch.Tensor.reshape protein_feats [ batch_size; -1 ] in

    let complex_ligand_energy =
      Torch.Tensor.reshape complex_feats.(frag1_node_indices_in_complex) [ batch_size; -1 ]
    in
    let complex_protein_energy =
      Torch.Tensor.reshape complex_feats.(frag2_node_indices_in_complex) [ batch_size; -1 ]
    in
    let complex_energy = Torch.Tensor.cat [ complex_ligand_energy; complex_protein_energy ] ~dim:-1 in

    let fea = Torch.Tensor.cat [ ligand_energy; protein_energy; complex_energy ] ~dim:-1 in
    Torch.Tensor.fea_layer fea
end