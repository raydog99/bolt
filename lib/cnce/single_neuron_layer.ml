open Torch

module SingleNeuronLayer = struct
  type t = {
    mutable w : Tensor.t;
    mutable b : Tensor.t;
    mutable indices : int list;
    mutable layer_dimensions : int * int;
    mutable is_initialized : bool;
    mutable activator : string;
    mutable hidden : (Tensor.t option * Tensor.t option * Tensor.t option * Tensor.t option * Tensor.t option);
  }

  let create ?(init = `Empty) () = 
    let w, b, indices, activator, layer_dimensions = 
      match init with
      | `Empty -> Tensor.empty, Tensor.empty, [], "", (0, 0)
      | `SingleNeuronLayer { w; b; indices; activator; layer_dimensions; _ } -> w, b, indices, activator, layer_dimensions
      | `File path -> 
        let data = Mat.load path in
        Mat.(data.savedimagemodel.W, data.savedimagemodel.b, data.savedimagemodel.indices, data.savedimagemodel.activator, data.savedimagemodel.layer_dimensions)
      | `Cell (ld, indices, activator) -> 
        let t = Tensor.zeros [|snd ld; fst ld|] in
        t, Tensor.zeros [|snd ld|], indices, activator, ld
      | `Struct { layer_dimensions; indices; _ } -> 
        let t = Tensor.zeros [|snd layer_dimensions; fst layer_dimensions|] in
        t, Tensor.zeros [|snd layer_dimensions|], indices, "", layer_dimensions
    in
    {
      w;
      b;
      indices;
      activator;
      layer_dimensions;
      is_initialized = false;
      hidden = (None, None, None, None, None);
    }

  let check_if_initialized model = 
    model.is_initialized <- not (Tensor.is_empty model.w || Tensor.is_empty model.b || List.length model.indices = 0 || model.layer_dimensions = (0, 0))

  let initialize_model model layer_dimensions indices = 
    if List.length indices <> 1 then failwith "indices must be a column vector."
    else
      let d, n = layer_dimensions in
      if List.exists (fun i -> i > n) indices then failwith "indices exceeds nNeurons."
      else
        model.w <- Tensor.zeros [|n; d|];
        model.b <- Tensor.zeros [|n|];
        model.indices <- indices;
        model.layer_dimensions <- layer_dimensions;
        model.is_initialized <- true

  let initialize_parameters ?(typ = "all") model = 
    if not model.is_initialized then failwith "SingleNeuronLayer not initialized!"
    else
      let d = fst model.layer_dimensions in
      let n_neurons = match typ with 
        | "all" -> snd model.layer_dimensions
        | "variable" -> List.length model.indices
        | _ -> failwith "Unknown type"
      in
      model.w <- Tensor.(mul_scalar (randn [|n_neurons; d|]) (1. /. sqrt (float_of_int (d - 1))));
      model.b <- Tensor.(neg (add_scalar (rand [|n_neurons|]) 1.))

  let clear_hidden model = 
    model.hidden <- (None, None, None, None, None)

  let save_model model filename = 
    let data = Mat.create () in
    Mat.(set data "savedimagemodel" (Mat.serialize model));
    Mat.save data filename

  let get_idx_w model typ = 
    match typ, model.indices with
    | "all", _ | _, [":"] -> [":"]
    | _, [] -> []
    | "variable", _ -> List.map (fun i -> i - 1) model.indices
    | _ -> failwith "Unknown type."

  let get_bias_inx model typ = 
    match typ, model.indices with
    | "all", _ | _, [":"] -> [":"]
    | _ -> model.indices

  let get_n_parameters model typ = 
    let n_inputs = fst model.layer_dimensions in
    let n_neurons = match typ with
      | "all" -> snd model.layer_dimensions
      | "variable" -> List.length model.indices
      | _ -> failwith "Unknown type"
    in
    n_inputs * n_neurons + n_neurons

  let get_n_neurons model typ = 
    match typ with
    | "all" -> snd model.layer_dimensions
    | "variable" -> List.length model.indices
    | _ -> failwith "Unknown type"

  let get_n_inputs model = 
    fst model.layer_dimensions

  let get_theta model typ = 
    let n_inputs = get_n_inputs model in
    let n_neurons = get_n_neurons model typ in
    let n_parameters = get_n_parameters model typ in
    let theta = Tensor.zeros [|n_parameters|] in
    let w_idx = get_idx_w model typ in
    let b_idx = get_bias_inx model typ in
    let theta = Tensor.copy_ ~src:(Tensor.index_select model.w w_idx) ~dst:theta in
    Tensor.copy_ ~src:(Tensor.index_select model.b b_idx) ~dst:(Tensor.narrow theta 0 (n_inputs * n_neurons) n_neurons)
end