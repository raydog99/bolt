open Torch

let rho s = Tensor.clamp s ~min:0. ~max:1.
(* let rho s = Tensor.sigmoid (Tensor.(4. * s - 2.)) *)

module Network = struct
  type t = {
    path : string;
    biases : Tensor.t list;
    weights : Tensor.t list;
    hyperparameters : (string, float) Hashtbl.t;
    training_curves : (string, float list) Hashtbl.t;
    external_world : External_World.t;
    persistent_particles : Tensor.t list;
    index : Tensor.t;
    x_data : Tensor.t;
    y_data : Tensor.t;
    y_data_one_hot : Tensor.t;
    layers : Tensor.t list;
    mutable change_mini_batch_index : (int -> unit);
    mutable measure : (unit -> float * float * float);
    mutable free_phase : (int -> float -> unit);
    mutable weakly_clamped_phase : (int -> float -> float -> float list -> float list);
  }

  let initialize_layer n_in n_out =
    let rng = Random.State.make_self_init () in
    Tensor.uniform ~low:(-.(sqrt (6. /. float_of_int (n_in + n_out)))) 
                   ~high:(sqrt (6. /. float_of_int (n_in + n_out)))
                   [n_in; n_out]

  let load_params path hyperparameters =
    if Sys.file_exists path then
      let file = open_in_bin path in
      let biases_values, weights_values, hyperparameters, training_curves = Marshal.from_channel file in
      close_in file;
      (biases_values, weights_values, hyperparameters, training_curves)
    else
      let layer_sizes = [28*28] @ hyperparameters @ [10] in
      let biases_values = List.map (fun size -> Tensor.zeros [size]) layer_sizes in
      let weights_values = List.map2 initialize_layer (List.tl layer_sizes) layer_sizes in
      let training_curves = Hashtbl.create 2 in
      Hashtbl.add training_curves "training error" [];
      Hashtbl.add training_curves "validation error" [];
      (biases_values, weights_values, hyperparameters, training_curves)

  let create name hyperparameters =
    let path = name ^ ".save" in
    let biases, weights, hyperparameters, training_curves = load_params path hyperparameters in
    let external_world = External_World.create () in
    let dataset_size = External_World.size_dataset external_world in
    let layer_sizes = [28*28] @ hyperparameters @ [10] in
    let values = List.map (fun size -> Tensor.zeros [dataset_size; size]) (List.tl layer_sizes) in
    let persistent_particles = List.map Tensor.of_list values in
    let batch_size = Hashtbl.find hyperparameters "batch_size" in
    let index = Tensor.scalar_int 0 in
    let x_data = Tensor.narrow (External_World.x external_world) 0 ~start:0 ~length:batch_size in
    let y_data = Tensor.narrow (External_World.y external_world) 0 ~start:0 ~length:batch_size in
    let y_data_one_hot = Tensor.of_list (List.init batch_size (fun i -> List.init 10 (fun j -> if j = Tensor.get y_data [i] then 1. else 0.))) in
    let layers = x_data :: (List.map (fun particle -> Tensor.narrow particle 0 ~start:0 ~length:batch_size) persistent_particles) in
    {
      path;
      biases;
      weights;
      hyperparameters;
      training_curves;
      external_world;
      persistent_particles;
      index;
      x_data;
      y_data;
      y_data_one_hot;
      layers;
      change_mini_batch_index = (fun _ -> ());
      measure = (fun () -> (0., 0., 0.));
      free_phase = (fun _ _ -> ());
      weakly_clamped_phase = (fun _ _ _ _ -> []);
    }

  let save_params t =
    let file = open_out_bin t.path in
    Marshal.to_channel file (List.map Tensor.to_list t.biases, List.map Tensor.to_list t.weights, t.hyperparameters, t.training_curves) [];
    close_out file

  let build_change_mini_batch_index t =
    let index_new = Tensor.scalar_int 0 in
    let change_mini_batch_index = fun i -> Tensor.copy_ t.index (Tensor.scalar_int i) in
    t.change_mini_batch_index <- change_mini_batch_index

  let energy t layers =
    let squared_norm = List.fold_left (fun acc layer -> Tensor.add acc (Tensor.sum (Tensor.pow (rho layer) 2.))) (Tensor.scalar_float 0.) layers in
    let linear_terms = List.fold_left2 (fun acc layer bias -> Tensor.add acc (Tensor.dot (rho layer) bias)) (Tensor.scalar_float 0.) layers t.biases in
    let quadratic_terms = List.fold_left3 (fun acc pre W post -> Tensor.add acc (Tensor.dot (Tensor.dot (rho pre) W) (rho post))) (Tensor.scalar_float 0.) (List.tl layers) t.weights (List.tl (List.tl layers)) in
    Tensor.add (Tensor.add squared_norm linear_terms) quadratic_terms

  let cost t layers =
    Tensor.sum (Tensor.pow (Tensor.sub (List.hd (List.rev layers)) t.y_data_one_hot) 2.)

  let total_energy t layers beta =
    Tensor.add (energy t layers) (Tensor.mul (Tensor.scalar_float beta) (cost t layers))

  let build_measure t =
    let E = Tensor.mean (energy t t.layers) in
    let C = Tensor.mean (cost t t.layers) in
    let y_prediction = Tensor.argmax (List.hd (List.rev t.layers)) in
    let error = Tensor.mean (Tensor.neq y_prediction t.y_data) in
    t.measure <- (fun () -> (Tensor.float_value E, Tensor.float_value C, Tensor.float_value error))

  let build_free_phase t =
    let n_iterations = Tensor.scalar_int 0 in
    let epsilon = Tensor.scalar_float 0. in
    let step layers =
      let E_sum = Tensor.sum (energy t layers) in
      let layers_dot = Tensor.grad ~target:E_sum layers in
      List.map2 (fun layer dot -> Tensor.clip (Tensor.add layer (Tensor.mul epsilon dot)) ~min:0. ~max:1.) layers layers_dot
    in
    let layers, updates = Tensor.scan ~f:step ~initial:t.layers ~n_steps:n_iterations in
    let layers_end = List.map List.last_exn layers in
    t.persistent_particles <- List.map2 (fun particle layer_end -> Tensor.copy_ particle layer_end) t.persistent_particles layers_end;
    t.free_phase <- (fun n_iterations_val epsilon_val -> Tensor.copy_ n_iterations (Tensor.scalar_int n_iterations_val); Tensor.copy_ epsilon (Tensor.scalar_float epsilon_val); updates ())

  let build_weakly_clamped_phase t =
    let n_iterations = Tensor.scalar_int 0 in
    let epsilon = Tensor.scalar_float 0. in
    let beta = Tensor.scalar_float 0. in
    let alphas = List.map (fun _ -> Tensor.scalar_float 0.) t.weights in
    let step layers =
      let F_sum = Tensor.sum (total_energy t layers beta) in
      let layers_dot = Tensor.grad ~target:F_sum layers in
      List.map2 (fun layer dot -> Tensor.clip (Tensor.add layer (Tensor.mul epsilon dot)) ~min:0. ~max:1.) layers layers_dot
    in
    let layers, updates = Tensor.scan ~f:step ~initial:t.layers ~n_steps:n_iterations in
    let layers_weakly_clamped = List.map List.last_exn layers in
    let E_mean_free = Tensor.mean (energy t t.layers) in
    let E_mean_weakly_clamped = Tensor.mean (energy t layers_weakly_clamped) in
    let biases_dot = Tensor.grad ~target:(Tensor.div (Tensor.sub E_mean_weakly_clamped E_mean_free) beta) t.biases ~consider_constant:layers_weakly_clamped in
    let weights_dot = Tensor.grad ~target:(Tensor.div (Tensor.sub E_mean_weakly_clamped E_mean_free) beta) t.weights ~consider_constant:layers_weakly_clamped in
    let biases_new = List.map3 (fun bias alpha dot -> Tensor.sub bias (Tensor.mul alpha dot)) t.biases alphas biases_dot in
    let weights_new = List.map3 (fun weight alpha dot -> Tensor.sub weight (Tensor.mul alpha dot)) t.weights alphas weights_dot in
    List.iter2 (fun bias bias_new -> Tensor.copy_ bias bias_new) t.biases biases_new;
    List.iter2 (fun weight weight_new -> Tensor.copy_ weight weight_new) t.weights weights_new;
    let delta_log = List.map2 (fun W W_new -> Tensor.div (Tensor.sqrt (Tensor.mean (Tensor.pow (Tensor.sub W_new W) 2.))) (Tensor.sqrt (Tensor.mean (Tensor.pow W 2.)))) t.weights weights_new in
    t.weakly_clamped_phase <- (fun n_iterations_val epsilon_val beta_val alphas_val -> Tensor.copy_ n_iterations (Tensor.scalar_int n_iterations_val); Tensor.copy_ epsilon (Tensor.scalar_float epsilon_val); Tensor.copy_ beta (Tensor.scalar_float beta_val); List.iter2 (fun alpha alpha_val -> Tensor.copy_ alpha (Tensor.scalar_float alpha_val)) alphas alphas_val; delta_log)
end