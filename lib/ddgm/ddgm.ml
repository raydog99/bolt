open Torch

module EnergyModelParams = struct
  let ndim_input = 28 * 28
  let ndim_output = 10
  let num_experts = 128
  let weight_init_std = 1.
  let weight_initializer = "Normal"
  let nonlinearity = "elu"
  let optimizer = "Adam"
  let learning_rate = 0.001
  let momentum = 0.5
  let gradient_clipping = 10.
  let weight_decay = 0.
end

module GenerativeModelParams = struct
  let ndim_input = 10
  let ndim_output = 28 * 28
  let distribution_output = "universal"
  let weight_init_std = 1.
  let weight_initializer = "Normal"
  let nonlinearity = "relu"
  let optimizer = "Adam"
  let learning_rate = 0.001
  let momentum = 0.5
  let gradient_clipping = 10.
  let weight_decay = 0.
end

module type DeepEnergyModel = sig
  val add_feature_extractor : nn.Sequential.t -> unit
  val add_experts : nn.Sequential.t -> unit
  val add_b : nn.Sequential.t -> unit
  val setup_optimizers : string -> float -> float -> float -> float -> unit
  val to_gpu : unit -> unit
  val load : string -> unit
  val save : string -> unit
end

module type DeepGenerativeModel = sig
  val add_sequence : nn.Sequential.t -> unit
  val setup_optimizers : string -> float -> float -> float -> float -> unit
  val to_gpu : unit -> unit
  val load : string -> unit
  val save : string -> unit
  val compute_entropy : unit -> Tensor.t
  val backprop : Tensor.t -> unit
end

module DDGM (EnergyModel : DeepEnergyModel) (GenerativeModel : DeepGenerativeModel) = struct
  let mutable gpu_enabled = false

  let build_network () =
    build_energy_model ();
    build_generative_model ()

  let build_energy_model () =
    let params = EnergyModelParams in
    EnergyModel.add_feature_extractor (nn.Sequential.of_dict params.feature_extractor);
    EnergyModel.add_experts (nn.Sequential.of_dict params.experts);
    EnergyModel.add_b (nn.Sequential.of_dict params.b);
    EnergyModel.setup_optimizers params.optimizer params.learning_rate params.momentum params.weight_decay params.gradient_clipping

  let build_generative_model () =
    let params = GenerativeModelParams in
    GenerativeModel.add_sequence (nn.Sequential.of_dict params.model);
    GenerativeModel.setup_optimizers params.optimizer params.learning_rate params.momentum params.weight_decay params.gradient_clipping

  let to_gpu () =
    EnergyModel.to_gpu ();
    GenerativeModel.to_gpu ();
    gpu_enabled <- true

  let xp =
    if gpu_enabled then Cuda.Cupy else Np

  let to_variable x =
    let x = Tensor.variable x in
    if gpu_enabled then Tensor.to_device x (Device.cuda_if_available ());
    x

  let to_numpy x =
    if Tensor.is_variable x then
      Tensor.to_cpu x |> Tensor.data
    else if Tensor.is_cuda x then
      Tensor.to_cpu x
    else
      x

  let get_batchsize x =
    Tensor.shape x |> List.hd

  let zero_grads () =
    EnergyModel.zero_grads ();
    GenerativeModel.zero_grads ()

  let compute_energy x_batch test =
    let x_batch = to_variable x_batch in
    EnergyModel.compute_energy x_batch test

  let compute_energy_sum x_batch test =
    let energy, experts = compute_energy x_batch test in
    Tensor.sum energy / (get_batchsize x_batch |> Float.of_int |> Tensor.scalar)

  let compute_entropy () =
    GenerativeModel.compute_entropy ()

  let sample_z batchsize =
    let ndim_z = GenerativeModelParams.ndim_input in
    Tensor.uniform ~shape:[batchsize; ndim_z] ~low:(Scalar.float (-1.0)) ~high:(Scalar.float 1.0) |> Tensor.to_type Float32

  let generate_x batchsize test as_numpy =
    generate_x_from_z (sample_z batchsize) test as_numpy

  let generate_x_from_z z_batch test as_numpy =
    let z_batch = to_variable z_batch in
    let x_batch = GenerativeModel.generate_x z_batch test in
    if as_numpy then to_numpy x_batch else x_batch

  let backprop_energy_model loss =
    EnergyModel.backprop loss

  let backprop_generative_model loss =
    GenerativeModel.backprop loss

  let compute_kld_between_generator_and_energy_model x_batch_negative =
    let energy_negative, experts_negative = compute_energy x_batch_negative false in
    let entropy = compute_entropy () in
    Tensor.sum energy_negative / (get_batchsize x_batch_negative |> Float.of_int |> Tensor.scalar) - entropy

  let load dir =
    if dir = "" then failwith "Directory cannot be empty";
    EnergyModel.load (dir ^ "/energy_model.hdf5");
    GenerativeModel.load (dir ^ "/generative_model.hdf5")

  let save dir =
    if dir = "" then failwith "Directory cannot be empty";
    (try Unix.mkdir dir 0o755 with _ -> ());
    EnergyModel.save (dir ^ "/energy_model.hdf5");
    GenerativeModel.save (dir ^ "/generative_model.hdf5")
end