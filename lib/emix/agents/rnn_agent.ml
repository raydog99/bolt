open Torch

module RNNAgent : sig
  type t
  type args = {
    rnn_hidden_dim : int;
    n_actions : int;
  }

  val create : int -> args -> t
  val init_hidden : t -> Tensor.t
  val forward : t -> Tensor.t -> Tensor.t -> Tensor.t * Tensor.t
end = struct
  type t = {
    fc1 : Torch.torch_nn_linear;
    rnn : Torch.torch_nn_gru_cell;
    fc2 : Torch.torch_nn_linear;
    args : args;
  }

  and args = {
    rnn_hidden_dim : int;
    n_actions : int;
  }

  let create input_shape args =
    let fc1 = Torch.nn_linear input_shape args.rnn_hidden_dim in
    let rnn = Torch.nn_gru_cell args.rnn_hidden_dim args.rnn_hidden_dim in
    let fc2 = Torch.nn_linear args.rnn_hidden_dim args.n_actions in
    { fc1; rnn; fc2; args }

  let init_hidden agent =
    Torch.zeros agent.fc1.weight#size#(0, agent.args.rnn_hidden_dim)

  let forward agent inputs hidden_state =
    let x = Torch.relu (Torch.forward_linear agent.fc1 inputs) in
    let h_in = hidden_state |> Torch.reshape ~sizes:[| -1; agent.args.rnn_hidden_dim |] in
    let h = Torch.forward_gru_cell agent.rnn x h_in in
    let q = Torch.forward_linear agent.fc2 h in
    q, h
end