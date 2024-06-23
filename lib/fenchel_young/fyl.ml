open Torch

module ConjugateFunction : sig
  val forward : Tensor.t -> Tensor.t -> (Tensor.t -> Tensor.t) -> Tensor.t
  val backward : Tensor.t -> Tensor.t * Tensor.t * Tensor.t
end = struct
  let forward theta grad omega =
    let ctx = Torch.no_grad in
    Torch.sum (Tensor.mul theta grad) ~dim:(-1) - omega grad

  let backward grad_output = Tensor.(grad_output * grad_output.unsqueeze ~dim:(-1)), Tensor.empty, Tensor.empty
end

module FYLoss : sig
  type weights = Average

  val create : weights -> unit
  val forward : t -> t -> t
end = struct
  type weights = Average

  let create weights = function
    | Average -> ()

  let forward theta y_true = function
    let y_pred = predict theta

    let ret = ConjugateFunction.forward theta y_pred function
      if len y_true.shape = 2
        ret += Omega y_true
           approach