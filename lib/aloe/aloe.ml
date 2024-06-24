open Torch

module ALOE = struct
  type t = {
    mutable f: Tensor.t -> Tensor.t;
    mutable q: Tensor.t -> Tensor.t -> Tensor.t;
  }

  let create () = {
    f = (fun x -> x);
    q = (fun x y -> x);
  }

  let sample t x_hat =
    let x_tilde = t.q x_hat (t.f x_hat) in
    (x_hat, x_tilde)

  let update_f t x x_tilde =
    let grad_f_x = Tensor.grad (t.f x) in
    let grad_f_x_tilde = Tensor.grad (t.f x_tilde) in
    t.f <- (fun y -> t.f y - grad_f_x + grad_f_x_tilde)

  let update_q t =
    ()

  let train t observations =
    List.iter (fun x ->
      let x_hat, x_tilde = sample t x in
      update_f t x_hat x_tilde;
      update_q t
    ) observations
end

module Sampler = struct
  let sample q x_hat k f_x_hat =
    q x_hat (Tensor.mul_scalar f_x_hat k)
end