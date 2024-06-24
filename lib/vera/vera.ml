open Torch

module VERA = struct
  type t = {
    mutable theta: Tensor.t;
    mutable phi: Tensor.t;
    mutable eta: Tensor.t;
    lambda: float;
    gamma: float;
  }

  let create theta phi eta lambda gamma =
    { theta; phi; eta; lambda; gamma }

  let sample_minibatch () =
    Tensor.randn [10; 784]

  let generate_minibatch t x z =
    Tensor.randn [10; 784]

  let compute_elbo t z0 x_g =
    Tensor.randn []

  let compute_log_f t x =
    Tensor.randn []

  let sample_zi t z0 k =
    List.init k (fun _ -> Tensor.randn [10; 100])

  let compute_score t x zi z0 =
    Tensor.randn [10; 784]

  let compute_entropy_gradient t s x_g =
    Tensor.randn [10; 784]

  let train t =
    while true do
      let x = sample_minibatch () in
      let z = Tensor.randn [10; 100] in
      let x_g, z0 = generate_minibatch t x z in

      let elbo = compute_elbo t z0 x_g in
      t.eta <- Optimizer.Adam.step t.eta elbo;

      let log_f_x = compute_log_f t x in
      let log_f_x_g = compute_log_f t x_g in
      let grad_log_f_x = Tensor.grad log_f_x in
      let ebm_loss = Tensor.(
        sub log_f_x log_f_x_g +
        (mul_scalar (norm grad_log_f_x) (t.gamma ** 2.))
      ) in
      t.theta <- Optimizer.Adam.step t.theta ebm_loss;

      let zi = sample_zi t z0 5 in
      let s = compute_score t x zi z0 in
      let g = compute_entropy_gradient t s x_g in
      
      let generator_loss = Tensor.(
        add (grad (compute_log_f t x_g))
            (mul_scalar g t.lambda)
      ) in
      t.phi <- Optimizer.Adam.step t.phi generator_loss;

    done;
    t.theta

end