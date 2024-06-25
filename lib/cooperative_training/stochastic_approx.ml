open Torch

module StochasticApproximation = struct
  type params = {
    theta: Tensor.t;
    y_tilde: Tensor.t;
  }

  let initialize n =
    let theta = Tensor.randn [1] in
    let y_tilde = Tensor.randn [n] in
    { theta; y_tilde }

  let langevin_dynamics y_tilde theta l_p step_size =
    let rec loop y i =
      if i = l_p then y
      else
        let noise = Tensor.(randn_like y * sqrt (Scalar.f 2.0 * step_size)) in
        let grad = Tensor.grad_of_fn (fun y -> Tensor.sum (energy_function y theta)) y in
        let y_next = Tensor.(y - step_size * grad + noise) in
        loop y_next (i + 1)
    in
    loop y_tilde 0

  let energy_function y theta =
    Tensor.(y * theta)

  let compute_l_p_prime theta y y_tilde =
    let energy_data = Tensor.mean (energy_function y theta) in
    let energy_model = Tensor.mean (energy_function y_tilde theta) in
    Tensor.(energy_data - energy_model)

  let update params y l_p t T learning_rate =
    let y_tilde' = langevin_dynamics params.y_tilde params.theta l_p (1.0 /. float_of_int T) in
    let l_p_prime = compute_l_p_prime params.theta y y_tilde' in
    let theta' = Tensor.(params.theta + learning_rate * l_p_prime) in
    { theta = theta'; y_tilde = y_tilde' }

  let train y l_p T =
    let n = Tensor.shape y |> List.hd in
    let params = initialize n in
    let rec loop params t =
      if t = T then params
      else
        let learning_rate = 0.01 /. sqrt (1.0 +. float_of_int t) in
        let params' = update params y l_p t T learning_rate in
        loop params' (t + 1)
    in
    loop params 0

  let run y l_p T =
    let final_params = train y l_p T in
    (final_params.theta, final_params.y_tilde)
end