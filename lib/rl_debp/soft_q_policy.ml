open Torch

module SoftQLearning = struct
  type params = {
    theta: Tensor.t;
    phi: Tensor.t;
    theta_target: Tensor.t;
    phi_target: Tensor.t;
  }

  type experience = {
    s_t: Tensor.t;
    a_t: Tensor.t;
    r_t: float;
    s_t_plus_1: Tensor.t;
  }

  let initialize () =
    let theta = Tensor.randn [1] in
    let phi = Tensor.randn [1] in
    { theta; phi; theta_target = theta; phi_target = phi }

  let sample_action phi s_t =
    let xi = Tensor.randn [1] in
    Tensor.(phi * s_t + xi)

  let sample_next_state s_t a_t =
    Tensor.randn [1]

  let reward s_t a_t =
    Random.float 1.0

  let collect_experience params s_t =
    let a_t = sample_action params.phi s_t in
    let s_t_plus_1 = sample_next_state s_t a_t in
    let r_t = reward s_t a_t in
    { s_t; a_t; r_t; s_t_plus_1 }

  let sample_minibatch replay_memory batch_size =
    List.init batch_size (fun _ -> List.nth replay_memory (Random.int (List.length replay_memory)))

  let compute_soft_values theta s_t_plus_1 =
    Tensor.randn [1]

  let compute_q_gradient theta experiences =
    Tensor.randn [1]

  let compute_policy_gradient phi experiences =
    Tensor.randn [1]

  let update params experiences =
    let theta_grad = compute_q_gradient params.theta experiences in
    let phi_grad = compute_policy_gradient params.phi experiences in
    let adam_update tensor grad =
      Tensor.(tensor - grad * Scalar.f 0.001)
    in
    { params with
      theta = adam_update params.theta theta_grad;
      phi = adam_update params.phi phi_grad;
    }

  let train num_epochs num_steps_per_epoch update_interval =
    let params = initialize () in
    let replay_memory = ref [] in
    for epoch = 1 to num_epochs do
      for _ = 1 to num_steps_per_epoch do
        let s_t = Tensor.randn [1] in (* Initial state *)
        let exp = collect_experience params s_t in
        replay_memory := exp :: !replay_memory;
        let minibatch = sample_minibatch !replay_memory 32 in
        let params' = update params minibatch in
        params = params'
      done;
      if epoch mod update_interval = 0 then
        params = { params with theta_target = params.theta; phi_target = params.phi }
    done;
    params

  let run num_epochs num_steps_per_epoch update_interval =
    train num_epochs num_steps_per_epoch update_interval
end