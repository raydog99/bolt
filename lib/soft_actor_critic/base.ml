open Torch

module SoftActorCritic = struct
  type params = {
    psi: Tensor.t;
    psi_bar: Tensor.t;
    theta: Tensor.t array;
    phi: Tensor.t;
  }

  let initialize () =
    let psi = Tensor.randn [1] in
    let psi_bar = Tensor.randn [1] in
    let theta = Array.init 2 (fun _ -> Tensor.randn [1]) in
    let phi = Tensor.randn [1] in
    { psi; psi_bar; theta; phi }

  let sample_action phi s_t =
    Tensor.randn [1]

  let sample_next_state s_t a_t =
    Tensor.randn [1]

  let reward s_t a_t =
    Tensor.randn []

  let environment_step params s_t =
    let a_t = sample_action params.phi s_t in
    let s_t_plus_1 = sample_next_state s_t a_t in
    let r_t = reward s_t a_t in
    (s_t, a_t, r_t, s_t_plus_1)

  let j_v psi =
    Tensor.randn []

  let j_q theta =
    Tensor.randn []

  let j_pi phi =
    Tensor.randn []

  let gradient_step params lambda_v lambda_q lambda_pi tau =
    let psi' = Tensor.(params.psi - lambda_v * grad (j_v params.psi) [params.psi]) in
    let theta' = Array.map (fun theta_i ->
      Tensor.(theta_i - lambda_q * grad (j_q theta_i) [theta_i])
    ) params.theta in
    let phi' = Tensor.(params.phi - lambda_pi * grad (j_pi params.phi) [params.phi]) in
    let psi_bar' = Tensor.(tau * params.psi + (Scalar.f 1. - tau) * params.psi_bar) in
    { psi = psi'; psi_bar = psi_bar'; theta = theta'; phi = phi' }

  let train num_iterations num_env_steps lambda_v lambda_q lambda_pi tau =
    let params = initialize () in
    let rec iterate params iter =
      if iter = num_iterations then params
      else
        let rec env_step params step d =
          if step = num_env_steps then d
          else
            let s_t = Tensor.randn [1] in (* Assume random initial state *)
            let (s_t, a_t, r_t, s_t_plus_1) = environment_step params s_t in
            let d' = (s_t, a_t, r_t, s_t_plus_1) :: d in
            env_step params (step + 1) d'
        in
        let _d = env_step params 0 [] in
        let params' = gradient_step params lambda_v lambda_q lambda_pi tau in
        iterate params' (iter + 1)
    in
    iterate params 0

  let run num_iterations num_env_steps lambda_v lambda_q lambda_pi tau =
    train num_iterations num_env_steps lambda_v lambda_q lambda_pi tau
end