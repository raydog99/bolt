open Torch

module ImageSynthesis = struct
  type params = {
    w: Tensor.t;
    i_tilde: Tensor.t;
  }

  let initialize m_tilde =
    let w = Tensor.zeros [1] in
    let i_tilde = Tensor.zeros [m_tilde; 28; 28] in (* Assuming 28x28 images *)
    { w; i_tilde }

  let langevin_dynamics i_tilde w l step_size =
    let rec loop i step =
      if step = l then i
      else
        let noise = Tensor.(randn_like i * sqrt (Scalar.f (2.0 *. step_size))) in
        let grad = Tensor.grad_of_fn (fun i -> f i w) i in
        let i_next = Tensor.(i + step_size * grad + noise) in
        loop i_next (step + 1)
    in
    loop i_tilde 0

  let f i w =
    Tensor.(sum (i * w))

  let calculate_h_obs training_images w =
    let m = Tensor.shape training_images |> List.hd in
    let grad_sum = Tensor.sum (Tensor.grad_of_fn (fun w -> 
      Tensor.mean (Tensor.map (fun img -> f img w) training_images))
    w) in
    Tensor.(grad_sum / float m)

  let calculate_h_syn i_tilde w =
    let m_tilde = Tensor.shape i_tilde |> List.hd in
    let grad_sum = Tensor.sum (Tensor.grad_of_fn (fun w -> 
      Tensor.mean (Tensor.map (fun img -> f img w) i_tilde))
    w) in
    Tensor.(grad_sum / float m_tilde)

  let update params training_images l step_size learning_rate =
    let i_tilde' = langevin_dynamics params.i_tilde params.w l step_size in
    let h_obs = calculate_h_obs training_images params.w in
    let h_syn = calculate_h_syn i_tilde' params.w in
    let w' = Tensor.(params.w + learning_rate * (h_obs - h_syn)) in
    { w = w'; i_tilde = i_tilde' }

  let train training_images m_tilde l t step_size learning_rate =
    let params = initialize m_tilde in
    let rec loop params iter =
      if iter = t then params
      else
        let params' = update params training_images l step_size learning_rate in
        loop params' (iter + 1)
    in
    loop params 0

  let run training_images m_tilde l t step_size learning_rate =
    let final_params = train training_images m_tilde l t step_size learning_rate in
    (final_params.w, final_params.i_tilde)
end