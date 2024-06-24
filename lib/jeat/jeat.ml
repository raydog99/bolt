open Torch

module JEAT = struct
  type t = {
    f: Tensor.t -> Tensor.t -> Tensor.t;
    epsilon: float;
    alpha: float;
    k: int;
    rho: float;
    eta: float;
    buffer: Tensor.t array;
  }

  let create f epsilon alpha k rho eta buffer_size =
    { f; epsilon; alpha; k; rho; eta; buffer = Array.make buffer_size Tensor.empty }

  let energy_xy t x y = Tensor.(- (log (exp (t.f x y))))
  let energy_x t x = Tensor.(- (log (sum (exp (t.f x (Tensor.empty))))))

  let generate_adversarial_sample t x y =
    let delta = Tensor.(uniform [Tensor.shape x] ~low:(float_of_int (- t.epsilon)) ~high:t.epsilon) in
    let grad = Tensor.grad (fun x -> energy_xy t x y) x in
    let delta = Tensor.(delta + (f t.epsilon * sign grad)) in
    let delta = Tensor.(clamp delta ~min:(f (- t.epsilon)) ~max:(f t.epsilon)) in
    Tensor.(x + delta)

  let sgld_step t x =
    let grad = Tensor.grad (fun x -> energy_x t x) x in
    let noise = Tensor.(randn_like x * sqrt (f t.alpha)) in
    Tensor.(x - (f (t.alpha /. 2.) * grad) + noise)

  let generate_sgld_sample t =
    let x0 = if Random.float 1.0 < t.rho then Tensor.randn [1] else t.buffer.(Random.int (Array.length t.buffer)) in
    let rec loop x i =
      if i = t.k then x
      else loop (sgld_step t x) (i + 1)
    in
    loop x0 0

  let update_parameters t x_adv y x_k =
    let grad_p_y = Tensor.grad (fun x -> Tensor.sub (energy_xy t x y) (energy_x t x)) x_adv in
    let grad_p = Tensor.grad (fun x -> Tensor.sub (energy_x t x) (energy_x t x_k)) x_adv in
    Tensor.(grad_p_y + grad_p)

  let train t dataset epochs =
    for i = 1 to epochs do
      Array.iter (fun (x, y) ->
        let x_adv = generate_adversarial_sample t x y in
        let x_k = generate_sgld_sample t in
        let grad = update_parameters t x_adv y x_k in
        Tensor.(t.f x y - (f t.eta * grad))
      ) dataset
    done

  let generate t y =
    let x0 = Tensor.randn [Tensor.shape y] in
    let rec loop x i =
      if i = t.k then x
      else
        let grad = Tensor.grad (fun x -> energy_xy t x y) x in
        let noise = Tensor.(randn_like x * sqrt (f t.alpha)) in
        let x_next = Tensor.(x - (f (t.alpha /. 2.) * grad) + noise) in
        loop x_next (i + 1)
    in
    loop x0 0
end