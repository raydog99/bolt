open Torch

module AdaptiveMultiStageDensityRatioEstimation = struct
  type t = {
    t : int;
    k : int;
    l : int;
    theta : Tensor.t;
    phi : Tensor.t array;
    p0 : Tensor.t -> Tensor.t;
  }

  let create t k l initial_prior =
    {
      t;
      k;
      l;
      theta = Tensor.randn [1];
      phi = Array.make k (Tensor.randn [1]);
      p0 = initial_prior;
    }

  let mini_batch x m =
    let n = Tensor.shape x |> List.hd in
    let indices = Tensor.randint ~high:n ~size:[m] 0 in
    Tensor.index_select x ~dim:0 ~index:indices

  let posterior_sampling model x =
    let rec sample_loop z i =
      if i >= model.l then z
      else
        let p_theta_x = Tensor.(exp (model.theta * x)) in
        let p_k_z = model.p0 z in
        let q_k_z = Tensor.(p_theta_x * p_k_z) in
        let z_new = Tensor.multinomial q_k_z ~num_samples:1 in
        sample_loop z_new (i + 1)
    in
    sample_loop (Tensor.randn_like x) 0

  let update_phi model q_k p_k =
    let log_ratio = Tensor.(log (q_k / p_k)) in
    Tensor.mean log_ratio ~dim:[0]

  let update_theta model x z =
    let gradient = Tensor.(mean (x - z) ~dim:[0]) in
    Tensor.(model.theta + gradient * scalar 0.01) (* Simple gradient ascent *)

  let update_prior model =
    let k = model.k in
    let r_phi = model.phi.(k-1) in
    let p_prev = model.p0 in
    fun z -> Tensor.(exp (r_phi * z) * (p_prev z))

  let train model x =
    let rec loop t k model =
      if t >= model.t then model
      else
        let mini_batch_x = mini_batch x model.t in
        let z_samples = List.map (posterior_sampling model) (Tensor.split_n mini_batch_x ~dim:0 (Tensor.shape mini_batch_x |> List.hd)) in
        let z_tensor = Tensor.stack z_samples ~dim:0 in
        let q_k = Tensor.mean z_tensor ~dim:[0] in
        let p_k = model.p0 (Tensor.randn_like q_k) in
        let phi_k = update_phi model q_k p_k in
        let theta = update_theta model mini_batch_x z_tensor in
        let model = { model with phi = Array.set model.phi k phi_k; theta } in
        let model =
          if t mod (model.t / model.k) = 0 then
            let k = k + 1 in
            let p_k = update_prior model in
            { model with k; p0 = p_k }
          else model
        in
        loop (t + 1) k model
    in
    loop 0 0 model

  let estimate model =
    (model.theta, model.phi)
end