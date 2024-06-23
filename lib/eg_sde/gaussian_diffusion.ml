open Torch

let get_named_beta_schedule schedule_name num_diffusion_timesteps =
  let module T = Owl_base_dense_ndarray.D in
  match schedule_name with
  | "linear" ->
    let scale = 1000. /. float_of_int num_diffusion_timesteps in
    let beta_start = scale *. 0.0001 in
    let beta_end = scale *. 0.02 in
    let linspace = T.linspace beta_start beta_end num_diffusion_timesteps in
    T.to_float64_array linspace

  | "cosine" ->
    let betas_for_alpha_bar num_diffusion_timesteps alpha_bar max_beta =
      let betas = ref [] in
      for i = 0 to num_diffusion_timesteps - 1 do
        let t1 = float_of_int i /. float_of_int num_diffusion_timesteps in
        let t2 = float_of_int (i + 1) /. float_of_int num_diffusion_timesteps in
        let beta = min (1.0 -. alpha_bar t2 /. alpha_bar t1) max_beta in
        betas := !betas @ [beta]
      done;
      T.of_array (Array.of_list !betas)
    in
    betas_for_alpha_bar num_diffusion_timesteps
      (fun t -> Owl_maths.sin (Owl_maths.pi *. (t +. 0.008) /. 1.008) ** 2.)
      0.999