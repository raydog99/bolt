open Core

let energy_training ?(step_size=0.1) ?(num_steps=10) ?(alpha=0.01) ~data_dist ~replay_buffer_size =
  let buffer = Array.create ~len:replay_buffer_size None in
  let rec loop theta buffer =
    let x_pos = Distributions.draw data_dist in
    let x_0 =
      if Random.bool () && Array.exists buffer ~f:(Option.is_some)
      then Array.findi buffer ~f:(Option.is_some) |> fst |> Array.get buffer |> Option.value
      else Distributions.draw Uniform.unit_d
    in
    let x_neg =
      let rec langevin_dynamics x k =
        if k = num_steps then x
        else
          let noise = Gaussian.sample Gaussian.unit in
          let grad_x = Energy.gradient ~theta x in
          let x' = x -. step_size *. grad_x +. sqrt step_size *. noise in
          langevin_dynamics x' (k + 1)
      in
      langevin_dynamics x_0 0 |> Fn.id
    in
    let theta' =
      let loss =
        alpha *. (Energy.eval ~theta x_pos ** 2. +. Energy.eval ~theta x_neg ** 2.)
        +. Energy.eval ~theta x_pos -. Energy.eval ~theta x_neg
      in
      Optimization.update ~theta ~loss
    in
    let buffer =
      Array.concat
        [ buffer
        ; [| Some x_neg |]
        ]
      |> Array.slice ~len:replay_buffer_size
    in
    loop theta' buffer
  in
  loop Energy.init_params buffer