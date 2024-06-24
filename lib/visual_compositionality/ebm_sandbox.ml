open Torch

(* Helper function to save an image *)
let save_image filename tensor =
  let open Stb_image_write in
  let tensor = Tensor.(tensor * f 255.0) |> Tensor.to_device ~device:Cpu |> Tensor.to_type ~type_:T.uint8 in
  let data = Tensor.to_data tensor ~device:Cpu ~kind:Tensor.Kind.uint8 in
  let shape = Tensor.shape tensor in
  write_png filename ~w:(List.hd shape) ~h:(List.hd (List.tl shape)) ~c:3 ~data

let continual_combine target_vars =
  let n = 64 in
  let x = target_vars.(0)
  and y_pos = target_vars.(1)
  and y_shape = target_vars.(2)
  and y_color = target_vars.(3)
  and x_final = target_vars.(4) in

  let data_corrupt = Tensor.uniform [n; 64; 64; 3] ~low:0. ~high:1. in
  let pos_label = Tensor.uniform [n; 2] ~low:(-1.) ~high:1. in
  let shape_label = Tensor.(repeat (eye 2 |> narrow ~dim:0 ~start:1 ~length:1) ~dims:[n; 1]) in
  let color_label = Tensor.eye 20 |> Tensor.narrow ~dim:0 ~start:2 ~length:18 |> Tensor.index_select ~dim:0 (Tensor.randint ~high:18 ~size:[n] |> Tensor.add_scalar 2) in

  Printf.printf "%s\n" (Tensor.to_string pos_label);

  let feed_dict = [Tensor.Pair(x, data_corrupt); Tensor.Pair(y_pos, pos_label); Tensor.Pair(y_shape, shape_label); Tensor.Pair(y_color, color_label)] in
  let data_corrupt = Tensor.run_forward ~tensors:feed_dict ~outputs:[x_final] |> List.hd in

  save_image "double_continual_large.png" (Tensor.view data_corrupt ~size:[-1; 64; 3]);

  assert false

let composition_figure target_vars =
  let batch = 32 in
  let x = target_vars.(0)
  and y_pos = target_vars.(1)
  and y_shape = target_vars.(2)
  and y_color = target_vars.(3)
  and y_size = target_vars.(4)
  and x_final = target_vars.(5) in

  let data_corrupt = Tensor.(uniform [batch; 64; 64; 3] ~low:0. ~high:1. / f 2. + uniform [batch; 1; 1; 3] ~low:0. ~high:1. / f 2.) in
  let pos_label = Tensor.repeat (Tensor.of_array2 [| [|0.0; -0.8|] |]) ~dims:[batch; 1] in
  let shape_label = Tensor.repeat (Tensor.eye 3 |> Tensor.narrow ~dim:0 ~start:2 ~length:1) ~dims:[batch; 1] in
  let color_label = Tensor.repeat (Tensor.eye 20 |> Tensor.narrow ~dim:0 ~start:15 ~length:1) ~dims:[batch; 1] in
  let size_label = Tensor.repeat (Tensor.of_array2 [| [|0.4|] |]) ~dims:[batch; 1] in

  let feed_dict = [Tensor.Pair(x, data_corrupt); Tensor.Pair(y_pos, pos_label); Tensor.Pair(y_shape, shape_label); Tensor.Pair(y_color, color_label); Tensor.Pair(y_size, size_label)] in
  let data_corrupt = Tensor.run_forward ~tensors:feed_dict ~outputs:[x_final] |> List.hd in

  let data_corrupt = Tensor.flip data_corrupt ~dims:[1] in
  save_image "2_composition_large.png" (Tensor.view data_corrupt ~size:[-1; 64; 3]);

  assert false